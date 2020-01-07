import os
import math

from pprint import pformat

import torch

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_transformers import (WEIGHTS_NAME, CONFIG_NAME)


def build_runner(opt, model, optimizer, amp):
    return Runner(opt, model, optimizer, amp)


def average_distributed_scalar(scalar, opt):
    """
    Average a scalar over the nodes if we are in distributed training.
    We use this for distributed evaluation.
    """
    if opt.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar,
                            dtype=torch.float,
                            device=opt.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


class Runner(object):
    """

    """
    def __init__(self, opt, model, optimizer, amp=None):
        # Basic attributes.
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.trainer = Engine(self.update)
        self.evaluator = Engine(self.inference)
        self.tb_logger = TensorboardLogger(log_dir=None)
        self.amp = amp

    def run(self, train_iter, valid_iter, train_sampler, valid_sampler):
        # Attach evaluation to enginer
        # self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.evaluator.run(valid_iter))
        # if self.opt.n_epochs < 1:
        #     self.trainer.add_event_handler(Events.COMPLETED, lambda _: self.evaluator.run(valid_iter))

        # Evaluation during training
        @self.trainer.on(Events.ITERATION_COMPLETED)
        def evaluation(engine):
            # if engine.state.iteration % int(0.1 * len(train_iter)) == 0:
            if engine.state.iteration % self.opt.valid_steps == 0:
                self.evaluator.run(valid_iter)

        # Make sure distributed data samplers split the dataset nicely between the distributed processes
        if self.opt.distributed:
            self.trainer.add_event_handler(
                Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
            self.evaluator.add_event_handler(
                Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

        # Linearly decrease the learning rate from lr to zero
        scheduler = PiecewiseLinear(
            self.optimizer, "lr", [(0, self.opt.lr), (self.opt.n_epochs * len(train_iter), 0.0)])

        self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

        # Prepare metrics - note how we compute distributed metrics
        RunningAverage(output_transform=lambda x: x[0]).attach(self.trainer, "loss")
        RunningAverage(output_transform=lambda x: x[1]).attach(self.trainer, "lr")
        metrics = {
            "nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1),
                        output_transform=lambda x: (x[0], x[1]))}
        metrics.update({
            "average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], self.opt)})
        metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])

        for name, metric in metrics.items():
            metric.attach(self.evaluator, name)

        # On the main process:
        # add progress bar, tensorboard, checkpoints and save model,
        # configuration and tokenizer before we start to train
        if self.opt.local_rank in [-1, 0]:
            pbar = ProgressBar(persist=True)
            pbar.attach(self.trainer, metric_names=["loss", "lr"])
            self.evaluator.add_event_handler(
                Events.COMPLETED,
                lambda _: pbar.log_message("Validation: %s" % pformat(self.evaluator.state.metrics)))

            self.tb_logger.attach(
                self.trainer,
                log_handler=OutputHandler(tag="training",
                                          metric_names=["loss"]),
                event_name=Events.ITERATION_COMPLETED)
            self.tb_logger.attach(
                self.trainer,
                log_handler=OptimizerParamsHandler(self.optimizer),
                event_name=Events.ITERATION_STARTED)
            self.tb_logger.attach(
                self.evaluator,
                log_handler=OutputHandler(tag="validation",
                                          metric_names=list(metrics.keys()),
                                          another_engine=self.trainer),
                event_name=Events.EPOCH_COMPLETED)

            checkpoint_handler = ModelCheckpoint(
                self.tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)

            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
                checkpoint_handler,
                {'mymodel': getattr(self.model, 'module', self.model)})
            # "getattr" take care of distributed encapsulation

            torch.save(self.opt, self.tb_logger.writer.logdir + '/model_training_args.bin')
            getattr(self.model, 'module', self.model).config.to_json_file(
                os.path.join(self.tb_logger.writer.logdir, CONFIG_NAME))
            # tokenizer.save_vocabulary(tb_logger.writer.logdir)

        # Run the training
        self.trainer.run(train_iter, max_epochs=self.opt.n_epochs)

        # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
        if self.opt.local_rank in [-1, 0] and self.opt.n_epochs > 0:
            os.rename(checkpoint_handler._saved[-1][1][-1],
                      os.path.join(self.tb_logger.writer.logdir, WEIGHTS_NAME))
            # TODO: PR in ignite to have better access to saved file paths (cleaner)
            self.tb_logger.close()


    # Training function and trainer
    def update(self, engine, batch):
        # inputs = [batch["input_gpt"], batch["label_gpt"]]
        # input_ids, lm_labels = tuple(torch.LongTensor(x).to(self.opt.device) for x in inputs)
        batch = tuple(input_tensor.to(self.opt.device) for input_tensor in batch)
        input_ids, lm_labels, token_type_ids = batch
        self.model.train()
        lm_loss = self.model(input_ids, labels=lm_labels)
        loss = lm_loss / self.opt.gradient_accumulation_steps
        if self.amp is not None:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), self.opt.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.max_norm)
        if engine.state.iteration % self.opt.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), self.optimizer.get_lr()[-1]

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(self, engine, batch):
        self.model.eval()
        with torch.no_grad():
            # inputs = [batch["input_gpt"], batch["label_gpt"]]
            # input_ids, lm_labels = tuple(torch.LongTensor(x).to(self.opt.device) for x in inputs)
            batch = tuple(input_tensor.to(self.opt.device) for input_tensor in batch)
            input_ids, lm_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            lm_logits = self.model(input_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted
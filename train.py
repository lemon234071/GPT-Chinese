# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler, CustomPeriodicEvent
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,
                                  GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  WEIGHTS_NAME, CONFIG_NAME, AdamW)

from od.inputters.tokenization_wb import WBTokenizer
from od.inputters.dataset_wb import WBDataset, WBCollate

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    logger.info("Build train and validation dataloaders")
    train_dataset = WBDataset(args, tokenizer, data_path=args.train_path)
    valid_dataset = WBDataset(args, tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=WBCollate(train_dataset),
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=WBCollate(valid_dataset),
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrian model')
    parser.add_argument("--num_workers", type=int, default=8, help="How many subprocesses to use for data loading")
    parser.add_argument("--warmup_steps", type=int, default=16000, help="")
    parser.add_argument("--valid_steps", type=int, default=2500, help="")
    parser.add_argument("--train_path", type=str, default="./data/toy_train.txt", help="Path of the dataset.")
    parser.add_argument("--valid_path", type=str, default="./data/toy_valid.txt", help="Path of the dataset.")

    parser.add_argument("--model_checkpoint", type=str, default="./pretrain/Cgpt/",
                        help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=25, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d",
                   args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer = WBTokenizer(os.path.join(args.model_checkpoint, "vocab.txt"), split=True)
    if args.load_pretrain:
        model = OpenAIGPTLMHeadModel.from_pretrained(args.model_checkpoint)
    else:
        config = OpenAIGPTConfig.from_json_file(args.model_checkpoint + "config.json")
        model = OpenAIGPTLMHeadModel(config)
    model.to(args.device)

    optimizer = AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        input_ids, lm_labels, token_type_ids = tuple(input_tensor.to(args.device) for input_tensor in batch)
        model.train()
        (lm_loss), *_ = model(input_ids, labels=lm_labels, token_type_ids=token_type_ids)
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            input_ids, lm_labels, token_type_ids = tuple(input_tensor.to(args.device) for input_tensor in batch)
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            lm_logits, *_ = model(input_ids, token_type_ids=token_type_ids)
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    cpe1 = CustomPeriodicEvent(n_iterations=40000)
    cpe1.attach(trainer)
    # Evaluation during training
    @trainer.on(cpe1.Events.ITERATIONS_40000_COMPLETED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        # if engine.state.iteration % args.valid_steps == 0:
        evaluator.run(val_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # noam decrease the learning rate
    # model_size = model.config.n_embd
    model_size = 768
    noam_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (
            model_size ** (-0.5) *
            min(step ** (-0.5), step * args.warmup_steps ** (-1.5))) if step != 0 else 1, last_epoch=411596)
    scheduler = LRScheduler(noam_scheduler)
    # scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=None)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=cpe1.Events.ITERATIONS_40000_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        # Let's define an event every 1000 iterations
        trainer.add_event_handler(cpe1.Events.ITERATIONS_40000_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.logdir,
                                                                     WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()

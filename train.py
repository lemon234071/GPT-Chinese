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
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTConfig, OpenAIGPTLMHeadModel, WEIGHTS_NAME, CONFIG_NAME)

from help.WB_tokenization import WBTokenizer, VOCAB_FILE
from help.dataset_weibolm_txt import WBDataset, WBCollate
from cotk.dataloader import GPTSingleTurnDialog


logger = logging.getLogger(__file__)


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]


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
                              # shuffle=(not args.distributed))
                              shuffle=False)
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=WBCollate(valid_dataset),
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler


def get_cotk_data_loaders(args):
    data_class = GPTSingleTurnDialog.load_class(args.dataset)
    data = data_class(args.datapath,
                      bert_vocab_name=args.vocab_path,
                      min_vocab_times=args.min_vocab_times,
                      max_sent_length=args.max_sent_length)
    # train_iter = data.get_batches("train", batch_size=args.train_batch_size, shuffle=True)
    # valid_iter = data.get_batches("dev", batch_size=args.valid_batch_size, shuffle=False)
    train_sampler, valid_sampler = None, None

    class cotk_loader(torch.utils.data.Dataset):
        def __init__(self, data, datakey, batch_size):
            self.data = data
            self.datakey = datakey
            self.shuffle = False
            # self.shuffle = True if datakey == "train" else False
            self.batch_size = batch_size
            self.tokenizer = data.tokenizer

        def __len__(self):
            return math.ceil(self.data.data_size[self.datakey] / self.batch_size)

        def __getitem__(self, index):
            return self.data.get_batch(self.datakey, [index])

        def __iter__(self):
            for batch in self.data.get_batches(self.datakey, batch_size=self.batch_size, shuffle=self.shuffle):
                yield batch

    train_iter = cotk_loader(data, "train", args.train_batch_size)
    valid_iter = cotk_loader(data, "dev", args.valid_batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_iter) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_iter) if args.distributed else None
    return train_iter, valid_iter, train_sampler, valid_sampler


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GPTOpenSubtitles", help="Dataset.")
    parser.add_argument("--datapath", type=str, default="./data/", help="Path of the dataset.")# resources://OpenSubtitles
    parser.add_argument("--vocab_path", type=str, default="./pretrain/Cgpt/vocab.txt", help="Path of the vocab.")
    parser.add_argument("--min_vocab_times", type=int, default=0, help="")
    parser.add_argument("--max_sent_length", type=int, default=256, help="")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="")
    parser.add_argument("--valid_steps", type=int, default=125, help="")


    parser.add_argument("--train_path", type=str, default="./data/train.txt", help="Path of the dataset.")
    parser.add_argument("--valid_path", type=str, default="./data/valid.txt", help="Path of the dataset.")
    parser.add_argument("--model_checkpoint", type=str, default="./pretrain/Cgpt/",
                        help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=25, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--num_workers", type=int, default=8, help="How many subprocesses to use for data loading")
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrian model')
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
    #tokenizer = WBTokenizer(os.path.join(args.model_checkpoint, VOCAB_FILE), split=True)
    if args.load_pretrain:
        model = OpenAIGPTLMHeadModel.from_pretrained(args.model_checkpoint)
    else:
        config = OpenAIGPTConfig.from_json_file(args.model_checkpoint + "config.json")
        model = OpenAIGPTLMHeadModel(config)

    # tokenizer.set_special_tokens(SPECIAL_TOKENS)
    # model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.to(args.device)

    logger.info("Prepare datasets")
    # train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    train_loader, val_loader, train_sampler, valid_sampler = get_cotk_data_loaders(args)

    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        inputs = [batch["input_gpt"], batch["label_gpt"]]
        input_ids, lm_labels = tuple(torch.LongTensor(x).to(args.device) for x in inputs)
        # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        # input_ids, lm_labels, token_type_ids = batch
        model.train()
        lm_loss = model(input_ids, lm_labels=lm_labels)
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
        return loss.item(), optimizer.get_lr()[-1]

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs = [batch["input_gpt"], batch["label_gpt"]]
            input_ids, lm_labels = tuple(torch.LongTensor(x).to(args.device) for x in inputs)
            # batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            # input_ids, lm_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            lm_logits = model(input_ids)
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

    # Evaluation every during training
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iterations(engine):
        if engine.state.iteration % int(0.1 * len(train_loader)) == 0:
        # if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(val_loader)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    # scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * args.warmup_steps, 0.0)])
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
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        #tokenizer.save_vocabulary(tb_logger.writer.logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.logdir,
                                                                     WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()

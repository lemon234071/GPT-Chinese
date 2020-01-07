#!/usr/bin/env python
import os
# import logging

from pprint import pformat
from argparse import ArgumentParser

import torch
from torch.nn.parallel import DistributedDataParallel
from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIGPTConfig,
                                  GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  WEIGHTS_NAME, CONFIG_NAME, AdamW)

import od.opts as opts
from od.utils.logging import init_logger, logger
from od.inputters.inputter import build_dataloader
from od.inputters.WB_tokenization import WBTokenizer, VOCAB_FILE
from od.runner import build_runner


#logger = logging.getLogger(__file__)
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]


def train(opt):
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    # logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", opt.local_rank)
    # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(opt))

    # Initialize distributed training if needed
    opt.distributed = (opt.local_rank != -1)
    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        opt.device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Building model.")
    model_class = GPT2LMHeadModel if "gpt2" in opt.model_checkpoint else OpenAIGPTLMHeadModel
    model_config = GPT2Config if "gpt2" in opt.model_checkpoint else OpenAIGPTConfig
    tokenizer_class = GPT2Tokenizer if "gpt2" in opt.model_checkpoint else OpenAIGPTTokenizer

    if opt.load_pretrain:
        logger.info("Train from %s." % opt.model_checkpoint)
        model = model_class.from_pretrained(opt.model_checkpoint)
    else:
        logger.info("Train from scratch.")
        assert opt.vocab_path
        config = model_config.from_json_file(opt.model_checkpoint + "config.json")
        model = model_class(config)
    model.to(opt.device)
    tokenizer = WBTokenizer(os.path.join(opt.model_checkpoint, VOCAB_FILE), split=True)
    # add_special_tokens_(model, tokenizer)
    tokenizer.set_special_tokens(SPECIAL_TOKENS)
    # model.set_num_special_tokens(len(SPECIAL_TOKENS))

    # Build optimizer.
    optimizer = AdamW(model.parameters(), lr=opt.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed
    # (order is important, distributed should be the last)
    if opt.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16)
    if opt.distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[opt.local_rank],
                                        output_device=opt.local_rank)

    runner = build_runner(opt, model, optimizer, amp=amp if opt.fp16 else None)

    logger.info("Prepare datasets")
    train_iter, valid_iter, train_sampler, valid_sampler = \
        build_dataloader(opt, tokenizer)

    if opt.device == "cuda":
        logger.info('Starting training on GPU: %s' % opt.local_rank)
    else:
        logger.info('Starting training on CPU, could be very slow')

    runner.run(train_iter, valid_iter, train_sampler, valid_sampler=valid_sampler)


def _get_parser():
    parser = ArgumentParser()
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()

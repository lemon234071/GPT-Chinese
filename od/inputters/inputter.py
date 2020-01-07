# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

from od.inputters.dataset_weibolm_txt import WBDataset, WBCollate
from od.utils.logging import logger


def build_dataloader(args, tokenizer):
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
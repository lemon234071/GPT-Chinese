#!/usr/bin/env python
# coding:utf8

from collections import defaultdict
from itertools import chain
import torch
# from torch.utils.data import Dataset
from od.inputters.dataset_base import Dataset

SPECIAL_TOKENS = ["<Lua heritage>", "<eos>", "madeupword0000", "madeupword0001"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


class WBCollate(object):

    def __init__(self, dataset):
        self.padding = dataset.tokenizer.pad()

    def __call__(self, batch):
        tensor_batch = []
        # TODO
        dataset = self.pad_dataset(batch, padding=self.padding)
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor = tensor.view((-1, ) + tensor.shape[1:])
            tensor_batch.append(tensor)

        return tensor_batch

    @staticmethod
    def pad_dataset(dataset, padding=0):
        """
        Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler.
        :param dataset:
        :param padding:
        :return:
        """
        # ["input_ids", "lm_labels", "token_type_ids"]
        input_ids = [x for instance in dataset for x in instance["input_ids"]]
        token_type_ids = [x for instance in dataset for x in instance["token_type_ids"]]
        lm_labels = [x for instance in dataset for x in instance["lm_labels"]]

        max_l = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            input_ids[i].extend([padding] * (max_l - len(input_ids[i])))
            token_type_ids[i].extend([padding] * (max_l - len(token_type_ids[i])))
            lm_labels[i].extend([-1] * (max_l - len(lm_labels[i])))

        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "lm_labels": lm_labels
                }


class WBDataset(Dataset):

    def __init__(self, args, tokenizer, is_train=False, *inputs, **kwargs):
        super(WBDataset, self).__init__(*inputs, **kwargs)
        self.args = args
        self.data = list()
        self.tokenizer = tokenizer

    # def __len__(self):
    #     return len(self.data)

    def __getitem__(self, index):
        line = self._get_line(index)
        hist_candi = line.strip().split("[SEP]")
        history = hist_candi[0].split("[POST]")
        candidates = hist_candi[1].split("[RESP]")
        return self.process({"history": history, "candidates": candidates})

    def process(self, utterance1):
        tokenizer = self.tokenizer

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        utterance = tokenize(utterance1)
        history = utterance["history"][-(2 * self.args.max_history + 1):]
        pack_instance = defaultdict(list)
        instance, _ = self.build_input_from_segments(history, utterance["candidates"][-1], lm_labels=True)
        for input_name, input_array in instance.items():
            pack_instance[input_name].append(input_array)
        return pack_instance

    def build_input_from_segments(self, history, reply, lm_labels=True, with_eos=True):
        """ Build a sequence of input from 3 segments: persona, history and last reply """
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        instance = {}
        sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        return instance, sequence



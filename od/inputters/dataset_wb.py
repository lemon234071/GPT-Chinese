#!/usr/bin/env python
# coding:utf8
import os
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ["<Lua heritage>", "<eos>", "madeupword0000", "madeupword0001"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]


class WBDataset(Dataset):

    def __init__(self, data, tokenizer, batch_first=True):
        self.data = data
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        return input_ids, token_type_ids, labels


def truncate_seq_pair(max_seq_len, tokens_a, tokens_b):
    """Truncate the sequence of pair, the last token will be removed
    if it is longer than the other.
    Args:
        max_seq_len: max length of target sequence
        tokens_a: first token sequence of the pair
        tokens_b: second token sequence of the pair
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_len - 2:  # for [CLS] and [SEP] tokens
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def find_first_sublist(main_list, sub_list):
    """Find the start and end indexes of sublist in main list
    Args:
        main_list: the main list.
        sub_list: the sublist.
    Return:
        the start and end indexes of sub_list in main_list
    """
    sub_len = len(sub_list)
    for i, _ in enumerate(main_list):
        if main_list[i: i + sub_len] == sub_list:
            return (i, i + sub_len - 1)


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))


class DatasetBase(Dataset):
    """The base class of different task dataset
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line


class WBdistributeDataset(DatasetBase):

    def __init__(self, args, tokenizer, *inputs, **kwargs):
        super(WBdistributeDataset, self).__init__(*inputs, **kwargs)
        self.args = args
        self.tokenizer = tokenizer

    # def __len__(self):
    #     return len(self.data)

    def __getitem__(self, index):
        line = self._get_line(index)
        hist_candi = line.strip().split("[SEP]")
        history = hist_candi[0].split("[POST]")
        candidates = hist_candi[1].split("[RESP]")
        return self.process({"history": history, "candidates": candidates})

    def process(self, text_dict):
        tokenizer = self.tokenizer

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        utterance = tokenize(text_dict)
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
        instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _
                                              in s]
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
        return instance, sequence

    def collate(self, batch):
        tensor_batch = []
        # TODO
        dataset = self.pad_dataset(batch, padding=self.tokenizer.pad())
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            tensor = tensor.view((-1,) + tensor.shape[1:])
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
                "lm_labels": lm_labels}

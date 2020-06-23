#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT) Task2.
This script evaluates the distinct[1] of the submitted model.
This uses a the version of the dataset which does not contain the "Golden Response" .
Leaderboard scores will be run in the same form but on a hidden test set.

reference:

[1] Li, Jiwei, et al. "A diversity-promoting objective function for neural conversation models."
    arXiv preprint arXiv:1510.03055 (2015).

This requires each team to implement the following function:
def gen_response(self, contexts):
    return a list of responses for each context
    Arguments:
    contexts -- a list of context, each context contains dialogue histories and personal profiles of every speaker
    Returns a list, where each element is the response of the corresponding context
"""
import json
import sys
import codecs
from collections import defaultdict
import numpy as np


def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return [json.loads(i) for i in contents]


def count_ngram_avg(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    tokens = 0
    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
            tokens += 1
    return len(ngram) / tokens


def eval_distinct(hyps_resp):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    # num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram_avg(hyps_resp, 1)
    dist2 = count_ngram_avg(hyps_resp, 2)

    return [dist1, dist2]


def calc_diversity(hyp):
    tokens = [0.0, 0.0]
    types = [defaultdict(int), defaultdict(int)]
    for line in hyp:
        for n in range(2):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / tokens[0]
    div2 = len(types[1].keys()) / tokens[1]
    return [div1, div2]


def calc_entropy(hyps, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for line in hyps:
        for n in range(4):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v / total * (np.log(v) - np.log(total))

    return etp_score


def load_txt(path):
    with open(path, encoding='UTF-8', errors='ignore') as f:
        data = [i.strip() for i in f.readlines() if len(i) > 0]
    return data


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Too few args for this script')

    # infer_path = sys.argv[1]
    infer_path = "../../result/440W_0.7t/LCCD_gpt_10_t0.7.txt"

    infer = load_txt(infer_path)
    # [[word,...,wordz],...,[word,...,word]]
    infer = [line.strip().split() for line in infer]

    random_distinct = [round(x, 4) for x in eval_distinct(infer)]
    print('random distinct', random_distinct)

    dist = calc_diversity(infer)
    print([round(x, 4) for x in dist])

    ent = calc_entropy(infer)
    print("ent", [round(x, 2) for x in ent])

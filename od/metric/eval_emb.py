#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
import jieba
import numpy as np
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
from od.utils.data_utils import *


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    # sim = 0.5 + 0.5 * cos
    return cos


def emb2dict(emb):
    emb_dict = {}
    for index, seq in enumerate(tqdm(emb[1:], mininterval=1)):
        assert seq[-1] == "\n"
        line = seq[:-1].split(" ")
        if len(line) != 201:
            import pdb
            pdb.set_trace()
        i = 1
        while True:
            try:
                float(line[i])
                break
            except:
                i += 1
        try:
            assert len(line[i:]) == 200
            k = "".join(line[:i])
            emb_dict[k] = [float(num) for num in line[i:]]
            if i > 1:
                print(k)
        except:
            import pdb
            pdb.set_trace()
    return emb_dict


def emb_lookup(doc, word_embedding):
    unk = set()
    doc_vecs = []
    for seq in doc:
        word_vecs = []
        for word in seq:
            if word == "，":
                word = ","
            word_vec = word_embedding.get(word, None)
            if word_vec is not None:
                word_vecs.append(word_vec)
            else:
                unk.add(word)
                # import pdb
                # pdb.set_trace()
                continue  # unk???
        doc_vecs.append(word_vecs)
    #print(unk)
    print(len(unk))
    return doc_vecs


def vector_extrema(x_words):
    '''
    对应公式部分：computing vector extrema by compapring maximun value of all word embeddings in same dimension.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :return: a 300 dimension list, vector extrema
    '''
    # print(np.array(x_words).shape)
    vec_extre = np.max(np.array(x_words), axis=0)
    return vec_extre


def eval_vector_extrema(infer, golden):
    cos = 0.0
    # hyps = []
    # refs = []
    for hyp, ref in tqdm(zip(infer, golden), mininterval=1):
        if len(hyp) > 0 and len(ref) > 0:
            vec_hyp = vector_extrema(hyp)
            vec_ref = vector_extrema(ref)
            cos += cos_sim(vec_hyp, vec_ref)
    #         hyps.append(vec_hyp.tolist())
    #         refs.append(vec_ref.tolist())
    # testa = cos_sim(hyps, refs)
    return cos / len(infer)


def eval_embedding_average(infer, golden):
    cos = 0.0
    # hyps = []
    # refs = []
    for hyp, ref in tqdm(zip(infer, golden), mininterval=1):
        if len(hyp) > 0 and len(ref) > 0:
            vec_hyp = embedding_avg(hyp)
            vec_ref = embedding_avg(ref)
            cos += cos_sim(vec_hyp, vec_ref)
    #         hyps.append(vec_hyp.tolist())
    #         refs.append(vec_ref.tolist())
    # testa = cos_sim(hyps, refs)
    return cos / len(infer)


def embedding_avg(x_words):
    '''
    上面的第一个公式：computing sentence embedding by computing average of all word embeddings of sentence.
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    sum_embed = np.sum(np.array(x_words), axis=0)
    avg_embed = sum_embed / np.linalg.norm(sum_embed)
    return avg_embed


def eval_greedy_match(infer, golden, infer_emb, golden_emb):
    score = 0.0
    for x, y, hyp, ref in tqdm(zip(infer, golden, infer_emb, golden_emb), mininterval=1):
        if len(hyp) > 0 and len(ref) > 0:
            sum_x = greedy(hyp, ref) / len(x)
            sum_y = greedy(ref, hyp) / len(y)
            score += (sum_x + sum_y) / 2
    return score / len(infer)


def greedy(x_words, y_words):
    '''
    上面提到的第一个公式
    :param x: a sentence, type is string.
    :param x_words: list[list1, list2,...,listn], listk(k=1...n) is word vector which from sentence x,
    :param y_words: list[list1, list2,..., listn], listk(k=1...n) is word vector which from sentence y,
    :return: a scalar, it's value is in [0, 1]
    '''
    cosine = []  # 存放一个句子的一个词与另一个句子的所有词的余弦相似度
    sum_x = 0.0  # 存放最后得到的结果
    for x_v in x_words:
        for y_v in y_words:
            cosine.append(cos_sim(x_v, y_v))
        if cosine:
            sum_x += max(cosine)
            cosine = []
    return sum_x


if __name__ == '__main__':
    # print(cosine_similarity([1, 1], [0, 0]))   # 0.0
    # print(cosine_similarity([1, 1], [-1, -1]))  # -1.0
    # print(cosine_similarity([1, 1], [2, 2]))  # 1.0

    path = 'E:/git/Tencent_AIlab_wordemb/test.txt'  # 这里改成你自己项目的路径
    word_embedding = load_txt(path)
    word_embedding = emb2dict(word_embedding)

    golden_path = "STC_test.txt"
    infer_path = "440W_0.7t/LCCD_gpt_20_t0.7.txt"
    infer = [list(jieba.cut(seq.replace(" ", ""))) for seq in load_txt(infer_path)[:1]]
    golden = [list(jieba.cut(seq.replace(" ", ""))) for seq in load_txt(golden_path)[:1]]
    infer_emb = emb_lookup(infer, word_embedding)
    golden_emb = emb_lookup(golden, word_embedding)

    sore_greedy = eval_greedy_match(infer, golden, infer_emb, golden_emb)
    score_avg = eval_embedding_average(infer_emb, golden_emb)
    score_ext = eval_vector_extrema(infer_emb, golden_emb)
    print(1)

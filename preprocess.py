import os
import gc
import random
from tqdm import tqdm
import collections

from od.utils.data_utils import *

random.seed(2019)


def de_generic(path, outpath):
    data = load_json(path)

    def ngrams(resp, n):
        ngram = []
        if len(resp) >= n:
            for i in range(len(resp) - n + 1):
                ngram.append(''.join(resp[i: i + n]))
        return ngram
    if os.path.exists("./temp/tri_grams.json"):
        generic = load_json("./temp/tri_grams.json")
    else:
        dataset = data["train"] + data["valid"]
        print("len raw: ", len(dataset))
        generic = collections.Counter()
        # assert isinstance(dataset[0][0], str)
        for dialog in dataset:
            for seq in tqdm(dialog, mininterval=1):
                seq = seq.replace(" ", "")
                tri_grams = ngrams(seq, 3)
                generic.update(list(set(tri_grams)))
        del dataset
        gc.collect()
        generic = sorted(generic.items(), key=lambda x: x[1], reverse=True)
        save_json(generic, "./temp/tri_grams.json")
    import pdb
    pdb.set_trace()
    screen = [(x, cnt) for x, cnt in generic if cnt > 1000]
    # print(screen)
    generic = set([x for x, cnt in screen])
    dirty_cnt = []
    dirty_gram = []
    new_data = {}
    for k, v in data:
        new_dataset = []
        for dialog in tqdm(v, mininterval=1):
            resp = dialog[-1].replace(" ", "")
            tri_grams = ngrams(resp, 3)
            flag = False
            cnt = collections.Counter(tri_grams)
            for word, num in cnt.items():
                if tri_grams.count(num)/len(tri_grams) > 0.9:
                    if word in generic:
                        dirty_cnt.append(resp)
                        flag = True
                        # break
                if num/len(tri_grams) > 0.9:
                    if word in generic:
                        dirty_gram.append(resp)
                        flag = True
                        break
            if flag:
                continue
            new_dataset.append(dialog)
        print("len new: ", len(new_dataset))
        new_data[k] = new_dataset
    save_json(dirty_cnt, "./temp/cnt.json")
    save_json(dirty_gram, "./temp/gram.json")
    # while len(new_data["test"]) < 10000:
    #     new_data["test"].append(new_data["train"].pop(-1))
    while len(new_data["valid"]) < 20000:
        new_data["valid"].append(new_data["train"].pop(-1))
    save_json(new_data, outpath)
    print("over")


def pro_CWB_json(indir, outdir, maxlen):
    single_data = load_json(indir + "single_v2.json")
    multi_data = load_json(indir + "multi_v2.json")
    new_single = []
    new_multi = []
    n_drop = 0
    max_len = 0
    print(len(single_data) + len(multi_data))
    for dialog in single_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in list(seq):
                char = token.strip()
                if char:
                    new_seq.append(char)
            assert len(new_seq) > 0
            new_dialog.append(new_seq)
        if sum([len(x) for x in new_dialog]) + 2 + len(new_dialog) > maxlen:
            n_drop += 1
            print(new_dialog)
            continue
        max_len = max(max_len, sum([len(x) for x in new_dialog]) + 2 + len(new_dialog))
        new_single.append(new_dialog)
    print(max_len, "single max len")
    for dialog in multi_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in list(seq):
                char = token.strip()
                if char:
                    new_seq.append(char)
            assert len(new_seq) > 0
            new_dialog.append(new_seq)
        new_multi.append(new_dialog)

    del single_data, multi_data
    gc.collect()
    print(len(new_multi) + len(new_single))

    max_len = 0
    split_muli = []
    for dialog in tqdm(new_multi):
        for i in range(2, len(dialog) + 1):
            new_dialog = dialog[:i]
            if sum([len(x) for x in new_dialog[-2:]]) + 2 + 2 > maxlen:
                n_drop += 1
                print(new_dialog[-2:])
                continue
            while sum([len(x) for x in new_dialog]) + 2 + len(new_dialog) > maxlen:
                new_dialog = new_dialog[1:]
            max_len = max(max_len, sum([len(x) for x in new_dialog]) + 2 + len(new_dialog))
            split_muli.append(new_dialog)
    print(max_len, "mul max len")
    print("drop", n_drop)
    new_multi = [[" ".join(seq) for seq in dialog] for dialog in split_muli]
    new_single = [[" ".join(seq) for seq in dialog] for dialog in new_single]
    random.shuffle(new_multi)
    random.shuffle(new_single)
    print(new_single[0])
    print(new_multi[0])
    print(len(new_single) + len(new_multi))

    valid = new_single[-10000:-3000] + new_multi[-20000:-7000]
    test = new_single[-3000:] + new_multi[-7000:]
    train = new_single[:-10000] + new_multi[:-20000]

    print(len(train))
    print(len(valid))
    print(len(test))
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    print(outdir + "data.json")
    save_json({"train": train, "valid": valid}, os.path.join(outdir, "CleanWB.json"))
    save_json({"test": test}, os.path.join(outdir, "CleanWB_test.json"))
    print("pro_CWB over")


def pro_CWB(indir, outdir, maxlen):
    single_data = load_json(indir + "single_v2.json")
    multi_data = load_json(indir + "multi_v2.json")
    new_single = []
    new_multi = []
    n_drop = 0
    max_len = 0
    print(len(single_data) + len(multi_data))
    for dialog in single_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in list(seq):
                char = token.strip()
                if char:
                    new_seq.append(char)
            assert len(new_seq) > 0
            new_dialog.append(new_seq)
        if sum([len(x) for x in new_dialog]) + 2 + len(new_dialog) > maxlen:
            n_drop += 1
            print(new_dialog)
            continue
        max_len = max(max_len, sum([len(x) for x in new_dialog]) + 2 + len(new_dialog))
        new_single.append(new_dialog)

    for dialog in multi_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in list(seq):
                char = token.strip()
                if char:
                    new_seq.append(char)
            assert len(new_seq) > 0
            new_dialog.append(new_seq)
        new_multi.append(new_dialog)

    del single_data, multi_data
    gc.collect()
    print(len(new_multi) + len(new_single))

    split_muli = []
    for dialog in tqdm(new_multi):
        for i in range(2, len(dialog) + 1):
            new_dialog = dialog[:i]
            if sum([len(x) for x in new_dialog[-2:]]) + 2 + 2 > maxlen:
                n_drop += 1
                print(new_dialog[-2:])
                continue
            while sum([len(x) for x in new_dialog]) + 2 + len(new_dialog) > maxlen:
                new_dialog = new_dialog[1:]
            max_len = max(max_len, sum([len(x) for x in new_dialog]) + 2 + len(new_dialog))
            split_muli.append(new_dialog)

    print("drop", n_drop)
    new_multi = split_muli
    random.shuffle(new_multi)
    random.shuffle(new_single)
    print(new_single[0])
    print(new_multi[0])
    print(len(new_single) + len(new_multi))

    valid = []
    test = []
    valid.extend(new_single[-10000:-3000])
    test.extend(new_single[-3000:])
    valid.extend(new_multi[-20000:-7000])
    test.extend(new_multi[-7000:])

    new_single = new_single[:-10000]
    new_multi = new_multi[:-20000]

    train = []
    for dialog in tqdm(new_multi + new_single):
        post = " [POST] ".join([" ".join(x) for x in dialog[:-1]])
        resp = dialog[-1]
        train.append(post + " [SEP] " + " ".join(resp))

    valid_txt = []
    for dialog in tqdm(valid):
        post = " [POST] ".join([" ".join(x) for x in dialog[:-1]])
        resp = dialog[-1]
        valid_txt.append(post + " [SEP] " + " ".join(resp))

    test_txt = []
    for dialog in tqdm(test):
        post = " [POST] ".join([" ".join(x) for x in dialog[:-1]])
        resp = dialog[-1]
        test_txt.append(post + " [SEP] " + " ".join(resp))

    print(len(train))
    print(len(valid))
    print(len(test))
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    print(outdir + "train.txt")
    save_txt("\n".join(train), outdir + "train.txt")
    save_txt("\n".join(valid_txt), outdir + "valid.txt")
    save_txt("\n".join(test_txt), outdir + "test.txt")
    print("pro_CWB over")


import unicodedata


def clean_data(indir, outdir):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

    def _is_whitespace(char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            if char != " ":
                # if is_number(char):
                #     import pdb
                #     pdb.set_trace()
                if char not in vocab:
                    if char not in safe_vocab:
                        dirty.add(char)
                        continue
            # if is_number(char):
            #     continue
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                dirty.add(char)
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    vocab = set(load_txt(indir + "bert_dirty/filter_vocab.txt"))
    safe_vocab = set(load_txt(indir + "bert_dirty/safe.txt"))
    single_data = load_json(indir + "single_final_v1.json")
    multi_data = load_json(indir + "multi_final_v1.json")
    print(safe_vocab)

    dirty = set()
    n_multi = 0
    n_single = 0
    new_multi = []
    white_multi = []
    white_single = []
    white_multi_dialog = []
    for dialog in tqdm(multi_data, mininterval=1):
        flag = False
        new_dialog = []
        for seq in dialog:
            one = _clean_text(seq).strip()
            if one:
                new_dialog.append(one)
            else:
                white_multi.append(seq)
                white_multi_dialog.append(dialog)
                n_multi += 1
                flag = True
                break
                # new_seq.append("哦")
                # import pdb
                # pdb.set_trace()
        if flag:
            continue
        new_multi.append(new_dialog)

    new_single = []
    for dialog in tqdm(single_data, mininterval=1):
        flag = False
        new_dialog = []
        for seq in dialog:
            one = _clean_text(seq).strip()
            if one:
                new_dialog.append(one)
            else:
                white_single.append(seq)
                n_single += 1
                flag = True
                break
        if flag:
            continue
        new_single.append(new_dialog)

    print(n_single, "multi:", n_multi)
    print(len(new_multi), "multi len")
    print(len(new_single), "single len")
    save_json(white_multi_dialog, outdir + "bert_dirty/multi_dirty_dialog.json")
    save_json(list(dirty), outdir + "bert_dirty/bert_dirty.json")
    save_json(white_single, outdir + "bert_dirty/single_white.json")
    save_json(white_multi, outdir + "bert_dirty/multi_white.json")
    save_json(new_multi, outdir + "multi_v2.json")
    save_json(new_single, outdir + "single_v2.json")


def main():
    # clean_data("/home/wangyida/211/v3/data/CleanWB/", "/home/wangyida/211/v3/data/CleanWB/")
    pro_CWB_json("/home/wangyida/data/CleanWB/", "./data/", 320)
    print("over")


if __name__ == '__main__':
    main()

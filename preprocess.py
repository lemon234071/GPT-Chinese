from od.utils.data_utils import *
import random
from tqdm import tqdm
import gc

random.seed(2019)


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
            for token in seq:
                if token.strip():
                    new_seq.append(token)
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
            for token in seq:
                if token.strip():
                    new_seq.append(token)
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
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char) or cp not in vocab:
                dirty.add(cp)
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    vocab = set(load_txt(indir+"filter_vocab.txt"))
    single_data = load_json(indir + "single_final_v1.json")
    multi_data = load_json(indir + "multi_final_v1.json")

    dirty = set()

    new_multi = []
    for dialog in tqdm(multi_data, mininterval=1):
        new_dialog = []
        for seq in dialog:
            new_seq = []
            one = _clean_text(seq).strip()
            if one:
                new_seq.append(one)
            else:
                import pdb
                pdb.set_trace()
            new_dialog.append(new_seq)
        new_multi.append(new_dialog)

    new_single = []
    for dialog in tqdm(single_data, mininterval=1):
        flag = False
        new_dialog = []
        for seq in dialog:
            new_seq = []
            one = _clean_text(seq).strip()
            if one:
                new_seq.append(one)
            else:
                flag = True
                break
            new_dialog.append(new_seq)
        if flag:
            continue
        new_single.append(new_dialog)

    save_json(new_multi, outdir + "multi_v2.json")
    save_json(new_single, outdir + "single_v2.json")
    save_json("\n".join(list(dirty)), outdir + "bert_dirty.txt")


def main():
    clean_data("/home/wangyida/211/v3/data/CleanWB/", "/home/wangyida/211/v3/data/CleanWB/")
    pro_CWB("/home/wangyida/211/v3/data/CleanWB/", "./data/", 320)
    print("over")


if __name__ == '__main__':
    main()

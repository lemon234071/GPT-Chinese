from od.utils.data_utils import *
from random import random

random.seed(2020)


def pro_CWB(indir, outdir):
    single_data = load_json(indir + "single_final_v1")
    multi_data = load_json(indir + "multi_final_v1")
    new_single = []
    new_multi = []
    for dialog in single_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in seq:
                if token:
                    new_seq.append(token)
            new_dialog.append(" ".join(new_seq))
        new_single.append(new_dialog)

    for dialog in multi_data:
        new_dialog = []
        for seq in dialog:
            new_seq = []
            for token in seq:
                if token:
                    new_seq.append(token)
            new_dialog.append(" ".join(new_seq))
        new_multi.append(new_dialog)

    del single_data, multi_data

    split_muli = []
    for dialog in new_multi:
        new_dialog = []
        for i in range(2, len(dialog)+1):
            new_dialog.append(dialog[:i])
        split_muli.append(new_dialog)

    new_multi = split_muli
    random.shuffle(new_multi)
    random.shuffle(new_single)
    print(len(new_single)+len(new_multi))

    valid = []
    test = []
    valid.append(new_single[-10000:-3000])
    test.append(new_single[-3000:])
    valid.append(new_multi[-20000:-7000])
    test.append(new_multi[-7000:])

    new_single = new_single[:-10000]
    new_multi = new_multi[:-20000]

    train = []
    for dialog in new_multi+new_single:
        post = " [POST] ".join(dialog[:-1])
        resp = dialog[-1]
        train.append(post + " [RESP] " + resp)

    print(len(train))
    print(len(valid))
    print(len(test))
    random.shuffule(train)
    random.shuffule(valid)
    random.shuffule(test)
    save_txt(train, outdir+"train.txt")
    save_txt(train, outdir + "valid.txt")
    save_txt(train, outdir + "test.txt")
    print("pro_CWB over")


def main():
    pro_CWB("/home/wangyida/211/v3/data/CleanWB/", "./data/")
    print("over")


if __name__ == '__main__':
    main()

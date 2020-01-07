from od.utils.data_utils import *
import random
from tqdm import tqdm

random.seed(2020)


def pro_CWB(indir, outdir):
    single_data = load_json(indir + "single_final_v1.json")
    multi_data = load_json(indir + "multi_final_v1.json")
    new_single = []
    new_multi = []
    print(len(single_data)+len(multi_data))
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
        new_multi.append(new_single)

    del single_data, multi_data
    print(len(new_multi)+len(multi_data))

    split_muli = []
    for dialog in new_multi:
        for i in range(2, len(dialog)+1):
            split_muli.append(dialog[:i])

    new_multi = split_muli
    random.shuffle(new_multi)
    random.shuffle(new_single)
    print(len(new_single)+len(new_multi))

    valid = []
    test = []
    valid.extend(new_single[-10000:-3000])
    test.extend(new_single[-3000:])
    valid.extend(new_multi[-20000:-7000])
    test.extend(new_multi[-7000:])

    new_single = new_single[:-10000]
    new_multi = new_multi[:-20000]

    train = []
    for dialog in tqdm(new_multi+new_single):
        post = " [POST] ".join(dialog[:-1])
        resp = dialog[-1]
        train.append(post + " [RESP] " + resp)

    valid_txt = []
    for dialog in tqdm(valid):
        post = " [POST] ".join(dialog[:-1])
        resp = dialog[-1]
        valid_txt.append(post + " [RESP] " + resp)

    test_txt = []
    for dialog in tqdm(test):
        post = " [POST] ".join(dialog[:-1])
        resp = dialog[-1]
        test_txt.append(post + " [RESP] " + resp)

    print(len(train))
    print(len(valid))
    print(len(test))
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    save_txt("\n".join(train), outdir+"train.txt")
    save_txt("\n".join(valid_txt), outdir + "valid.txt")
    save_txt("\n".join(test_txt), outdir + "test.txt")
    print("pro_CWB over")


def main():
    pro_CWB("/home/wangyida/211/v3/data/CleanWB/", "./data/")
    print("over")


if __name__ == '__main__':
    main()

import random
import json

from load_utils import *


def pre_single(path, outpath):
    data = load_json(path)
    random.shuffle(data)
    train = data[:10000]
    dev = data[10000:12000]
    test = data[12000:]

    CleanWB_pair_train_post = "\n".join([x[0] for x in train])
    CleanWB_pair_train_response = "\n".join([x[1] for x in train])

    CleanWB_pair_dev_post = "\n".join([x[0] for x in dev])
    CleanWB_pair_dev_response = "\n".join([x[1] for x in dev])

    CleanWB_pair_test_post = "\n".join([x[0] for x in test])
    CleanWB_pair_test_response = "\n".join([x[1] for x in test])

    save_txt(CleanWB_pair_train_post, outpath + "opensub_pair_train.post")
    save_txt(CleanWB_pair_train_response, outpath + "opensub_pair_train.response")

    save_txt(CleanWB_pair_dev_post, outpath + "opensub_pair_dev.post")
    save_txt(CleanWB_pair_dev_response, outpath + "opensub_pair_dev.response")

    save_txt(CleanWB_pair_test_post, outpath + "opensub_pair_test.post")
    save_txt(CleanWB_pair_test_response, outpath + "opensub_pair_test.response")


def test_Cgpt(path):
    def char_split(line):
        res = []
        seq = line.replace(" ", "")
        for word in seq:
            res.append(word)
        return " ".join(res)

    train_post = load_txt(path + "opensub_pair_train.post")
    train_post = list(map(lambda x: char_split(x), train_post))

    train_resp = load_txt(path + "opensub_pair_train.response")
    train_resp = list(map(lambda x: char_split(x), train_resp))

    valid_post = load_txt(path + "opensub_pair_dev.post")
    valid_post = list(map(lambda x: char_split(x), valid_post))

    valid_resp = load_txt(path + "opensub_pair_dev.response")
    valid_resp = list(map(lambda x: char_split(x), valid_resp))

    train = ["\t".join([x.strip(), y.strip()]) for x, y in zip(train_post, train_resp)]
    valid = ["\t".join([x.strip(), y.strip()]) for x, y in zip(valid_post, valid_resp)]
    save_txt("\n".join(train), "./train.txt")
    save_txt("\n".join(valid), "./valid.txt")


def main():
    # pre_single("./single_13000.json", "./")
    test_Cgpt("./")
    print("over")


if __name__ == '__main__':
    main()

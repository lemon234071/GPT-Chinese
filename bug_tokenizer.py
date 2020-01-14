from torch.utils.data import DataLoader
from pytorch_transformers import BertTokenizer
from argparse import ArgumentParser
from od.utils.data_utils import save_txt
from collections import defaultdict
from itertools import chain
import torch
# from torch.utils.data import Dataset
from od.inputters.dataset_base import Dataset

SPECIAL_TOKENS = ["<Lua heritage>", "<eos>", "madeupword0000", "madeupword0001"]
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]




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
                return tokenizer.tokenize(obj)
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        utterance = tokenize(utterance1)
        utterance["origin"] = utterance
        return utterance
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





class BugCollate(object):
    def __init__(self, dataset):
        self.padding = dataset.tokenizer.pad_token_id

    def __call__(self, batch):
        return batch


def bug_toke():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/toy_train.txt", help="Path of the dataset.")
    parser.add_argument("--model_checkpoint", type=str, default="./pretrain/Cgpt/",
                        help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=25, help="Number of previous exchanges to keep in history")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint,
                                              unk_token="<unk>", sep_token="</s>",
                                              pad_token="<pad>", cls_token="<Lua heritage>")
    train_dataset = WBDataset(args, tokenizer, data_path=args.train_path)

    train_loader = DataLoader(train_dataset,
                              collate_fn=BugCollate(train_dataset),
                              batch_size=4,
                              shuffle=False)

    test_tokenizer = []
    from tqdm import tqdm
    for batch in tqdm(train_loader, mininterval=1):
        for seq in batch:
            for k, v in seq.items():
                if k == "origin":
                    continue
                for sent in v:
                    one = "".join(sent)
                    try:
                        assert "[UNK]" not in one
                        assert "<unk>" not in one
                    except:
                        import pdb
                        pdb.set_trace()
                    test_tokenizer.append(one)
    save_txt("\n".join(test_tokenizer), "./bug_tokenizer.txt")
    print(1)


if __name__ == '__main__':
    bug_toke()

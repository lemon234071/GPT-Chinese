# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging
import random
from argparse import ArgumentParser
from pprint import pformat
from itertools import chain
import math

import torch
import torch.nn.functional as F

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, GPT2LMHeadModel
from od.WB_tokenization import WBTokenizer, VOCAB_FILE
from cotk.dataloader import GPTSingleTurnDialog


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]"]

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def sample_sequence(batch, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        inputs = [batch["input_gpt"], batch["label_gpt"]]
        input_ids, lm_labels = tuple(torch.LongTensor(x).to(args.device) for x in inputs)
        logits = model(input_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def get_cotk_data_loaders(args):
    data_class = GPTSingleTurnDialog.load_class(args.dataset)
    data = data_class(args.datapath,
                      bert_vocab_name=args.vocab_path,
                      min_vocab_times=args.min_vocab_times,
                      max_sent_length=args.max_sent_length)

    class cotk_loader(torch.utils.data.Dataset):
        def __init__(self, data, datakey, batch_size=1):
            self.data = data
            self.datakey = datakey
            self.shuffle = True if datakey == "train" else False
            self.batch_size = batch_size
            self.tokenizer = data.tokenizer

        def __len__(self):
            return math.ceil(self.data.data_size[self.datakey] / self.data.batch_size[self.datakey])

        def __getitem__(self, index):
            return self.data.get_batch(self.datakey, [index])

        def __iter__(self):
            for batch in self.data.get_batches(self.datakey, batch_size=self.batch_size, shuffle=self.shuffle):
                yield batch

    test_iter = cotk_loader(data, "test", 1)
    return test_iter


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GPTOpenSubtitles", help="Dataset.")
    parser.add_argument("--datapath", type=str, default="./data/",
                        help="Path of the dataset.")  # resources://OpenSubtitles
    parser.add_argument("--vocab_path", type=str, default="./pretrain/Cgpt/vocab.txt", help="Path of the vocab.")
    parser.add_argument("--min_vocab_times", type=int, default=0, help="")
    parser.add_argument("--max_sent_length", type=int, default=256, help="")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="")
    parser.add_argument("--valid_steps", type=int, default=125, help="")
    parser.add_argument("--out_path", type=str, default="", help="Path of response generated.")

    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    # tokenizer = WBTokenizer(os.path.join(args.model_checkpoint, VOCAB_FILE))
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    test_loader = get_cotk_data_loaders(args)
    tokenizer = test_loader.tokenizer

    out = []
    from tqdm import tqdm
    for batch in tqdm(test_loader, mininterval=2):
        with torch.no_grad():
            out_ids = sample_sequence(batch, tokenizer, model, args)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        out.append(out_text)

    with open(args.out_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()

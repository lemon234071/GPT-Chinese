from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import json
import sys
import os
import logging

import collections
from help.tokenization import WordpieceTokenizer, BasicTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILE = "vocab.txt"

VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt'
SPECIAL_TOKENS_NAME = 'special_tokens.txt'
BERT_VOCAB = './help/bert_vocab.txt'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class WBTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""
    def __init__(self, vocab_file, lowcase=False, special_tokens=None, split=False):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}' ".format(vocab_file))
        self.encoder = load_vocab(vocab_file)
        # self.decoder = {v: k for k, v in self.encoder.items()}
        self.decoder = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.encoder.items()])
        self.lowcase = lowcase
        self.split = split
        self.cache = {}
        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder, bert_vocab=load_vocab(BERT_VOCAB))

    def convert_to_dict(self, lprobs):
        vocab = dict()
        lprobs = lprobs.tolist()
        for ids, tok in self.decoder.items():
            vocab[tok] = lprobs[ids]
        return vocab

    def __len__(self):
        return len(self.encoder)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v:k for k, v in self.special_tokens.items()}

    def tokenize(self, text):
        """ Tokenize a string. """
        if self.split:
            return text.split()
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.encoder["[PAD]"]
        #return self.convert_tokens_to_ids("<pad>")

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.encoder["[SEP]"]

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.encoder["[UNK]"]

    def cls(self):
        return self.encoder["[CLS]"]

    def sep(self):
        return self.encoder["[SEP]"]

    def mask(self):
        return self.encoder["[MASK]"]

    def mask_lprobs(self, lprob, n=30000):
        lprob[:-n] = float("-inf")
        return lprob

    def mask_smp(self, lprob, n=30000):
        lprob[:,6:-n] = float("-inf")
        lprob[:,self.pad()] = float("-inf")
        lprob[:,self.cls()] = float("-inf")
        lprob[:,self.mask()] = float("-inf")
        lprob[:,self.unk()] = float("-inf")
        return lprob

    def add_symbol(self, word):
        """Adds a word to the dictionary"""
        if word not in self.encoder:
            idx = len(self.symbols)
            self.encoder[word] = idx
            self.symbols.append(word)
            return idx

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if self.lowcase:
                tokens = tokens.lower()
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, self.unk())
        for token in tokens:
            if self.lowcase:
                token = token.lower()
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, self.unk()))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
                # tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_text_to_ids(self, text):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = text.strip().split()
        return self.convert_tokens_to_ids(tokens)

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """Converts a sequence of ids in a string."""
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        out_string = ' '.join(tokens).strip()
        if clean_up_tokenization_spaces:
            # out_string = out_string.replace('<unk>', '')
            out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').replace(' ,', ','
                    ).replace(" ' ", "'")
        return out_string

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        # with open(vocab_file, 'w', encoding='utf-8') as f:
        #     f.write(json.dumps(self.encoder, ensure_ascii=False))
        with open(vocab_file, 'w', encoding='UTF-8') as f:
            f.write("\n".join(self.encoder))

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file, special_tokens_file




        


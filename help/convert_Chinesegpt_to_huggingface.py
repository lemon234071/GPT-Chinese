#!/usr/bin/env python
# coding:utf8

# Copyright (c) 2019, Tencent. All rights reserved
# Author: Tang Jing (jamesjtang@tencent.com)

# Convert AI Lab pretrained model to huggingface model structure.


import argparse
import collections
import json
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--Cgpt_model_path", default=None, type=str, required=True,
                    help="The Cgpt model path.")
parser.add_argument("--Cgpt_vocab_path", default=None, type=str, required=True,
                    help="The Cgpt vocab file path.")
parser.add_argument("--huggingface_dump_path", default=None, type=str, required=True,
                    help="The output path, it is a dir!")
args = parser.parse_args()


def gen_huggingface_gpt_config(converted_Cgpt_params):
    huggingface_gpt_config = dict()
    huggingface_gpt_config["afn"] = "gelu"
    huggingface_gpt_config["attn_pdrop"] = 0.1
    huggingface_gpt_config["embd_pdrop"] = 0.1
    huggingface_gpt_config["resid_pdrop"] = 0.1
    huggingface_gpt_config["initializer_range"] = 0.02
    huggingface_gpt_config["layer_norm_epsilon"] = 1e-05
    huggingface_gpt_config["n_ctx"] = 512
    huggingface_gpt_config["n_embd"] = 768
    huggingface_gpt_config["n_head"] = 12
    huggingface_gpt_config["n_layer"] = 12
    huggingface_gpt_config["n_positions"] = \
        converted_Cgpt_params["transformer.positions_embed.weight"].size(0)
    huggingface_gpt_config["n_special"] = 0
    huggingface_gpt_config["vocab_size"] = \
        converted_Cgpt_params["transformer.tokens_embed.weight"].size(0)
    # where is  the <unk>
    return huggingface_gpt_config


def gen_huggingface_gpt_model(Cgpt_params):
    huggingface_gpt_params = collections.OrderedDict()
    for k, v in Cgpt_params.items():
        if k == 'decoder.embeddings.weight':
            huggingface_gpt_params['transformer.tokens_embed.weight'] = v
        if k == 'decoder.pos_embeddings.weight':
            huggingface_gpt_params['transformer.positions_embed.weight'] = v
        if k == 'decoder.pre_softmax.weight':
            huggingface_gpt_params['lm_head.decoder.weight'] = v

        if k.endswith('.attn.qkv_proj.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.attn.c_attn.weight'
            huggingface_gpt_params[new_name] = v.t()
        if k.endswith('.attn.qkv_proj.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.attn.c_attn.bias'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.attn.out_proj.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.attn.c_proj.weight'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.attn.out_proj.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.attn.c_proj.bias'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.attn_norm.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.ln_1.weight'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.attn_norm.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.ln_1.bias'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.ff.layer_1.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.mlp.c_fc.weight'
            huggingface_gpt_params[new_name] = v.t()
        if k.endswith('.ff.layer_1.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.mlp.c_fc.bias'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.ff.layer_2.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.mlp.c_proj.weight'
            huggingface_gpt_params[new_name] = v.t()
        if k.endswith('.ff.layer_2.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.mlp.c_proj.bias'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.ff_norm.weight'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.ln_2.weight'
            huggingface_gpt_params[new_name] = v
        if k.endswith('.ff_norm.bias'):
            num_layer = k.split('.')[2]
            new_name = 'transformer.h.' + num_layer + '.ln_2.bias'
            huggingface_gpt_params[new_name] = v
    return huggingface_gpt_params


def gen_huggingface_vocab(Cgpt_vocab_file, vocab_size):
    vocab = ["<Lua heritage>", "<pad>", "</s>", "<unk>"]
    # vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    with open(Cgpt_vocab_file, "r", encoding="UTF-8") as f:
        for line in f:
            data = line.strip().split()
            if len(vocab) < vocab_size:
                vocab.append(data[0])
            else:
                break
    return vocab


def convert():
    Cgpt_model = torch.load(args.Cgpt_model_path, map_location="cpu")
    huggingface_model = gen_huggingface_gpt_model(Cgpt_model)
    huggingface_config = gen_huggingface_gpt_config(huggingface_model)
    huggingface_vocab = gen_huggingface_vocab(args.Cgpt_vocab_path,
                                              huggingface_config["vocab_size"])
    if not os.path.exists(args.huggingface_dump_path):
        os.mkdir(args.huggingface_dump_path)
    shape_dict = {k: v.shape for k, v in huggingface_model.items()}
    with open('./shape_c.json', 'w', encoding='UTF-8') as f:
        json.dump(shape_dict, f, ensure_ascii=False, indent=2)
    torch.save(huggingface_model,
               os.path.join(args.huggingface_dump_path, "pytorch_model.bin"))
    with open(os.path.join(args.huggingface_dump_path, "config.json"), "w") as f:
        json.dump(huggingface_config, f, indent=2)
    with open(os.path.join(args.huggingface_dump_path, "vocab.txt"), "w", encoding="UTF-8") as f:
        for token in huggingface_vocab:
            f.writelines(token + "\n")


if __name__ == "__main__":
    convert()

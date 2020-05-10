# GPT-Chinese  
* This project provides a large-scale Cleaned Chinese conversation dataset and generative pre-training models trained 
on this dataset, and more details refer to [A Large-scale Cleaned Chinese Dataset for Open-domain Conversation Generation](). 
The code is adapted from [TransferTransfo](https://github.com/huggingface/transfer-learning-conv-ai) using 
[Transformers](https://github.com/huggingface/transformers) of HuggingFace, 
which can be used for pre-training and fine-tuning.
* 本项目提供了一个大规模的经过系统清洗的中文对话数据集，并提供在此数据集上的对话预训练模型，更多信息可参考论文 
[A Large-scale Cleaned Chinese Dataset for Open-domain Conversation Generation]()。
本项目代码使改自[TransferTransfo](https://github.com/huggingface/transfer-learning-conv-ai)，使用HuggingFace Pytorch版的
[Transformers](https://github.com/huggingface/transformers), 可用于预训练与微调。

## Contents
* [Dataset](##Dataset)
* [Pre-training](##Pre-training)
* [Generation](##Generation)

## Dataset
In this project, we present a Large-scale Cleaned Chinese Conversation corpus (LCCC) consists of 
[LCCC-base](https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip) and 
[LCCC-large](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCD.zip). 
The LCCC-base is cleaner but smaller than LCCC-large. 
The quality of our dataset is ensured by a rigorous data cleaning pipeline, which is built based on a set of rules and 
learned filter trained on manually an-notated dialogue pairs. 
The noises we considered include swearing words, sensitive words, special symbols, emoji, grammatical errors, and context-free conversations.
The statistic is described below, and the vocabulary words is counted based on [Jieba](https://github.com/fxsjy/jieba) segmentation.

| [LCCC-base](https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip) | Single-turn  | Multi-turn  |
| :----------- | :----------: | :---------: |
| Coversations  | 3,354,382    | 3,466,607   | 
| Utterances    | 6,708,554    | 13,365,268  |
| Characters    | 68,559,727   | 163,690,614 | 
| Vocabulary    | 372,063      | 666,931     |
| Avg. words    |    6.79      | 8.32        |
| Avg. turns    |    2         | 3.86        |

| [LCCC-large](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCD.zip) | Single-turn  | Multi-turn  |
| :----------- | :----------: | :---------: |
| Coversations  | 7,273,804    | 4,733,955   | 
| Utterances    | 14,547,608   | 18,341,167  |
| Characters    | 162,301,556  | 217,776,649 | 
| Vocabulary    | 662,514      | 690,027     |
| Avg. words    |    7.45      | 8.14        |
| Avg. turns    |    2         | 3.87        |


The LCCC-base is constructed on Weibo Corpus, which we crawled from [Weibo](www.weibo.com). 
The LCCC-large is a combination of several conversation datasets:

| Dataset                              | Sessions  | Sample                               | 
| :---------------------------------- | :-------: | :---------------------------------- | 
| Weibo Courpus                      | 79M | Q:火锅我在重庆成都吃了七八顿火锅 A: 哈哈哈哈！那我的嘴巴 可能要烂掉！ |
| [Chinese Chatterbot Corpus](https://github.com/gunthercox/chatterbot-corpus) | 560 | Q:火锅我在重庆成都吃了七八顿火锅 A: 哈哈哈哈！那我的嘴巴 可能要烂掉！ |
| [PTT Gossiping Corpus](https://github.com/zake7749/Gossiping-Chinese-Corpus) | 40W | Q:为什么乡民总是欺负国高中生呢QQ A:如果以为选好科系就会变成比尔盖兹那不如退学吧 | 
| [Subtitle Corpus](https://github.com/skdjfla/dgklostconv) | 274W | Q:京戏里头的人都是不自由的 A:他们让人拿笼子给套起来了了 |
| [Xiaohuangji Corpus](https://github.com/skdjfla/dgklostconv) | 45W | Q:前排，鲁迷们都起床了吧 A:标题说助攻，但是看了那球，真是活生生的讽刺了 | 
| [Tieba Corpus](https://github.com/codemayq/chinesechatbotcorpus) | 232W | Q:前排，鲁迷们都起床了吧 A:标题说助攻，但是看了那球，真是活生生的讽刺了 |
| [Qingyun Corpus](https://github.com/codemayq/chinesechatbotcorpus) | 10W | Q:看来你很爱钱 A:噢是吗？那么你也差不多了 |
| [Douban Conversa-tion Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) | 400W | Q: 看 原版 英文 电影 学 纯正 英语 A: 大 爱 老友 记 反复 看 了 好多 次 了 Q: 一样 光盘 都 快 被 我 看 花 了 A: 那 你 现在 的 英语 应该 不错 了 |
| [E-commerical Conversation Corpus](https://github.com/cooelf/DeepUtteranceAggregation) | 50W | Q: 这个 会 不会 聚 划算 A: 暂时 没有 哦 Q: 后期 会 不会 有 A: 不 一定 哦 亲 多多 关注 我们 哦 |
| [Chinese chat corpus](https://github.com/yangjianxin1/GPT2-chitchat) | 50W | Q: 我今天腿都废了，你们过节，我搬砖 A: 辛苦啊，圣诞节还去赚大钱了加油 Q: 毕竟是没男朋友的人，什么节都是一样的 |

## Pre-trianing
### Installation  
Install from the sources:  

    git clone https://github.com/lemon234071/GPT-Chinese.git
    cd GPT-Chinese
    pip install -r requirements.txt 
    
### Models  
We present Chinese dialog generative pre-training models which are first pre-trained on the Chinese novel dataset 
and then post-trained on our LCCC dataset. The architecture is adapted from 
[TransferTransfo](https://arxiv.org/abs/1901.08149) that we removed the classification module.

| Models        | Parameter Size | Train Dataset            | Description                                       | 
|---------------| ------ |--------------------------|-------------------------------------------------- |
| [GPT<sub>Novel</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/Novel_GPT.zip) | 104M | Chinese Novel            | Trained on Chinese Novel dataset (1.3B words)     |
| [GPT<sub>LCCC-base</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB_GPT.zip)  | 104M | [LCCC-base](##datasets)  | Post-trained on LCCC-base dataset from GPT_novel  | 
| [GPT2<sub>LCCC-base</sub>]() | 104M | [LCCC-base](##datasets)  | Post-trained on LCCC-base dataset from GPT_novel  | 
| [GPT<sub>LCCC-large</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCD_GPT.zip) | 104M | [LCCC-large](##datasets) | Post-trained on LCCC-large dataset from GPT_novel | 


### Quickstart
Step 1: Prepare the data (train data or fine-tune data, E.g., [STC dataset](https://arxiv.org/abs/1503.02364)) 
    
Step 2: Train the model

    python train.py # Single GPU training
    python -m torch.distributed.launch --nproc_per_node=8 train.py  # Training on 8 GPUs

Step 3: Generating

    python infer.py --model_checkpoint ./models/  # Infer on a corpus
    python interact.py --model_checkpoint ./models/  # Interact on the terminal

Training Arguments

| Arguments  | Type     | Default value  | Description | 
| :---- | :---------- | :----- | :------- | 
| model_checkpoint | str | "" | Path or URL of model files (Dirpath of pre-training model or config/vocab files) |
| pretrained  | bool | False | If False train from scratch (config/vocab files) | 
| data_path | str | "" | Path of the dataset | 
| dataset_cache | str | default="dataset_cache" | Path or url of the dataset cache | 
| train_path | str | "" | Path of the train dataset for distributed dataset | 
| valid_path | str | "" | Path of the valid dataset for distributed dataset | 
| log_file | str | "" | Output logs to a file under this path | 
| num_workers | int | 1 | Number of subprocesses for data loading | 
| n_epochs | int | 70 | Number of training epochs | 
| train_batch_size | int | 8 | Batch size for training | 
| valid_batch_size | int | 8 | Batch size for validation |
| max_history | int | 15 | Number of previous exchanges to keep in history |  
| scheduler | str | "noam" | Method of optimizer |
| n_emd | int | 768 | Number of n_emd in config file (for noam) |
| eval_before_start | bool | False | If true start with a first evaluation before training |    
| warmup_steps | int | 5000 | Warm up steps |   
| valid_steps | int | 0 | Perfom validation every X steps, if is not 0" |   
| gradient_accumulation_steps | int | 64 | Accumulate gradients on several steps |   
| max_norm | float | 1.0 | Clipping gradient norm |   
| device | str | "cuda" if torch.cuda.is_available() else "cpu" | Device (cuda or cpu) |   
| fp16 | str | "" | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation) |   
| local_rank | int | -1 | Local rank for distributed training (-1: not distributed) |   

Input data representation

We concatenate all history utterances into one sequence as a long text described in [Transfertransfo](https://arxiv.org/abs/1901.08149). 
As shown below, the input of the model is the sum of word embedding, speaker embedding, and position embedding.

![Input representation](Figures/inputs.png) 

## Evaluation (Fine-tuned on [STC dataset](https://arxiv.org/abs/1503.02364))  
All samples evaluated are generated by [Nucleus Sampling](https://arxiv.org/abs/1904.09751) with threshold 0.9 and temperature 0.7.  
Automatic evaluation  

| Models  | Model Size | PPL  | BLEU-2 | BLEU-4 | Dist-1 | Dist-2 | Greedy Matching | Embedding Average |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Attn-Seq2seq | 73M | 34.20 | 39.3 | 9.0 | 8.5 | 11.91 | 65.84 | 83.38 |
| Transformer | 113M | 22.10 | 67.2 | 31.4 | 8.8 | 13.97 | 66.06 | 83.55 |
| GPT2-chitchat | 88M | - | 22.8 | 5.4 | 10.3 | 16.25 | 61.54 | 78.94 |
| GPT<sub>Novel</sub> | 104M | 21.27 | 59.6 | 27.1 | 8.0 | 11.72 | 66.12 | 83.34 |
| GPT<sub>LCCC-base</sub> | 104M | 18.38 | 64.8 | 30.8 | 8.3 | 12.68 | 66.21 | 83.54 |
| GPT<sub>LCCC-base</sub> | 104M | 22.76 | 56.9 | 25.0 | 7.7 | 10.87 | 66.24 | 83.46 |
| GPT<sub>LCCC-large</sub> | 104M | 18.23 | 66.3 | 32.0 | 8.3 | 12.71 | 66.25 | 83.63 |

Human evaluation
We also employed 3 workers to evaluate 200 samples of each model about fluency, relevance, and diversity based on 
a 2/1/0 score schema, more details refer to our [paper]().

| Models  | +2      | +1      | +0      | Kappa   |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| Attn-Seq2Seq | 27.1% | 21.4% | 51.4% | 0.4544 |
| Transformer | 42.4% | 23.6% | 34.0% | 0.4429 |
| GPT2-chitchat | 24.3% | 19,1% | 56.6% | 0.3941 |
| GPT<sub>LCCC-base</sub> | 46.7% | 21.8% | 31.5% | 0.3954 |
| GPT<sub>LCCC-large</sub> | 48.6% | 24.5% | 27.0% | 0.4203 |

Toxic generation statistic 

We randomly select 1000 samples of generation and make a statistic of toxic generation.

| Samples | Repetition | Swearing | Others | 
| :-----: | :----------: | :-----: | :-------: | 
| 1000 | 23 | 9 | 2 |

##Case study
Generation of GPT<sub>LCCC-large</sub>

Single-turn generation examples form STC dataset (fine-tuned on STC):

| Post | Response |
| :----- | :---------- |
| 这 居 然 是 北 京 不 是 哈 尔 滨 。 | 哈 尔 滨 的 天 气 好 像 比 北 京 好 点 ， 我 这 里 就 是 小 雨 ， 白 天 都 是 大 风 |
| 运 营 商 校 园 广 告 大 P K ， 太 欢 乐 了 ！ 哈 哈 哈 。 | 我 以 为 是 马 云 的 广 告 。 |
| D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？ | 最 后 一 件 太 美 了 ， 美 的 不 像 话 ！ |

Multi-turn generation examples from an interactive environment:

| Roel | Response |
| :----- | :---------- |
| User | 别爱我没结果 |
| Bot | 没 结 果 的 爱 是 多 么 的 痛 。 |
| User | 但是爱过就可以了 |
| Bot | 有 时 候 ，爱 情 就 是 一 场 游 戏 。 |
| User | 输的总是我 |
| Bot | 我 爱 你 ， 就 像 我 爱 你 一 样 。 |
| User | 你可拉到吧 |
| Bot | 你 可 拉 到 吧 。 |
| User | 别学我说话 |
| Bot | 我 说 话 就 是 你 不 行 。 |
| User | 逼我动手是吧 |
| Bot | 不 ， 是 逼 你 动 口 是 吧 |

Multi-turn self-playing dialogue with user prompt:

Commonsense question answering examples:

## Authors
Yida Wang, Pei Ke, Yinhe Zheng, Kaili Huang, Minlie Huang, Xiaoyan Zhu

## Acknowledgements

## Disclaimer
The LCCC dataset and pre-trained models aim to facilitate research for conversation generation. 
The LCCC dataset provided in this repository is crawled from Weibo. This project has carried out a rigorous cleaning 
on it, but it is not guaranteed to completely clean out the inappropriate content, which does not represent the 
author's opinion.
This repository contains only part of the modeling machinery needed actually to produce a model weight file in a running dialog.
The decoding script provided in this project is only for the researcher to test the generation effect of the pre-trained model.
We are not responsible for any generation from the 3rd party utilization of the pre-trained system.

## Citation
If you use the dataset or models in your research, you can cite our [arxiv paper]():

    Coming soon
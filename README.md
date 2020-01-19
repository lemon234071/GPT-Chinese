# GPT-Chinese  
* 
* 本项目提供一个在1.3G小说数据上训练的中文GPT模型以及代码，并提供 [Large-scale Cleaned Chinese Dataset for Open-domain Conversation]()
Generation中的大规模对话预料GPT、GPT2模型。本项目代码使改自[TransferTransfo]()，使用HuggingFace Pytorch版的[Transformers](), 可用于预训练与微调。
## Installation  
Install from the sources:  

    git clone https://github.com/lemon234071/Chinese-GPT.git
    cd Chinese-GPT
    pip install -r requirements.txt 
    
##Pretrained model  
Model details  

| Models      | Layers  | Description | 
| :---------: | :-----: | :-------: | 
| [GPT-C]()   |   12    |    Trained on 1.3B novel data from scratch    |
| [GPT-CD]()  |   12    |    Fine tuned on 6.8M CleanWB data from C-GPT     | 
| [GPT2-CD]() |   12    |    Fine tuned on 6.8M CleanWB data from C-GPT    | 
| [GPT2-CD]() |   24    |    Trained on the dataset below from scratch     | 

The largest dialog dataset is a combination of several dialog datasets:

| Dataset     | Source  | Scale  | Description | 
| :---------: | :-----: | :-----: | :-------: | 
| [CleanWB]() |    -    |  6.8M    |        |
| [C-DGPT]()  |    -    |      |         | 
| [C-DGPT2]() |    -    |      |        | 
| [C-DGPT2]() |    -    |      |         |


##Fine-tune quickstart
Step 1: Prepare the fine-tuned data

    
Step 2: Train the model

    python train.py # Single GPU training
    python -m torch.distributed.launch --nproc_per_node=8 train.py  # Training on 8 GPUs

Step 3: interaction

    python interact.py --model_checkpoint ./runs/

Training Arguments

| Arguments  | Type     | Default value  | Description | 
| :-----: | :----------: | :-----: | :-------: | 
| C-GPT   | from scratch |   12    |    1.3B novel data     |
| C-DGPT  | from C-GPT   |   12    |    6.8M CleanWB data     | 
| C-DGPT2 | from C-GPT   |   12    |    6.8M CleanWB data     | 
| C-DGPT2 | from scratch |   24    |    -     | 
Interact Arguments

| Arguments  | Type     | Default value  | Description | 
| :-----: | :----------: | :-----: | :-------: | 
| C-GPT   | from scratch |   12    |    1.3B novel data     |
| C-DGPT  | from C-GPT   |   12    |    6.8M CleanWB data     | 
| C-DGPT2 | from C-GPT   |   12    |    6.8M CleanWB data     | 
| C-DGPT2 | from scratch |   24    |    -     | 

##Evaluation  
Zero-shot  

| Models  |   PPL   |    BLEU   | 
| :-----: | :-----: | :-------: | 
| C-GPT   |   12    |    no     |
| C-DGPT  |   12    |    no     | 
| C-DGPT2 |   12    |    no     | 
| C-DGPT2 |   24    |    no     | 

Fine-tuned on   

| Models  |   PPL   |    BLEU   | 
| :-----: | :-----: | :-------: | 
| C-GPT   |   12    |    no     |
| C-DGPT  |   12    |    no     | 
| C-DGPT2 |   12    |    no     | 
| C-DGPT2 |   24    |    no     | 

##Case study

##Author
Yida

##Acknowledgements
##Disclaimer
##Citation
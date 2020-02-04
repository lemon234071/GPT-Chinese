# CUDA_VISIBLE_DEVICES=3,4,5,7 OMP_NUM_THREADS=32 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --pretrained --eval_before_start --model_checkpoint ./pretrain/Cgpt/ --data_path ./data/CleanWB.json --n_epochs 70 --valid_steps 40000 --train_batch_size 8 --valid_batch_size 8 --lr 0.3521 --warmup_steps 41326
CUDA_VISIBLE_DEVICES=3,4,5,7 OMP_NUM_THREADS=32 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --gpt2 --pretrained --eval_before_start --model_checkpoint ./pretrain/Cgpt/ --data_path ./data/CleanWB.json --n_epochs 70 --valid_steps 40000 --train_batch_size 8 --valid_batch_size 8 --lr 1.11336 --warmup_steps 413190 --num_workers 32

""" Implementation of all available options """
import torch


def train_opts(parser):
    parser.add_argument("--dataset", type=str, default="GPTOpenSubtitles", help="Dataset.")
    parser.add_argument("--datapath", type=str, default="./data/", help="Path of the dataset.")# resources://OpenSubtitles
    parser.add_argument("--vocab_path", type=str, default="./pretrain/Cgpt/vocab.txt", help="Path of the vocab.")
    parser.add_argument("--min_vocab_times", type=int, default=0, help="")
    parser.add_argument("--max_sent_length", type=int, default=512, help="")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="")
    parser.add_argument("--valid_steps", type=int, default=125, help="")

    parser.add_argument("--train_path", type=str, default="./data/train.txt", help="Path of the dataset.")
    parser.add_argument("--valid_path", type=str, default="./data/valid.txt", help="Path of the dataset.")
    parser.add_argument("--model_checkpoint", type=str, default="./pretrain/Cgpt/",
                        help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=25, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--num_workers", type=int, default=8, help="How many subprocesses to use for data loading")
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrian model')
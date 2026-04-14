# Импорт библиотек
import os, torch, sys
import argparse
from transformers import (
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ModelTLite, ModelQwen
from promts import format_chat_template 
from util import print_options, seed_everything, start_tensorboard
from metrics import make_metric_fn, encode_sent
from trainer_callback import ChatGenerationCallback
from train_rl import train_grpo
from train_sft import train_sft


def main(args):
    seed_everything()

    if args.model == 't-lite':
        model = ModelTLite(quantization=True, path=args.from_pretrain, 
                           markup=args.markup, train_mode=args.train_mode)
    elif args.model == 'qwen':
        model = ModelQwen(quantization=True, path=args.from_pretrain, 
                          markup=args.markup, train_mode=args.train_mode)
    model.model.train()

    # LoRA config / адаптер 
    if args.from_pretrain == '':
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
    else:
        peft_config = None

    eval_data = pd.read_csv(args.test_dataset, index_col='Unnamed: 0')
    if os.path.isdir(args.train_dataset):
        all_dfs = []
        # Рекурсивно обходим все файлы в папке
        for root, _, files in os.walk(args.train_dataset):
            for file in files:
                file_path = Path(root) / file
                all_dfs.append(pd.read_csv(file_path, index_col='Unnamed: 0'))
        train_data = pd.concat(all_dfs)
    else:
        train_data = pd.read_csv(args.train_dataset)
        if 'Unnamed: 0' in train_data.columns:
            train_data = train_data.set_index('Unnamed: 0')
    dataset = {
        'train': train_data,
        'test': eval_data,
    }

    markup = args.markup if args.train_mode == 'sft' else None

    format_chat_template_ = lambda row: format_chat_template(row, model.tokenizer, args.pretrain, markup=markup)
    dataset['train'] = dataset['train'].apply(
        format_chat_template_, axis=1
    )
    dataset['test'] = dataset['test'].apply(
        format_chat_template_, axis=1
    )
    if args.train_mode == 'sft':
        dataset = {
            'train': Dataset.from_pandas(dataset['train'][['text']]),
            'test': Dataset.from_pandas(dataset['test'][['text']]),
        }
    elif args.train_mode == 'grpo':
        dataset['train']['input_emb'] = encode_sent(dataset['train']['input'].tolist()).tolist()
        dataset['test']['input_emb'] = encode_sent(dataset['test']['input'].tolist()).tolist()
        dataset = {
            'train': Dataset.from_pandas(dataset['train'].rename(columns={'text': 'prompt'})
                                    .reset_index(drop=True)[['prompt', 'input_emb', 'rhyme_scheme', 'meter']]),
            'test': Dataset.from_pandas(dataset['test'].rename(columns={'text': 'prompt'})
                                    .reset_index(drop=True)[['prompt', 'input_emb', 'rhyme_scheme', 'meter']]),
        }

    if args.train_mode == "sft":
        trainer = train_sft(
            model.model, 
            model.tokenizer, dataset, 
            peft_config, 
            eval_data[~eval_data['meter'].isin(['dolnik2', 'dolnik3'])].iloc[:20], 
            args
        )
    elif args.train_mode == "grpo":
        trainer = train_grpo(
            model.model, 
            model.tokenizer, dataset, 
            peft_config, 
            args
        )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--name_run', type=str, default='', help='Name of the run (for logging purposes). If empty uses args.model instead.')
    parser.add_argument('--train_dataset', type=str, default='dataset/trainset.csv', help='Path to training dataset')
    parser.add_argument('--test_dataset', type=str, default='dataset/testset.csv', help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to existing model checkpoint to resume training')
    parser.add_argument('--model', type=str, default='t-lite', choices=['t-lite', 'qwen'], help='Model type: "t-lite" or "qwen"')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_steps', type=int, default=2000, help='Save model every N // args.batch_size steps')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warm-up steps')
    parser.add_argument('--log_steps', type=int, default=100, help='Log training metrics every N // args.batch_size steps')
    parser.add_argument('--pretrain', action='store_true', help='If set, enables poetry-only pretraining mode')
    parser.add_argument('--from_pretrain', type=str, default='', help='Path to pretrained model checkpoint')
    parser.add_argument('--markup', type=str, default='stanzas', choices=['rhyme_markup', 'stress_markup', 'stanzas', 'rhyme_stress_markup'], help='The used markup')

    parser.add_argument('--train_mode', type=str, default='sft', choices=['sft', 'grpo'], help='The used training mode')

    parser.add_argument('--rhyme_coef', type=float, default=0.3, help='Rhyme score coefficient in rl metric')
    parser.add_argument('--meter_coef', type=float, default=0.2, help='Meter score coefficient in rl metric')
    parser.add_argument('--len_coef', type=float, default=0.1, help='Len score coefficient in rl metric')
    parser.add_argument('--sem_coef', type=float, default=0.4, help='Semantic score coefficient in rl metric')
    parser.add_argument('--num_generations', type=int, default=4, help='Number of generations in GRPO')

    args, unknown1 = parser.parse_known_args()

    unknown_args = set(unknown1)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)
    print_options(args, parser)
    main(args)
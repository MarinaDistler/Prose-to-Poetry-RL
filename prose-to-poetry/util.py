import torch
import random
import numpy as np
import typing as tp
import os
import shutil
import re
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import ast
from torch.amp import autocast

from promts import get_train_prompt, system_instruction

def clean_responses(responses):
    for i in range(len(responses)):
        responses[i] = re.sub(r"<rhyme[AB]>.*?</rhyme[AB]>", "", responses[i])
        responses[i] = re.sub(r'<(?:[sS]\d+|count\d+)>', '', responses[i])
    return responses

def filter_lines(lines):
    return [line.strip() for line in lines if any(ch.isalpha() for ch in line)]

def text_to_lines(text):
    return [line.strip() for line in text.split('\n') if any(ch.isalpha() for ch in line)]

def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        if parser is not None:
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------\n'
    print(message)


def seed_everything(seed: int = 1729) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def start_tensorboard(name, project, config=None, log_dir="runs"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{project}/{name}_{timestamp}"
    
    log_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    if config is not None:
        for key, value in config.items():
            writer.add_text(f"config/{key}", str(value))
    
    return writer, log_dir





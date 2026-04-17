import torch
import random
import numpy as np
import typing as tp
import os
import sys
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
    lines = [line.strip() for line in lines if any(ch.isalpha() for ch in line)]
    if len(lines) > 0 and lines[0] == 'assistant':
        lines = lines[1:]
    return lines

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



class Tee:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, "a")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()





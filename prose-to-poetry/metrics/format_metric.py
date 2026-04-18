import numpy as np
import re

from util import filter_lines


def non_russian_penalty(text):
    # всё, что НЕ разрешено
    forbidden = re.findall(r"[^а-яёА-ЯЁ\s.,!?—:;()\-\«\»\"]", text)
    
    if len(text) == 0:
        return 1.0  # максимальный штраф
    
    ratio = len(forbidden) / len(text)
    return ratio  # от 0 до 1

def format_score(lines, filtered_lines):
    # число строк (идеал = 4)
    line_score = np.exp(-abs(len(filtered_lines) - 4))   # плавно: 1 → 0.37 → 0.14
    
    # пустые строки
    empty = len(lines) - len(filtered_lines)
    if len(lines) > 0 and lines[0] == 'assistant':
        empty -= 1
    empty_score = np.exp(-empty)

    penalty = non_russian_penalty(''.join(filtered_lines))
    lang_score = np.exp(-5 * penalty)
    
    return lang_score * (0.6 * line_score + 0.4 * empty_score)

def make_format_reward(coef):
    def format_reward(completions, **kwargs):
        rewards = []
        
        for text in completions:
            lines = text.split('\n')
            f_lines = filter_lines(lines)
            
            score = format_score(lines, f_lines)
            rewards.append(coef * score)
        
        return rewards
    
    return format_reward
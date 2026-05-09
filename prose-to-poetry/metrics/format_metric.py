import numpy as np
import re
from pymorphy2 import MorphAnalyzer

from util import filter_lines

morph = MorphAnalyzer()

def unknown_word_ratio(text):
    words = re.findall(r"[а-яёА-ЯЁ]+", text)

    if not words:
        return 0.0

    unknown = 0

    for w in words:
        parses = morph.parse(w)
        best = parses[0]

        if 'UNKN' in best.tag or best.score < 0.2:
            unknown += 1

    return unknown / len(words)

def length_reward_smooth(len_in, len_out, min_ratio=0.9, max_ratio=1.5, sigma_l=0.1, sigma_r=0.3):
    if len_out == 0: return 0
    
    ratio = len_in / len_out
    
    # Если внутри границ - идеальная награда 1.0
    if min_ratio <= ratio <= max_ratio:
        return 1.0
    
    if ratio < min_ratio:
        # Если вышли за границы - считаем расстояние до ближайшего края
        dist = abs(ratio - min_ratio)
        
        # Экспоненциальный штраф: чем дальше, тем ближе к 0
        # sigma управляет "крутизной" падения. 0.2 - довольно плавно.
        return np.exp(-(dist**2) / (2 * sigma_l**2))
    else:
        # Если вышли за границы - считаем расстояние до ближайшего края
        dist = abs(ratio - max_ratio)
        
        # Экспоненциальный штраф: чем дальше, тем ближе к 0
        # sigma управляет "крутизной" падения. 0.2 - довольно плавно.
        return np.exp(-(dist**2) / (2 * sigma_r**2))

def non_russian_penalty(text):
    # всё, что НЕ разрешено
    forbidden = re.findall(r"[^а-яёА-ЯЁ\s.,!?—:;()\-\«\»\"]", text)
    
    if len(text) == 0:
        return 1.0  # максимальный штраф
    
    ratio = len(forbidden) / len(text)
    return ratio  # от 0 до 1

def intra_repetition_reward(lines):
    if len(lines) <= 1:
        return 1.0
    
    num_total = len(lines)
    num_unique = len(set(lines))
    
    # Награда = доля уникальных строк (от 0 до 1)
    # Если все 4 разные -> 1.0. Если 2 одинаковые -> 0.75. Если все 4 одинаковые -> 0.25.
    return num_unique / num_total

def format_score(text, lines, filtered_lines, input_len, use_unknown_ratio=True):
    # число строк (идеал = 4)
    line_score = np.exp(-abs(len(filtered_lines) - 4))   # плавно: 1 → 0.37 → 0.14
    
    # пустые строки
    empty = len(lines) - len(filtered_lines)
    if len(lines) > 0 and lines[0] == 'assistant':
        empty -= 1
    if len(lines) > 0 and lines[-1] == '':
        empty -= 1
    empty_score = np.exp(-empty / 2)

    penalty = non_russian_penalty(text)
    lang_score = np.exp(-5 * penalty)

    if use_unknown_ratio:
        grammar_score = np.exp(-5 * unknown_word_ratio(text))
    else:
        grammar_score = 1.

    len_score = length_reward_smooth(input_len, len(text))

    rep_score = intra_repetition_reward(filtered_lines)
    
    return rep_score * len_score * grammar_score * lang_score * (0.6 * line_score + 0.4 * empty_score)

def make_format_reward(coef, use_unknown_ratio):
    def format_reward(completions, input_len=None, **kwargs):
        rewards = []
        
        for text, input_len_ in zip(completions, input_len):
            lines = text.split('\n')
            f_lines = filter_lines(lines)
            
            score = format_score(text, lines, f_lines, input_len_, use_unknown_ratio=use_unknown_ratio)
            rewards.append(coef * score)
        
        return rewards
    
    return format_reward
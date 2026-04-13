import numpy as np

from util import filter_lines

def len_score(lines, filtered_lines):
    # число строк (идеал = 4)
    line_score = np.exp(-abs(len(filtered_lines) - 4))   # плавно: 1 → 0.37 → 0.14
    
    # пустые строки
    empty = len(lines) - len(filtered_lines)
    empty_score = np.exp(-empty)
    
    return 0.6 * line_score + 0.4 * empty_score

def make_len_reward(coef):
    def len_reward(completions, **kwargs):
        rewards = []
        
        for text in completions:
            lines = text.split('\n')
            f_lines = filter_lines(lines)
            
            score = len_score(lines, f_lines)
            rewards.append(coef * score)
        
        return rewards
    
    return len_reward
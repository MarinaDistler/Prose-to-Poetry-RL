import os
import sys
import torch

current_dir = os.path.dirname(__file__)  # Папка, где лежит текущий скрипт
external_code_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'external_code', 'verslibre', 'py'))
sys.path.append(external_code_path)
from .rhyme_metric import check_rhyme_scheme, make_rhyme_reward
from .meter_metric import check_meter_fast, make_meter_reward
from .len_metric import len_score, make_len_reward
from .semantic_metric import embedding_sim_score, make_semantic_reward

from util import filter_lines


def compute_metrics(texts, rhyme_schemes):
    total_penalty = 0
    perfect_count = 0
    rhyme_score = 0

    for pred, rhyme_scheme in zip(texts, rhyme_schemes):
        lines = [line.strip() for line in pred.split("\n") if line.strip()]
        num_lines = len(lines)

        # Штраф за отклонение от 4 строк
        penalty = abs(num_lines - 4)
        total_penalty += penalty

        if num_lines == 4:
            perfect_count += 1

        rhyme_score += check_rhyme_scheme(lines[:4], scheme=rhyme_scheme)

    avg_penalty = total_penalty 

    return {
        "eval/avg_line_count_penalty": avg_penalty,       # чем меньше, тем лучше
        "eval/perfect_4_line_ratio": perfect_count,
        "eval/avg_rhyme_accuracy": rhyme_score,         # от 0 до 1, чем выше — тем лучше
    }

class ComputeAggMetrics:
    def __init__(self):
        self.metrics = {}
        self.count = 0
        self.zero_metrics()
    
    def zero_metrics(self):
        self.metrics = {
            "eval/avg_line_count_penalty": 0.,       # чем меньше, тем лучше
            "eval/perfect_4_line_ratio": 0.,
            "eval/avg_rhyme_accuracy": 0.,
        }
        self.count = 0
    
    def __call__(self, texts, schemes, compute_result=False):
        if compute_result:
            result = {}
            for key in self.metrics:
                result[key] = self.metrics[key] / self.count
            self.zero_metrics()
            return result
        batch_metrics = compute_metrics(
            texts, schemes
        )
        for key, value in batch_metrics.items():
            self.metrics[key] += value
        self.count += len(texts)
        return None



def make_metric_fn():
    return ComputeAggMetrics()


class ComputeMetricsRL:
    def __init__(self, args):
        self.coef_metrics = {}
        for name in ['rhyme_coef', 'meter_coef', 'len_coef', 'sem_coef']:
            self.coef_metrics[name] = args[name] 
        self.metrics = {}
        self.count = 0
        self.zero_metrics()
    
    def zero_metrics(self):
        self.metrics = {
            f"eval/rhyme_score": 0.,       # чем меньше, тем лучше
            f"eval/meter_score": 0.,
            f"eval/len_score": 0.,
            f"eval/semantic_score": 0.,
            f"eval/reward": 0.,
        }
        self.count = 0
    
    def __call__(self, texts, schemes, meters, emb_prose, compute_result=False):
        if compute_result:
            result = {}
            for key in self.metrics:
                result[key] = self.metrics[key] / self.count
            self.zero_metrics()
            return result
        
        rewards = [0] * len(texts)

        for i, (pred, rhyme_scheme, meter) in enumerate(zip(texts, schemes, meters)):
            lines = pred.split('\n')
            f_lines = filter_lines(lines)
            
            # RHYME
            if self.coef_metrics['rhyme_coef'] > 0. or self.eval == 'eval':
                rhyme_score = check_rhyme_scheme(f_lines, rhyme_scheme)
                self.metrics[f'eval/rhyme_score'] += rhyme_score
                rewards[i] += self.coef_metrics['rhyme_coef'] * rhyme_score

            # METER
            if self.coef_metrics['meter_coef'] > 0. or self.eval == 'eval':
                meter_score = check_meter_fast(f_lines, meter)
                self.metrics[f'eval/meter_score'] += meter_score
                rewards[i] += self.coef_metrics['meter_coef'] * meter_score

            # LENGTH / FORMAT (твоя структура)
            if self.coef_metrics['len_coef'] > 0. or self.eval == 'eval':
                len_score_ = len_score(lines, f_lines)   # или text, или f_lines — как у тебя реализовано
                self.metrics[f'eval/len_score'] += len_score_
                rewards[i] += self.coef_metrics['len_coef'] * len_score_

        rewards = torch.asarray(rewards)

        # SEMANTIC
        if self.coef_metrics['sem_coef'] > 0. or self.eval == 'eval':
            semantic_scores = embedding_sim_score(texts, emb_prose)
            self.metrics[f'eval/semantic_score'] += semantic_scores.sum()
            rewards += self.coef_metrics['sem_coef'] * semantic_scores

        self.metrics[f'eval/reward'] += rewards.sum()

        self.count += len(texts)
        return rewards


def make_metric_fn_rl(args):
    return ComputeMetricsRL(args)

def build_reward_functions(args):
    reward_funcs = []

    if args['rhyme_coef'] > 0:
        reward_funcs.append(make_rhyme_reward(args['rhyme_coef']))

    if args['meter_coef'] > 0:
        reward_funcs.append(make_meter_reward(args['meter_coef']))

    if args['len_coef'] > 0:
        reward_funcs.append(make_len_reward(args['len_coef']))

    if args['sem_coef'] > 0:
        reward_funcs.append(make_semantic_reward(args['sem_coef']))

    return reward_funcs

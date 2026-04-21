import os
import sys
import torch

current_dir = os.path.dirname(__file__)  # Папка, где лежит текущий скрипт
external_code_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'external_code', 'verslibre', 'py'))
sys.path.append(external_code_path)
from .rhyme_metric import check_rhyme_scheme, make_rhyme_reward
from .meter_metric import check_meter_fast, make_meter_reward
from .format_metric import format_score, make_format_reward
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

'''
class ComputeMetricsRL:
    def __init__(self, args):
        self.coef_metrics = {}
        for name in ['rhyme_coef', 'meter_coef', 'format_coef', 'sem_coef']:
            self.coef_metrics[name] = args[name] 
        self.metrics = {}
        self.count = 0
        self.zero_metrics()
    
    def zero_metrics(self):
        self.metrics = {
            f"eval/rhyme_score": 0.,       # чем меньше, тем лучше
            f"eval/meter_score": 0.,
            f"eval/format_score": 0.,
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
            if self.coef_metrics['format_coef'] > 0. or self.eval == 'eval':
                format_score_ = format_score(lines, f_lines)   # или text, или f_lines — как у тебя реализовано
                self.metrics[f'eval/format_score'] += format_score_
                rewards[i] += self.coef_metrics['format_coef'] * format_score_

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

'''

def compute_gate(sem_scores: torch.Tensor, format_scores: torch.Tensor,
    k_sem: float = 8.0, k_format: float = 5.0,
    sem_thr: float = 0.7, format_thr: float = 0.9,):
    """
    sem_scores: (batch,)
    format_scores: (batch,)
    returns: (batch,) gate in (0,1)
    """
    gate_sem = torch.sigmoid(k_sem * (sem_scores - sem_thr))
    gate_fmt = torch.sigmoid(k_format * (format_scores - format_thr))

    gates = gate_sem * gate_fmt

    return gates

def build_reward_functions_all(args):
    reward_funcs = []
    reward_weights = []

    if args.rhyme_coef > 0:
        reward_funcs.append(make_rhyme_reward(1., args.rhyme_alpha))
        reward_weights.append(args.rhyme_coef)

    if args.meter_coef > 0:
        reward_funcs.append(make_meter_reward(1.))
        reward_weights.append(args.meter_coef)

    if args.format_coef > 0:
        reward_funcs.append(make_format_reward(1.))
        reward_weights.append(args.format_coef)

    if args.sem_coef > 0:
        reward_funcs.append(make_semantic_reward(1.))
        reward_weights.append(args.sem_coef)

    return reward_funcs, reward_weights

def build_reward_functions(args):
    # --- base reward functions ---
    rhyme_fn = None
    meter_fn = None

    if args.rhyme_coef > 0:
        rhyme_fn = make_rhyme_reward(1., args.rhyme_alpha)

    if args.meter_coef > 0:
        meter_fn = make_meter_reward(1.)

    format_fn = make_format_reward(1.)
    sem_fn = make_semantic_reward(1.)

    def reward(log_metric=None, **kwargs):
        # --- 1. compute all base scores ---

        rhyme_scores = rhyme_fn(**kwargs) if rhyme_fn else None
        meter_scores = meter_fn(**kwargs) if meter_fn else None

        # --- 2. convert to torch ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=device)
        
        rhyme_t = to_tensor(rhyme_scores) if rhyme_scores is not None else 0
        meter_t = to_tensor(meter_scores) if meter_scores is not None else 0

        if args.sum_reward:
            format_scores = format_fn(**kwargs) if args.format_coef > 0 else None
            sem_scores = sem_fn(**kwargs) if args.sem_coef > 0 else None

            format_t = to_tensor(format_scores) if format_scores is not None else 0.0
            sem_t = to_tensor(sem_scores) if sem_scores is not None else 0.0

            reward = (
                args.rhyme_coef * rhyme_t +
                args.meter_coef * meter_t +
                args.format_coef * format_t +
                args.sem_coef * sem_t
            )

            gate = None
            form = None

        else:
            format_scores = format_fn(**kwargs) 
            sem_scores = sem_fn(**kwargs) 

            format_t   = to_tensor(format_scores)  
            sem_t   = to_tensor(sem_scores)  

            # --- 3. gate ---
            gate = compute_gate(sem_t, format_t,
                                k_sem=args.k_sem,
                                k_format=args.k_format,
                                sem_thr=args.sem_thr,
                                format_thr=args.format_thr)

            # --- 4. form reward ---
            form = args.rhyme_coef * rhyme_t + args.meter_coef * meter_t

            reward = (1 - args.sem_coef - args.format_coef) * gate * form  + args.sem_coef * sem_t + args.format_coef * format_t
        if log_metric:
            def log_stats(name, tensor):
                if tensor is None:
                    return
                log_metric(f"{name}_mean", tensor.mean().item())
                log_metric(f"{name}_std", tensor.std().item())

            log_stats("rhyme", rhyme_t)
            log_stats("meter", meter_t)
            log_stats("format", format_t)
            log_stats("semantic", sem_t)
            if not args.sum_reward:
                log_stats("gate", gate)
                log_stats("form", form)
                log_stats("gated_reward", reward)
            else:
                log_stats("sum_reward", reward)

            
        return reward.detach().cpu().tolist()

    return [reward]

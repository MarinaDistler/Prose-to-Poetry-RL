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
from .language_metric import make_language_reward

from util import text_to_lines


def compute_metrics(texts, rhyme_schemes, meters):
    total_penalty = 0
    perfect_count = 0
    rhyme_score = 0
    meter_score = 0

    for pred, rhyme_scheme, meter in zip(texts, rhyme_schemes, meters):
        lines = [line.strip() for line in pred.split("\n") if line.strip()]
        num_lines = len(lines)

        # Штраф за отклонение от 4 строк
        penalty = abs(num_lines - 4)
        total_penalty += penalty

        if num_lines == 4:
            perfect_count += 1

        rhyme_score += check_rhyme_scheme(lines, scheme=rhyme_scheme)
        meter_score += check_meter_fast(lines, meter, rhyme_scheme)

    avg_penalty = total_penalty 

    return {
        "eval/avg_line_count_penalty": avg_penalty,       # чем меньше, тем лучше
        "eval/perfect_4_line_ratio": perfect_count,
        "eval/avg_rhyme_accuracy": rhyme_score,         # от 0 до 1, чем выше — тем лучше
        "eval/avg_meter_score": meter_score,
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
            "eval/avg_meter_score": 0.,
        }
        self.count = 0
    
    def __call__(self, texts, schemes, meters, compute_result=False):
        if compute_result:
            result = {}
            for key in self.metrics:
                result[key] = self.metrics[key] / self.count
            self.zero_metrics()
            return result
        batch_metrics = compute_metrics(
            texts, schemes, meters
        )
        for key, value in batch_metrics.items():
            self.metrics[key] += value
        self.count += len(texts)
        return None



def make_metric_fn():
    return ComputeAggMetrics()


def compute_gate(sem_scores: torch.Tensor, format_scores: torch.Tensor, lang_scores: torch.Tensor,
    k_sem: float = 30.0, k_format: float = 25.0, k_lang: float = 25.0,
    sem_thr: float = 0.6, format_thr: float = 0.8, lang_thr: float = 0.55,):
    """
    sem_scores: (batch,)
    format_scores: (batch,)
    returns: (batch,) gate in (0,1)
    """
    gate_sem = torch.sigmoid(k_sem * (sem_scores - sem_thr))
    gate_fmt = torch.sigmoid(k_format * (format_scores - format_thr))
    gate_lng = torch.sigmoid(k_lang * (lang_scores - lang_thr))

    gates = gate_sem * gate_fmt * gate_lng

    return gates

def build_reward_functions(args):
    # --- base reward functions ---
    rhyme_fn = None
    meter_fn = None

    if args.rhyme_coef > 0:
        rhyme_fn = make_rhyme_reward(1., args.rhyme_alpha)

    if args.meter_coef > 0:
        meter_fn = make_meter_reward(1.)

    if args.lang_coef > 0 or not args.sum_reward:
        lang_fn = make_language_reward(1., path_base=args.from_pretrain)

    if args.format_coef > 0 or not args.sum_reward:
        format_fn = make_format_reward(1., use_unknown_ratio=args.unknown_ratio)

    if args.sem_coef > 0 or not args.sum_reward:
        sem_fn = make_semantic_reward(1.)

    def reward(log_metric=None, **kwargs):
        # --- 1. compute all base scores ---

        rhyme_scores = rhyme_fn(**kwargs) if rhyme_fn else None
        meter_scores = meter_fn(**kwargs) if meter_fn else None
        lang_scores = lang_fn(**kwargs) if lang_fn else None
        format_scores = format_fn(**kwargs) if format_fn else None
        sem_scores = sem_fn(**kwargs) if sem_fn else None

        # --- 2. convert to torch ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=device)
        
        rhyme_t = to_tensor(rhyme_scores) if rhyme_scores is not None else 0
        meter_t = to_tensor(meter_scores) if meter_scores is not None else 0
        lang_t = to_tensor(lang_scores) if lang_scores is not None else 0
        format_t = to_tensor(format_scores) if format_scores is not None else 0.0
        sem_t = to_tensor(sem_scores) if sem_scores is not None else 0.0

        sem_coef = args.sem_coef
        if args.sem_scheduler:
            progress = kwargs['trainer_state'].global_step / kwargs['trainer_state'].max_steps
            warmup_ratio = 0.7
            sem_scale = min(progress / warmup_ratio, 1.0)
            sem_coef = sem_coef * sem_scale

        if args.sum_reward:
            reward = (
                args.rhyme_coef * rhyme_t +
                args.meter_coef * meter_t +
                args.format_coef * format_t +
                sem_coef * sem_t +
                args.lang_coef * lang_t
            )

            gate = None
            form = None

        else:
            # --- 3. gate ---
            gate = compute_gate(sem_t, format_t, lang_t,
                                k_sem=args.k_sem,
                                k_format=args.k_format,
                                k_lang=args.k_lang,
                                sem_thr=args.sem_thr,
                                format_thr=args.format_thr,
                                lang_thr=args.lang_thr)

            # --- 4. form reward ---
            form = args.rhyme_coef * rhyme_t + args.meter_coef * meter_t

            reward = ((1 - args.sem_coef - args.format_coef - args.lang_coef) * gate * form  + 
                      sem_coef * sem_t + 
                      args.format_coef * format_t +
                      args.lang_coef * lang_t)
        if log_metric:
            def log_stats(name, tensor):
                if tensor is None or not torch.is_tensor(tensor):
                    return
                log_metric(f"{name}_mean", tensor.mean().item())
                log_metric(f"{name}_std", tensor.std().item())

            log_stats("rhyme", rhyme_t)
            log_stats("meter", meter_t)
            log_stats("format", format_t)
            log_stats("semantic", sem_t)
            log_stats("language", lang_t)
            if not args.sum_reward:
                log_stats("gate", gate)
                log_stats("form", form)
                log_stats("gated_reward", reward)
            else:
                log_stats("sum_reward", reward)
            log_metric(f"sem_coef", sem_coef)

            
        return reward.detach().cpu().tolist()

    return [reward]

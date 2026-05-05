import torch
from typing import List

from models import ModelQwen


def compute_nll_batch(
    prompts: List[str],
    completions: List[str],
    model, tokenizer,
    batch_size: int = 8,
):
    """
    Считает reward по NLL только на completion токенах,
    но модель получает prompt + completion как контекст.

    На вход:
        prompts      - список запросов (проза)
        completions  - список ответов (стихи)

    На выход:
        список score в диапазоне 0..1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    normalized_scores = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_completions = completions[i:i + batch_size]

        full_texts = [
            p + c for p, c in zip(batch_prompts, batch_completions)
        ]

        # токенизация полного текста
        enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # отдельно считаем длину prompt в токенах
        prompt_enc = tokenizer(
            batch_prompts,
            padding=False,
            add_special_tokens=False,
        )

        prompt_lengths = [
            len(x) for x in prompt_enc["input_ids"]
        ]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            loss = loss.view(shift_labels.size())

            # маска: считаем только completion токены
            token_mask = torch.zeros_like(shift_labels, dtype=torch.float32)

            for j in range(len(prompt_lengths)):
                pl = prompt_lengths[j]
                # после shift токен completion начинается с позиции pl-1
                start = max(pl - 1, 0)
                valid_len = int(attention_mask[j].sum().item()) - 1
                if start < valid_len:
                    token_mask[j, start:valid_len] = 1.0

            token_mask = token_mask.to(device)

            # зануляем всё кроме completion
            loss = loss * token_mask
            denom = token_mask.sum(dim=1).clamp(min=1)
            avg_nll = loss.sum(dim=1) / denom

            # нормализация в 0..1
            norm_score = 1 / (1 + torch.exp(2.0 * (avg_nll - 3.5)))

            normalized_scores.append(
                norm_score.detach().cpu()
            )

    return torch.cat(normalized_scores) 

def make_language_reward(coef, path_base):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_base = ModelQwen(quantization=True, path=path_base, generate=False)
    tokenizer = model_base.tokenizer
    model = model_base.model
    model.to(device)
    model.eval()

    def lang_reward(prompts, completions, **kwargs):
        scores = compute_nll_batch(prompts, completions, model, tokenizer)
        return (coef * scores).tolist()
    
    return lang_reward
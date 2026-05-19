# Импорт библиотек
import os, torch, sys
import argparse
import pandas as pd
from tqdm.auto import tqdm
from evaluate import load
import numpy as np
import re
import language_tool_python 
from typing import List

bertscore = load("bertscore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import print_options, text_to_lines
from metrics import get_rhyme_score, check_meter_fast, encode_sent, embedding_sim_score, format_score
from models import ModelQwen7B    


def compute_pure_perplexity(
    texts: List[str],
    model, 
    tokenizer,
    batch_size: int = 8
) -> List[float]:
    """
    Считает классическую перплексию (PPL) для списка изолированных текстов 
    (без контекста промптов) с учётом паддингов.

    На вход:
        texts      - список строк (сгенерированные стихотворения)
        model      - оценивающая языковая модель (находящаяся в eval())
        tokenizer  - токенизатор для модели

    На выход:
        список значений PPL (float) для каждой строки
    """
    device = next(model.parameters()).device  # автоматически определяем девайс модели
    perplexity_scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Токенизация батча с заполнением (padding) справа
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits

            # Сдвигаем логиты и лейблы для задачи Causal LM
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            # Считаем лосс отдельно для каждого токена без усреднения
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss = loss.view(shift_labels.size())

            # Маскируем токены паддинга, чтобы они не влияли на лосс текста
            # Так как мы сдвинули токены на 1, маску тоже сдвигаем
            token_mask = attention_mask[:, 1:]
            loss = loss * token_mask

            # Считаем средний лосс (NLL) для каждой строки индивидуально
            tokens_per_sequence = token_mask.sum(dim=1).clamp(min=1)
            avg_nll = loss.sum(dim=1) / tokens_per_sequence

            # Перплексия — это экспонента от среднего NLL
            ppl = torch.exp(avg_nll)

            # Переводим в обычный список float и сохраняем
            perplexity_scores.extend(ppl.detach().cpu().tolist())

    return perplexity_scores

def distinct_n(text, n=2):
    tokens = re.findall(r"\w+", text.lower())

    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

IMPORTANT_TYPES = [
    "grammar",
    "misspelling",
]
ALL_TYPES = ['uncategorized', "grammar", "misspelling", 'whitespace', 'typographical', 'duplication']

def grammar_error_rate(text, lang_tool):
    matches = lang_tool.check(text)
    filtered = [
        m for m in matches
        if m.rule_issue_type in IMPORTANT_TYPES
    ]

    total_errors = len(filtered)
    total_words = len(re.findall(r"\w+", text.lower()))

    if total_words == 0:
        return 0.0
    return total_errors / total_words

def eval_poetry(inputs, outputs, args):
    if args.grammar:
        lang_tool = language_tool_python.LanguageTool('ru-RU')
        result = pd.DataFrame(columns=['grammar'])
        
        for name, outputs_ in outputs.items():
            grammar = []
            for i, output in tqdm(enumerate(outputs_)):
                grammar.append(grammar_error_rate(output, lang_tool))
                
            res = {
                'grammar': np.mean(grammar),
            }
            result.loc[name] = res
            print(name, res)
        return result
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_base = ModelQwen7B(quantization=False, path='', generate=False)
    tokenizer = model_base.tokenizer
    model = model_base.model
    model.to(device)
    model.eval()

    result = pd.DataFrame(columns=['BERTscore', 'semantic_score', 'rhyme_score', 'meter_score', 
                                'format_score', 'perplexity', 'distinct_2'])
    input_emb = encode_sent(inputs['input'].tolist())
    for name, outputs_ in outputs.items():
        perpl = compute_pure_perplexity(outputs_.tolist(), model, tokenizer)
        bertscore_ = bertscore.compute(predictions=outputs_, references=inputs['input'], lang="ru")
        sem_scores = embedding_sim_score(outputs_.tolist(), input_emb).tolist()
        rhyme_scores = []
        meter_scores = []
        format_scores = []
        distinct_2 = []
        for i, output in tqdm(enumerate(outputs_)):
            lines = output.split('\n')
            f_lines = text_to_lines(output)
            rhyme_scores.append(get_rhyme_score(f_lines, inputs.iloc[i]['rhyme_scheme'], alpha=0.1))
            meter_scores.append(check_meter_fast(f_lines, inputs.iloc[i]['meter'], inputs.iloc[i]['rhyme_scheme']))
            format_scores.append(format_score(output, lines, f_lines, len(inputs.iloc[i]['input']), use_unknown_ratio=False))
            distinct_2.append(distinct_n(output, n=2))
            
        valid = np.sum(~np.isnan(meter_scores))
        total = len(meter_scores)
        print(f"Valid meter for {name}: {valid / total * 100}%")

        res = {
            'BERTscore': np.mean(bertscore_["f1"]),
            'semantic_score': np.mean(sem_scores),
            'rhyme_score': np.mean(rhyme_scores),
            'meter_score': np.nanmean(meter_scores),
            'format_score': np.mean(format_scores),
            'perplexity': np.mean(perpl),
            'distinct_2': np.mean(distinct_2),
        }
        result.loc[name] = res
        print(name, res)
    return result
    

def main(args):
    inputs = pd.read_csv(args.test_dataset)
    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Папка '{args.input_dir}' не существует.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    outputs = {}
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(args.input_dir, filename)
            name = filename[:-4]
            outputs[name] = pd.read_csv(file_path)[name].values
    res = eval_poetry(inputs, outputs, args) 
    print(res)
    res.to_csv(args.output_dir + f"{'_'.join(outputs.keys())}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--test_dataset', type=str, default='dataset/prosa_test_text.csv', help='Path to the test prose dataset')
    parser.add_argument('--input_dir', type=str, default='output/models_output/', help='Directory containing model outputs')
    parser.add_argument('--output_dir', type=str, default='output/', help='Where to save computed metrics')
    parser.add_argument('--grammar', action='store_true', help='If set, counts only grammar, if not all other metrics')

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
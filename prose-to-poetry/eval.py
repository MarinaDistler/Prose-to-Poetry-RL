# Импорт библиотек
import os
import sys
import argparse
import json
import pandas as pd
from tqdm.auto import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ModelTLite, ModelQwen
from util import print_options


def main(args):
    # Гарантируем наличие папки для вывода результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Пути к файлам результатов
    final_csv_path = os.path.join(args.output_dir, f'{args.name}.csv')
    temp_jsonl_path = os.path.join(args.output_dir, f'{args.name}_temp.jsonl')

    # Шаг 1. Загрузка датасета
    eval_data = pd.read_csv(args.test_dataset)
    
    # Шаг 2. Проверяем, есть ли уже сохраненный прогресс (для Resume)
    processed_indices = set()
    completed_results = {} # Сюда будем собирать {index: result_text}

    if os.path.exists(temp_jsonl_path):
        print(f"Обнаружен незавершенный прогресс в {temp_jsonl_path}. Загружаем...")
        with open(temp_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    idx = data['index']
                    processed_indices.add(idx)
                    completed_results[idx] = data['result']
        print(f"Успешно восстановлено {len(processed_indices)} из {len(eval_data)} шагов.")
        
        # Если вдруг уже все обработано, просто собираем финальный файл
        if len(processed_indices) == len(eval_data):
            print("Все данные уже были обработаны. Формируем финальный CSV...")
            save_final_csv(eval_data, completed_results, args.name, final_csv_path, temp_jsonl_path)
            return

    # Шаг 3. Инициализация модели (делаем только если есть что обрабатывать)
    quantization = False
    if args.model == 't-lite':
        model = ModelTLite(quantization=quantization, path=args.checkpoint, generate=args.generate)
    elif args.model == 'qwen':
        model = ModelQwen(quantization=quantization, path=args.checkpoint, generate=args.generate)
    if args.checkpoint != '':
        model.save_for_inference(args.checkpoint)
        model.load_for_inference(args.checkpoint)

    # Шаг 4. Цикл обработки с автосохранением
    # Открываем jsonl в режиме 'a' (append) — дозапись в конец файла
    with open(temp_jsonl_path, 'a', encoding='utf-8') as f_out:
        for i, row in tqdm(eval_data.iterrows(), total=len(eval_data)):
            # Если этот индекс уже обрабатывался в прошлый раз — пропускаем
            if i in processed_indices:
                continue

            # Генерация
            res_text = model.use(
                row['input'], 
                row['rhyme_scheme'], 
                row['meter'], 
                clean=not args.not_clean, 
                prompt_type=args.prompt_type
            )
            
            # Сохраняем в оперативку для текущего сеанса
            completed_results[i] = res_text
            
            # И сразу же пишем на диск (jsonl гарантирует, что строка не повредится при Killed)
            log_entry = {'index': i, 'result': res_text}
            f_out.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            f_out.flush() # Принудительно выталкиваем данные из буфера Python на диск

    # Шаг 5. Сборка финального CSV, если цикл успешно дошел до конца
    save_final_csv(eval_data, completed_results, args.name, final_csv_path, temp_jsonl_path)


def save_final_csv(eval_data, completed_results, column_name, csv_path, jsonl_path):
    """Строит итоговый DataFrame строго по индексам исходного датасета и удаляет временный файл"""
    result_list = [completed_results.get(i, "") for i in eval_data.index]
    
    df = pd.DataFrame({column_name: result_list}, index=eval_data.index)
    df.to_csv(csv_path)
    print(f"Финальный результат успешно сохранен в {csv_path}")
    
    # Удаляем временный файл, так как задача выполнена
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        print("Временный файл прогресса удален.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--name', type=str, default='t-lite', help='Saves the name.csv file with one column: name')
    parser.add_argument('--test_dataset', type=str, default='dataset/prosa_test_text.csv', help='Path to test dataset')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/', help='Output directory for results')
    parser.add_argument('--model', type=str, default='t-lite', choices=['t-lite', 'qwen'], help="Model type: 't-lite' or 'qwen'")
    parser.add_argument('--generate', action='store_true', help='If set, runs poetry generation instead of prose-to-poetry conversion')
    parser.add_argument('--not_clean', action='store_true', help='If set, disables postprocessing (doesn`t clean the output from markup)')
    parser.add_argument('--prompt_type', type=str, default='long', choices=['short', 'mid', 'long'], help='Choose prompt type')

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
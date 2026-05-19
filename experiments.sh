#!/bin/bash

CHECKPOINT_FILE="output/done.txt"
touch "$CHECKPOINT_FILE"

experiments=(
    "python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=output/SFT/sft_short_prompt_2ep-05-11-06-46/checkpoint-626 \
  --save_steps=90 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=1 \
  --log_steps=10 \
  --eval_steps=180 \
  --warmup_steps=370 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.05 --sem_coef=0.05 --lang_coef=0.05 \
  --train_mode=grpo --name_run=poetry_model_semsched_new_model --sem_scheduler"
  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.2 --meter_coef=0.2 --format_coef=0.2 --sem_coef=0.2\
  --train_mode=grpo --name_run=final_sum --lang_coef=0.2 --sum_reward"
  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.2 --meter_coef=0.2 --format_coef=0.2 --sem_coef=0.2\
  --train_mode=grpo --name_run=final_sum_scheduler --lang_coef=0.2 --sum_reward --coef_scheduler"
  
  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=final_gate --lang_coef=0."

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler --lang_coef=0.1 --coef_scheduler"

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler_nolang --lang_coef=0. --coef_scheduler --no_lang"

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler_unknown --lang_coef=0.1 --coef_scheduler --unknown_ratio"

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0. --meter_coef=1. --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_only_meter --lang_coef=0.1 --coef_scheduler"

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0. --meter_coef=1. --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=final_only_meter_gate --lang_coef=0."

  "python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=1. --meter_coef=0. --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_only_rhyme --lang_coef=0.1 --coef_scheduler"

  "python3 prose-to-poetry/train.py --model='qwen' \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_no_sft --lang_coef=0.1 --coef_scheduler --L_lang=0.14 --R_lang=0.44"

  "python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_newset_long_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_newset_long_prompt --lang_coef=0.1 --coef_scheduler --L_lang=0.31 --R_lang=0.61"

  "python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_newset_short_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_newset_short_prompt --lang_coef=0.1 --coef_scheduler --prompt_type=short --L_lang=0.32 --R_lang=0.62"

  "python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_oldpretrain_newset_long_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_oldpretrain_newset_long_prompt --lang_coef=0.1 --coef_scheduler --L_lang=0.33 --R_lang=0.63"
)

echo "Начинаю выполнение очереди экспериментов..."

for command in "${experiments[@]}"; do

    # достаём name_run
    name_run=$(echo "$command" | grep -oP -- '--name_run=\K[^ ]+')

    if grep -qx "$name_run" "$CHECKPOINT_FILE"; then
        echo ">>> Эксперимент $name_run уже выполнен. Пропускаю."
        continue
    fi

    echo ">>> Запуск: $name_run"
    echo ">>> Команда: $command"

    sudo systemctl stop gdm3

    # гарантия возврата GUI
    trap 'sudo systemctl start gdm3' EXIT

    eval "$command"
    exit_code=$?

    sudo systemctl start gdm3

    if [ $exit_code -eq 0 ]; then
        echo ">>> Успешно: $name_run"
        echo "$name_run" >> "$CHECKPOINT_FILE"
    else
        echo "!!! Ошибка: $name_run"
    fi

    echo ">>> Пауза 10 минут..."
    sleep 600

done

echo "Все эксперименты завершены!"
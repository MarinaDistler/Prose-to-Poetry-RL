# Prose-to-Poetry

This project is a developing version of [Prose-to-Poetry](https://github.com/MarinaDistler/Prose-to-Poetry) project.
This project focuses on transforming Russian **prose** into **structured poetry**, specifically **quatrains** with target meter (e.g., iamb) and rhyme scheme (e.g., ABAB) using RL.  
It leverages large language models (Qwen) with fine-tuning and markup to control rhythm and rhyme.



python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=30 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=1 \
  --log_steps=10 \
  --eval_steps=90 \
  --warmup_steps=30 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=poetry_model_no_sem_unknown

python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=30 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=1 \
  --log_steps=10 \
  --eval_steps=90 \
  --warmup_steps=30 \
  --lr=5e-6 --rhyme_coef=0.25 --meter_coef=0.25 --format_coef=0.25 --sem_coef=0.25\
  --train_mode=grpo --sum_reward --name_run=poetry_model_sum_reward

python3 prose-to-poetry/train.py \
  --model='qwen' \
  --save_steps=30 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=1 \
  --log_steps=10 \
  --eval_steps=90 \
  --warmup_steps=30 \
  --lr=5e-6 --rhyme_coef=0.45 --meter_coef=0.45 --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=base_model_long_prompt_no_sem

  sft pretrain
  python3 prose-to-poetry/train.py \
  --pretrain \
  --model='qwen' \
  --save_steps=150 --eval_steps=150\
  --train_dataset=dataset/trainset_pretrain \
  --epochs=1 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=320 \
  --lr=2e-5 --name_run=sft_long_prompt_pretrain

sft
  python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=output/qwen-05-22-17-18-pretrain/checkpoint-10738 \
  --save_steps=150 --eval_steps=150\
  --train_dataset=dataset/trainset.csv \
  --epochs=1 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=30 \
  --lr=5e-6  --name_run=sft_long_prompt 

  python3 prose-to-poetry/eval.py \
  --name=qwen \
  --model=qwen \
  --checkpoint=models/qwen \
  --markup=stanzas
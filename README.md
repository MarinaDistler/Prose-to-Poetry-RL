# Prose-to-Poetry

This project is a developing version of [Prose-to-Poetry](https://github.com/MarinaDistler/Prose-to-Poetry) project.
This project focuses on transforming Russian **prose** into **structured poetry**, specifically **quatrains** with target meter (e.g., iamb) and rhyme scheme (e.g., ABAB) using RL.  
It leverages large language models (Qwen) with fine-tuning and markup to control rhythm and rhyme.



python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=60 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=10 \
  --log_steps=10 \
  --eval_steps=180 \
  --warmup_steps=30 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --len_coef=0 --sem_coef=0.1\
  --train_mode=grpo

python3 prose-to-poetry/train.py \
  --model='qwen' \
  --save_steps=30 \
  --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv \
  --epochs=10 \
  --log_steps=10 \
  --eval_steps=90 \
  --warmup_steps=30 \
  --lr=5e-6 --rhyme_coef=0.45 --meter_coef=0.45 --len_coef=0 --sem_coef=0.1\
  --train_mode=grpo
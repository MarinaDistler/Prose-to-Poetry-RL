# Poetry Generation Pipeline: Run Commands & Configurations

This file provides a complete, sequential log of all execution commands across the three training phases: Pretraining, Supervised Fine-Tuning (SFT), and Reinforcement Learning (GRPO alignment). It serves as the definitive reference for reproducing the experimental environment.

---

## Thesis Naming vs. Code Configuration Mapping

Since  internal flags used during training differ slightly from the concise names used in the thesis text, use the table below to cross-reference the setups:

### 1. Pretraining & SFT Foundations
| Thesis Reference Name | CLI Run Name (`--name_run`) | Base Model / Initial Weights | Key Notes / Setup |
| :--- | :--- | :--- | :--- |
| — | `sft_pretrain_short_prompt` | `models/qwen` | Intermediate poetry pretraining (short instructions) |
| — | `sft_pretrain_long_prompt` | `models/qwen` | Intermediate poetry pretraining (long instructions) |
| **`base_sft`** | `models/qwen` | *Legacy Checkpoint* | Baseline SFT weights from previous project checkpoint |
| **`short_prompt`** | `sft_newset_short_prompt` | `models/sft_pretrain_short_prompt` | SFT initialized via short-prompt intermediate pretrain |
| **`long_prompt`** | `sft_newset_long_prompt` | `models/sft_pretrain_long_prompt` | SFT initialized via long-prompt intermediate pretrain |
| **`d2refined_sft`** | `sft_oldpretrain_newset_long_prompt`| `models/Qwen_G` | Legacy SFT base adapted to the new refined corpus |


### 2. Reinforcement Learning (GRPO Phase)
| Thesis Configuration Name | CLI Run Name (`--name_run`) | Base Model Checkpoint (`--from_pretrain`) | Key Reward Configurations & Flags |
| --- | --- | --- | --- |
| **`gate`** | `final_gate` | `models/qwen` | Gated mechanism baseline (`w_m = 0.5`, `w_r = 0.5`, `w_f = 0.1`) |
| **`sum`** | `final_sum` | `models/qwen` | Classic weighted sum of rewards fixed 0.2 weights  |
| **`sum_sc`** |  `final_sum_scheduler` | `models/qwen` | Classic weighted sum of rewards with scheduler |
| **`gate_sc`** | `final_gate_scheduler` | `models/qwen` | Gated mechanism with active reward scheduler (`w_s = 0.1`, `w_l = 0.1`) |
| **`gate_sc_lex`** | `final_gate_scheduler_unknown` | `models/qwen` | `gate_sc` setup + lexical penalty for non-dictionary tokens (`--unknown_ratio`) |
| **`only_meter`** | `final_only_meter` | `models/qwen` | Meter optimization only (`w_m = 1.0`, `w_r = 0.0`) with reward scheduler |
| **`only_meter_gate`** | `final_only_meter_gate` | `models/qwen` | Meter optimization only (`w_m = 1.0`, `w_r = 0.0`) using pure gating |
| **`only_rhyme`** | `final_only_rhyme` | `models/qwen` | Rhyme optimization only (`w_r = 1.0`, `w_m = 0.0`) with reward scheduler |
| **`no_lang`** | `final_gate_scheduler_nolang` | `models/qwen` | `gate_sc` configuration completely excluding language reward (`--no_lang`) |
| **`not_sft_rl`** | `final_no_sft` | *Not specified (vanilla Qwen)* | Pure GRPO alignment without any preliminary SFT stage |
| **`long_prompt_rl`** | `final_newset_long_prompt` | `models/sft_newset_long_prompt` | SFT initialization on a cleaned dataset using the original long instruction |
| **`short_prompt_rl`** | `final_newset_short_prompt` | `models/sft_newset_short_prompt` | SFT initialization on a cleaned dataset using a concise instruction (`--prompt_type=short`) |
| **`d2refined_sft_rl`** | `final_oldpretrain_newset_long_prompt` | `models/sft_oldpretrain_newset_long_prompt` | Legacy SFT base model additionally fine-tuned on the new refined dataset |


---

## Note on Checkpoint Paths & Manual Management

The training script automatically saves artifacts into nested step-specific directories (e.g., `SFT-pretrain/sft_long_prompt_pretrain-05-13-00-29-pretrain/checkpoint-XXXX`). 

To maintain clean and reproducible CLI commands across the pipeline phases, the final converged checkpoint from each run was **manually copied and consolidated** into a clean top-level directory before being passed to the next phase via the `--from_pretrain` argument. 

---

## Phase 1: Pretraining

Intermediate domain-specific learning runs performed on the primary poetic corpora.

### 1. `sft_pretrain_short_prompt`
```bash
python3 prose-to-poetry/train.py \
  --pretrain \
  --model='qwen' \
  --save_steps=150 --eval_steps=150 \
  --train_dataset=dataset/trainset_pretrain \
  --epochs=2 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=320 \
  --lr=2e-5 --name_run=sft_pretrain_short_prompt --prompt_type=short

```

### 2. `sft_pretrain_long_prompt`

```bash
python3 prose-to-poetry/train.py \
  --pretrain \
  --model='qwen' \
  --save_steps=150 --eval_steps=150 \
  --train_dataset=dataset/trainset_pretrain \
  --epochs=2 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=320 \
  --lr=2e-5 --name_run=sft_pretrain_long_prompt

```

---

## Phase 2: Supervised Fine-Tuning (SFT)

Fine-tuning pipelines mapping structural text transformations using targeted training instruction formats.

### 1. `sft_newset_short_prompt`

```bash
python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/sft_pretrain_short_prompt \
  --save_steps=150 --eval_steps=150 \
  --train_dataset=dataset/trainset.csv \
  --epochs=2 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=30 \
  --lr=5e-6 --name_run=sft_newset_short_prompt --prompt_type=short

```

### 2. `sft_newset_long_prompt`

```bash
python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/sft_pretrain_long_prompt \
  --save_steps=150 --eval_steps=150 \
  --train_dataset=dataset/trainset.csv \
  --epochs=2 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=30 \
  --lr=5e-6 --name_run=sft_newset_long_prompt

```

### 3. `sft_oldpretrain_newset_long_prompt`

```bash
python3 prose-to-poetry/train.py \
  --model='qwen' \
  --from_pretrain=models/Qwen_G \
  --save_steps=150 --eval_steps=150 \
  --train_dataset=dataset/trainset.csv \
  --epochs=2 \
  --log_steps=10 \
  --markup=stanzas \
  --warmup_steps=30 \
  --lr=5e-6 --name_run=sft_oldpretrain_newset_long_prompt

```

---


## Phase 3: Reinforcement Learning (GRPO)

Alignment runs executing reward-driven constraints for structural, metrical, and rhythmic optimization.


### 1. `final_sum`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.2 --meter_coef=0.2 --format_coef=0.2 --sem_coef=0.2\
  --train_mode=grpo --name_run=final_sum --lang_coef=0.2 --sum_reward

```

### 2. `final_sum_scheduler`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.2 --meter_coef=0.2 --format_coef=0.2 --sem_coef=0.2\
  --train_mode=grpo --name_run=final_sum_scheduler --lang_coef=0.2 --sum_reward --coef_scheduler

```

### 3. `final_gate`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=final_gate --lang_coef=0.

```

### 4. `final_gate_scheduler`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler --lang_coef=0.1 --coef_scheduler

```

### 5. `final_gate_scheduler_nolang`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler_nolang --lang_coef=0. --coef_scheduler --no_lang

```

### 6. `final_gate_scheduler_unknown`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_gate_scheduler_unknown --lang_coef=0.1 --coef_scheduler --unknown_ratio

```

### 7. `final_only_meter`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0. --meter_coef=1. --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_only_meter --lang_coef=0.1 --coef_scheduler

```

### 8. `final_only_meter_gate`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0. --meter_coef=1. --format_coef=0.1 --sem_coef=0.\
  --train_mode=grpo --name_run=final_only_meter_gate --lang_coef=0.

```

### 9. `final_only_rhyme`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --from_pretrain=models/qwen \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=1. --meter_coef=0. --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_only_rhyme --lang_coef=0.1 --coef_scheduler

```

### 10. `final_no_sft`

```bash
python3 prose-to-poetry/train.py --model='qwen' \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_no_sft --lang_coef=0.1 --coef_scheduler --L_lang=0.14 --R_lang=0.44

```

### 11. `final_newset_long_prompt`

```bash
python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_newset_long_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_newset_long_prompt --lang_coef=0.1 --coef_scheduler --L_lang=0.31 --R_lang=0.61

```

### 12. `final_newset_short_prompt`

```bash
python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_newset_short_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_newset_short_prompt --lang_coef=0.1 --coef_scheduler --prompt_type=short --L_lang=0.32 --R_lang=0.62

```

### 13. `final_oldpretrain_newset_long_prompt`

```bash
python3 prose-to-poetry/train.py --model='qwen' --from_pretrain=models/sft_oldpretrain_newset_long_prompt \
  --save_steps=90 --train_dataset=dataset/prosa_train_text.csv \
  --test_dataset=dataset/prosa_val_text.csv --epochs=1 \
  --log_steps=10 --eval_steps=180 --warmup_steps=100 \
  --lr=5e-6 --rhyme_coef=0.5 --meter_coef=0.5 --format_coef=0.1 --sem_coef=0.1\
  --train_mode=grpo --name_run=final_oldpretrain_newset_long_prompt --lang_coef=0.1 --coef_scheduler --L_lang=0.33 --R_lang=0.63

```
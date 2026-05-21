# Prose-to-Poetry-RL

This project focuses on transforming Russian **prose** into **structured poetry**, specifically targeting **quatrains** with fixed meters (such as iambic) and specific rhyme schemes (e.g., ABAB). 

The core architecture leverages Large Language Models (Qwen) optimized via supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) reinforcement learning,  to closely control rhythm, rhyme, and structural constraints.

> **Project Lineage:** This repository is an advanced, developed version of the original [Prose-to-Poetry](https://github.com/MarinaDistler/Prose-to-Poetry) project (which focused on the  impact of rhyme and stress markup). This current iteration expands the framework by introducing  reinforcement learning  pipeline.

---

## Project Resources

* **Pretrained & Aligned Models:** [Yandex.Disk Download Link](https://disk.360.yandex.ru/d/A5ZTOBWi0jmHjw)
* **Detailed Training Log & Experiments:** Refer to [RUN_COMMANDS.md](RUN_COMMANDS.md) for the exact execution sequence, hyperparameter configurations, and model mapping definitions.

---

## Quick Start

### 1. Install Dependencies
Set up the environment using the provided Conda configuration file and install the system-level phonetization dependency:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate prose2poetry

# Install system dependency for phonetic processing
sudo apt-get install espeak -y

```

### 2. Download Phonetization & Stress Resources

The rhythm evaluation module relies on external stress models and lemmatizers. Download the components from the [verslibre-files Google Drive folder](https://drive.google.com/drive/folders/1oIEM5_UuK-5phD5LtJqCPnSQ5CVQiOoM?usp=sharing) and map them to the corresponding paths inside the submodules:

```bash
# Copy lookup structures and neural accentuation weights
cp verslibre-files/word2lemma.pkl external_code/verslibre/models/
cp verslibre-files/accents.pkl external_code/verslibre/tmp/

# Move the stress evaluation model directories
mkdir -p external_code/verslibre/tmp/stress_model
cp -r verslibre-files/stress_model/* external_code/verslibre/tmp/stress_model/

```

---

## Project Structure

```
├── dataset/                        # Data artifacts for all pipeline phases
│   ├── all_poems.csv               # Raw poems with extracted meter annotations
│   ├── all_stanzas.csv              # Individual quatrains with meter & rhyme metadata
│   ├── trainset_pretrain/          # Target-domain quatrains (used for intermediate pretraining)
│   ├── trainset.csv                # Paired prose-poetry dataset (SFT training split)
│   ├── testset.csv                 # Paired prose-poetry dataset (SFT test split)
│   ├── prosa_train_text.csv        # Prose source segments for GRPO alignment training
│   ├── prosa_val_text.csv          # Prose source segments for GRPO alignment validation
│   ├── prosa_test_text.csv         # Prose source segments for final model assessment
│   └── prosa_remain_text.csv       # Remaining unallocated prose source segments
├── dataset-creation/               # Jupyter notebooks for corpus curation and automated metrics analysis
├── external_code/verslibre/        # Modified version of the Koziev/verslibre analysis engine
└── prose-to-poetry/               # Core execution scripts (Training, Generation, Reward functions)

```

---

## Training & Evaluation

### 1. Model Training

To retrain or fine-tune models across any of the three distinct processing phases (Intermediate Pretraining, SFT, or GRPO alignment), follow the step-by-step terminal instructions documented in **[RUN_COMMANDS.md](RUN_COMMANDS.md)**.

### 2. Model Inference & Evaluation

#### Mode A: Prose-to-Verse Generation (Default)

Transform input prose into structured verse using a fine-tuned model checkpoint:

```bash
python3 prose-to-poetry/eval.py \
  --name=qwen_prose_to_poetry \
  --model=qwen \
  --checkpoint=output/qwen/checkpoint-624 \
  --markup=rhyme_markup

```

#### Mode B: Unconditional Poetry Generation

Generate raw poetry directly from the domain-pretrained base model:

```bash
python3 prose-to-poetry/eval.py \
  --name=qwen_pure_generation \
  --model=qwen \
  --checkpoint=output/qwen-pretrain/checkpoint-10738 \
  --markup=rhyme_markup \
  --generate

```


### 3. Compute Metrics

The evaluation suite is divided into two separate execution modes to isolate heavy linguistic dependencies:

#### Option A: Core Poetic Metrics (Default)
Computes a comprehensive suite of automatic metrics analyzing semantic alignment with the source prose, poetic form constraints (phonetic meter adherence and rhyming quality), structural text formatting, as well as language fluency and token generation diversity. This mode runs out of the box using standard project dependencies:
```bash
python3 prose-to-poetry/compute_scores.py

```

#### Option B: Grammatical Correctness Only (Optional)

Evaluates  grammatical error rates. This analysis runs in isolation and requires setting up a specialized grammar-checking package that may demand additional manual troubleshooting:

```bash
python3 prose-to-poetry/compute_scores.py --grammar

```

---

## License

This repository is distributed under the terms of the **MIT License**, with the exception of the `external_code/verslibre` submodule, which retains its original **Unlicense** terms.


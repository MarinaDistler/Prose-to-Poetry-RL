from trl import GRPOTrainer, GRPOConfig
from datetime import datetime
import os
import sys

from util import print_options, start_tensorboard
from metrics import build_reward_functions

def train_grpo(model, tokenizer, datasets, peft_config, args):
    checkpoint = None if args.checkpoint == '' else args.checkpoint
    args.name_run = args.name_run if args.name_run != '' else args.model

    if checkpoint is not None:
        print(f'Use checkpoint {checkpoint}')
        run_name = f"{args.name_run}-from-{checkpoint}"
    else:
        run_name = f"{args.name_run}-{datetime.now().strftime('%m-%d-%H-%M')}"

    output_dir = os.path.join(
        args.output_dir,
        run_name + '-grpo'
    )

    config = vars(args)
    project = 'Poetry-GRPO' 

    writer, log_dir = start_tensorboard(
        run_name,
        project=project,
        config={key: config[key] for key in set(config.keys()) - {'name_run'}}
    )

    print_options(args, None)

    # --- batch логика ---
    if args.model == 't-lite':
        fact_batch_size = 2
    else:
        fact_batch_size = 4  

    # --- GRPO config ---
    training_arguments = GRPOConfig(
        output_dir=output_dir,

        per_device_train_batch_size=fact_batch_size,
        gradient_accumulation_steps=max(1, args.batch_size // fact_batch_size),

        learning_rate=args.lr,  # RL обычно поменьше
        num_train_epochs=args.epochs,

        logging_steps=max(1, args.log_steps // args.batch_size),
        save_steps=max(1, args.save_steps // args.batch_size),
        warmup_steps=args.warmup_steps,

        bf16=True,
        fp16=False,

        report_to="tensorboard",
        logging_dir=log_dir,

        save_total_limit=5,

        max_prompt_length=256,
        max_completion_length=256,

        num_generations=args.num_generations,  # ключевой параметр GRPO

        remove_unused_columns=False,

        log_completions=True,
        num_completions_to_print=10,

        eval_strategy='steps',
        eval_steps=args.save_steps // args.batch_size,
        per_device_eval_batch_size=fact_batch_size,
        num_generations_eval=args.num_generations // 2,
    )

    # --- reward function ---
    metric_fn = build_reward_functions(args)

    # --- trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,

        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],

        reward_funcs=metric_fn,

        peft_config=peft_config,

        args=training_arguments,
    )

    trainer.train(resume_from_checkpoint=checkpoint)

    return trainer
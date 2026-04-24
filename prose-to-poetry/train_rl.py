from trl import GRPOTrainer, GRPOConfig
from datetime import datetime
import os
import sys
from transformers.integrations import TensorBoardCallback

from util import print_options
from metrics import build_reward_functions
from trainer_callback import CustomTensorBoardCallback
from util import Tee

def train_grpo(model, tokenizer, datasets, peft_config, args):
    checkpoint = None if args.checkpoint == '' else args.checkpoint
    args.name_run = args.name_run if args.name_run != '' else args.model

    if checkpoint is not None:
        print(f'Use checkpoint {checkpoint}')
        run_name = f"{args.name_run}-from-{checkpoint}-grpo"
    else:
        run_name = f"{args.name_run}-{datetime.now().strftime('%m-%d-%H-%M')}-grpo"

    
    project = 'RL' 
    output_dir = os.path.join(
        args.output_dir,
        project,
        run_name 
    )
    log_file = os.path.join(output_dir, f"{run_name}.log")

    sys.stdout = Tee(log_file)
    sys.stderr = sys.stdout

    config = vars(args)
    print_options(args, None)

    # --- batch логика ---
    if args.model == 't-lite':
        fact_batch_size = 2
    else:
        fact_batch_size = 4  

    reward_func = build_reward_functions(args)

    # --- GRPO config ---
    training_arguments = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,

        per_device_train_batch_size=fact_batch_size,
        gradient_accumulation_steps=max(1, args.batch_size // fact_batch_size),

        learning_rate=args.lr,  # RL обычно поменьше
        num_train_epochs=args.epochs,

        logging_steps=max(1, args.log_steps),
        save_steps=max(1, args.save_steps),
        warmup_steps=args.warmup_steps,

        bf16=True,
        fp16=False,

        report_to="tensorboard",

        save_total_limit=100,

        max_completion_length=256,

        num_generations=args.num_generations,  # ключевой параметр GRPO

        remove_unused_columns=False,

        log_completions=True,
        num_completions_to_print=10,

        eval_strategy='steps',
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=fact_batch_size,
        num_generations_eval=1,

        beta=args.kl_beta
    )

    # --- trainer ---
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,

        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],

        reward_funcs=reward_func,

        peft_config=peft_config,

        args=training_arguments,

        callbacks=[CustomTensorBoardCallback(config)]
    )

    trainer.remove_callback(TensorBoardCallback)

    trainer.train(resume_from_checkpoint=checkpoint)

    return trainer
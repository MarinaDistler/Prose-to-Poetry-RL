# Импорт библиотек
import os, torch, sys
from transformers import (
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer, SFTConfig
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import print_options, start_tensorboard
from metrics import make_metric_fn
from trainer_callback import ChatGenerationCallback

class TrainDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.pad = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,  # Маскируем паддинг
            pad_to_multiple_of=8  # Выравниваем длину батча до кратности 8
        )

    def __call__(self, features):
        # Скопируем input_ids в labels до паддинга
        for feature in features:
            feature["labels"] = feature["input_ids"].copy()
        
        # Теперь можно паддить
        batch = self.pad(features)
        return batch

def train_sft(model, tokenizer, datasets, peft_config, clean_eval_data, args):
    checkpoint = None if args.checkpoint == '' else args.checkpoint
    args.name_run = args.name_run if args.name_run != '' else args.model
    if checkpoint is not None:
        print(f'Use checkpoint {checkpoint}')
        run_name = f"{args.name_run}-from-{checkpoint}"
    else:
        run_name = f"{args.name_run}-{datetime.now().strftime('%m-%d-%H-%M')}"
    output_dir = os.path.join(args.output_dir, run_name + ('-pretrain' if args.pretrain else ''))
    config = vars(args)
    project = 'Poetry-pretrain' if args.pretrain else 'Poetry'
    writer, log_dir = start_tensorboard(
        run_name, project=project, 
        config={key: config[key] for key in set(config.keys()) - {'name_run'}}
    )
    print_options(args, None)

    if args.model == 't-lite':
        fact_bach_size = 4
    else:
        fact_bach_size = 8

    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=fact_bach_size,
        per_device_eval_batch_size=fact_bach_size,
        gradient_accumulation_steps=args.batch_size // fact_bach_size,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        eval_strategy='steps',
        eval_steps=args.save_steps // args.batch_size,
        logging_steps=args.log_steps // args.batch_size,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        group_by_length=True,
        report_to="tensorboard",
        logging_dir=log_dir,
        save_strategy='steps',
        save_steps=args.save_steps // args.batch_size,              # Сохранять каждые 500 шагов
        save_total_limit=10 if args.pretrain else 2,          # Макс. число чекпоинтов (старые удаляются)
        load_best_model_at_end=True, # Загружать лучшую модель в конце
        metric_for_best_model="eval_loss",  # Критерий выбора лучшей модели
        max_seq_length=512,
        packing= False,
    )

    data_collator = TrainDataCollator(
        tokenizer=tokenizer,
        model=model,
    )

    callbacks = [ChatGenerationCallback(
        tokenizer, clean_eval_data, output_dir, batch_size=fact_bach_size,
        compute_metrics=make_metric_fn(), generate=args.pretrain,
    )]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        peft_config=peft_config, # сам адаптер, который создали ранее
        #dataset_text_field="chat",
        data_collator=data_collator, # был импортирован
        args=training_arguments,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    return trainer
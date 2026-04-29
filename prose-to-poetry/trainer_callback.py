import torch
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast
import json
from transformers.integrations import TensorBoardCallback
from torch.amp import autocast
from datasets import Dataset
import pandas as pd

from prompts import format_chat_template
from util import clean_responses

class CustomTensorBoardCallback(TensorBoardCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logged_config = False

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not self.logged_config:
            self.tb_writer.add_text(
                "config",
                f"<pre>{json.dumps(self.config, indent=2, ensure_ascii=False)}</pre>"
            )
            self.logged_config = True


    def _get_mem(self):
        if torch.cuda.is_available():
            return {
                "gpu/mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu/mem_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return {}
    
    def log_tb(self, dict, step):
        for name, value in dict.items():
            self.tb_writer.add_scalar(
                name,
                value,
                step
            )
        self.tb_writer.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        if not state.is_world_process_zero:
            return

        if torch.cuda.is_available():
            mem = self._get_mem()
            self.log_tb(mem, state.global_step)

def tokenize_from_chat_json(example, tokenizer, max_length=512, train=True):
    """
    JSON (system/user/assistant) -> input_ids + attention_mask + labels
    """
    
    messages = example["json"]

    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=not train
    )['input_ids']

    # truncate
    full_ids = full_ids[:max_length]
    
    attention_mask = [1] * len(full_ids)

    if train:
        if len(messages) != 3:
            raise ValueError(f"len of message shold be 3 not {len(messages)}")
        prompt_messages = messages[:-1]  # remove assistant

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True   # важно: даёт границу assistant
        )['input_ids']
        prompt_len = min(len(prompt_ids), max_length)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    else:
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
        }


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        indices = [f["index"] for f in features]
        features = [{k: v for k, v in f.items() if k != "index"} for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        batch["index"] = indices  # вернём индексы отдельно
        return batch



class ChatGenerationCallback(CustomTensorBoardCallback):
    def __init__(
                self, tokenizer, eval_dataset, output_dir, batch_size,
                show_examples=10, compute_metrics=None,
                max_new_tokens=256, generate=False, config=None
                ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.show_examples = show_examples
        self.output_dir = output_dir
        self.compute_metrics = compute_metrics
        self.batch_size = batch_size
        self.generate = generate
        self.max_new_tokens = max_new_tokens
        self.device = None  # будет установлен позже
        self.dataloader = None  # создадим в on_train_begin

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        self.device = model.device
        format_chat_template_ = lambda row: format_chat_template(row, self.tokenizer, self.generate, markup=None)
        self.eval_dataset = self.eval_dataset.apply(
            format_chat_template_, axis=1
        )
        self.eval_dataset['index'] = self.eval_dataset.index
        dataset = Dataset.from_pandas(self.eval_dataset[['json', 'index']])
        tokenize_ = lambda ex: tokenize_from_chat_json(ex, self.tokenizer, train=False)
        tokenized_dataset = dataset.map(tokenize_, remove_columns=["json"])
        collator = CustomDataCollator(tokenizer=self.tokenizer)
        self.dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator
        )

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not model:
            return

        to_table = []
        count = 0
        for batch in tqdm(self.dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            index = batch["index"]  

            with autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
            ]

            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            if count < self.show_examples:
                original_rows = [self.eval_dataset.loc[i] for i in index]
                for row, response in zip(original_rows, responses):
                    
                    to_table.append({
                        "User Prompt": row["user_prompt"],
                        "Generated": response,
                        "Ground Truth": '\n'.join(ast.literal_eval(row["stanzas"]))
                    })

            if self.compute_metrics:
                self.compute_metrics(
                    clean_responses(responses), 
                    self.eval_dataset.loc[index, 'rhyme_scheme'], 
                    self.eval_dataset.loc[index, 'meter'], 
                    compute_result=False
                )

            count += len(index)
        
        if self.compute_metrics:
            metrics = self.compute_metrics(None, None, None, compute_result=True)
            self.log_tb(metrics, state.global_step)

        # Логируем примеры
        if to_table:
            print(f"Global step: {state.global_step}")
            df = pd.DataFrame(to_table)
            print(df.to_markdown(index=False))




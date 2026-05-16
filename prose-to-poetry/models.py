from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
import re

from prompts import format_chat_template
from util import clean_responses

class BaseModel:
    def __init__(self, model_name, path, quantization=False, generate=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.quantization = quantization
        self.generate = generate
        if quantization: 
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
            )
        if path != '':
            self.model = PeftModel.from_pretrained(self.model, path)
            self.model.enable_adapter_layers()
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def save_for_inference(self, path):
        base_model = self.model

        base_model = base_model.merge_and_unload()
        base_model.save_pretrained(os.path.join(path, 'merged'), safe_serialization=True)

    def load_for_inference(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(path, 'merged'), 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def use(self, text, scheme='ABAB', meter='ямб', clean=True, prompt_type='long'):
        self.model.eval()
        row = {
            'input': text,
            'meter': meter,
            'rhyme_scheme': scheme,
        }
        prompt = format_chat_template(row, self.tokenizer, generate=self.generate, markup=None, 
                                                                prompt_type=prompt_type)['text']
        
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if clean:
            response = clean_responses([response])[0]
        return response

class ModelQwen(BaseModel):
    def __init__(self, quantization=False, path='', generate=False):
        super().__init__(
            'Qwen/Qwen2.5-3B-Instruct', path, 
            quantization, generate)

class ModelTLite(BaseModel):
    def __init__(self, quantization=False, path='', generate=False):
        super().__init__(
            "t-tech/T-lite-it-1.0", path, quantization, 
            generate)
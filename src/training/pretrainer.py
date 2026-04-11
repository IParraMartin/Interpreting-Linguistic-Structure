import os
import yaml

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

import wandb


class LMTrainer():
    def __init__(
            self,
            hf_model: str,
            dataset: DatasetDict,
            training_args: dict | str,
            model_args: dict | str,
            cache_dir: str,
            max_len: int = 1024,
        ) -> None:

        self.cache_dir = self.make_cache_dir(cache_dir)
        self.model_args = self.return_arguments(model_args)
        self.training_args = self.return_arguments(training_args)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_config(self.get_model_config())
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.training_args.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.dataset = dataset
        self.max_len = max_len
            
    def train(self) -> None:
        print("Tokenizing...")
        tokenized = self.get_tokenized_dataset(self.dataset)
        print("Launching training...")
        Trainer(
            model=self.model,
            args=TrainingArguments(**self.training_args),
            data_collator=self.get_data_collator(),
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['eval'],
            processing_class=self.tokenizer,
            callbacks=self.callbacks
        ).train()
        wandb.finish()

    def get_model_config(self) -> AutoConfig:
        model_type = self.model_args.get("model_type")
        if model_type is None:
            raise ValueError(
                "'model_args' must include 'model_type' (e.g., 'gpt2', 'llama'...)"
            )
        return AutoConfig.for_model(
            model_type,
            vocab_size=self.tokenizer.vocab_size,
            **{k: v for k, v in self.model_args.items() if k != "model_type"}
        )
    
    def get_tokenized_dataset(self, dataset: DatasetDict) -> DatasetDict:
        tokenized_dataset = dataset.map(
            self.tokenize,
            batched=True
        )
        return tokenized_dataset
    
    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )

    def tokenize(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
    
    def make_cache_dir(self, cache_dir: str = None) -> str:
        if cache_dir is None:
            cache_dir = 'cache'
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def return_arguments(self, args: str | dict) -> dict:
        if isinstance(args, dict):
            return args
        else:
            try:
                with open(args, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                raise ValueError("Argument should be a dict or path to a YAML file.")

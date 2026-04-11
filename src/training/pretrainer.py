import os
import yaml

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import DatasetDict

import wandb


class LMTrainer():
    """Trainer wrapper for causal language model pretraining using HuggingFace Transformers.

    Initializes a model from scratch using a specified architecture config, tokenizes
    the provided dataset, and runs a full training loop via the HuggingFace `Trainer`
    API. Training metrics are logged to Weights & Biases.

    Attributes:
        cache_dir (str): Path to the directory used for caching tokenizer downloads.
        model_args (dict): Parsed model architecture arguments.
        training_args (dict): Parsed training hyperparameters.
        tokenizer (AutoTokenizer): Tokenizer loaded from the specified HuggingFace hub model.
        model (AutoModelForCausalLM): Causal language model instantiated from config.
        dataset (DatasetDict): The raw dataset containing 'train' and 'eval' splits.
        max_len (int): Maximum token sequence length used during tokenization.
    """

    def __init__(
            self,
            hf_model: str,
            dataset: DatasetDict,
            training_args: dict | str,
            model_args: dict | str,
            cache_dir: str
        ) -> None:
        """Initializes LMTrainer by loading the tokenizer, building the model, and preparing config.

        The model is initialized from scratch using `AutoModelForCausalLM.from_config`,
        meaning no pretrained weights are loaded — only the architecture is borrowed from
        `model_args`. If `gradient_checkpointing` is enabled in `training_args`, it is
        activated on the model to reduce memory usage during backpropagation.

        Args:
            hf_model (str): HuggingFace model identifier used to load the tokenizer
                (e.g., ``'gpt2'``, ``'meta-llama/Llama-2-7b-hf'``).
            dataset (DatasetDict): A HuggingFace `DatasetDict` with at minimum ``'train'``
                and ``'eval'`` splits, each containing a ``'text'`` column.
            training_args (dict | str): Training hyperparameters. Either a dictionary
                of arguments compatible with `TrainingArguments`, or a path to a YAML
                file that will be parsed into one.
            model_args (dict | str): Model architecture arguments. Must include a
                ``'model_type'`` key (e.g., ``'gpt2'``). Either a dictionary or a path
                to a YAML file.
            cache_dir (str): Directory for caching downloaded tokenizer files. Will be
                created if it does not already exist.
        """
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
        self.max_len = self.model_args["n_positions"]

    def train(self) -> None:
        """Runs the full pretraining loop and finalizes the W&B run.

        Tokenizes the dataset, constructs a HuggingFace `Trainer` with the configured
        model, training arguments, and data collator, then calls `.train()`. After
        training completes, the active Weights & Biases run is closed via `wandb.finish()`.
        """
        print("Tokenizing...")
        tokenized = self.get_tokenized_dataset(self.dataset)
        print("Launching training...")
        eval_available = tokenized.get("eval")
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**self.training_args),
            data_collator=self.get_data_collator(),
            train_dataset=tokenized['train'],
            eval_dataset=eval_available,
            processing_class=self.tokenizer
        )
        try:
            trainer.train()
        finally:
            if self.training_args.get("report_to") == "wandb":
                wandb.finish()

    def get_model_config(self) -> AutoConfig:
        """Builds an `AutoConfig` for the target model architecture.

        Reads ``'model_type'`` from `model_args` to determine the architecture, then
        passes the remaining arguments (e.g., ``n_layer``, ``n_head``, ``n_embd``) as
        keyword arguments alongside the tokenizer's vocabulary size.

        Returns:
            AutoConfig: A configuration object for the specified model type, populated
                with the provided architecture hyperparameters and the tokenizer's
                vocabulary size.

        Raises:
            ValueError: If ``'model_type'`` is not present in `model_args`.
        """
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
        """Tokenizes all splits in the dataset using batched processing.

        Applies the `tokenize` method across the full `DatasetDict` using
        `datasets.DatasetDict.map` with batching enabled for efficiency.

        Args:
            dataset (DatasetDict): The raw dataset to tokenize. Each example must
                contain a ``'text'`` field.

        Returns:
            DatasetDict: A new `DatasetDict` with the same splits as the input,
                where each example has been replaced with tokenized tensors
                (``input_ids``, ``attention_mask``, etc.).
        """
        tokenized_dataset = dataset.map(
            self.tokenize,
            batched=True
        )
        return tokenized_dataset

    def get_data_collator(self) -> DataCollatorForLanguageModeling:
        """Constructs a data collator for causal (autoregressive) language modeling.

        Configures `DataCollatorForLanguageModeling` with masked language modeling
        disabled (``mlm=False``), making it suitable for decoder-only, next-token
        prediction objectives. Labels are automatically shifted inside the model.

        Returns:
            DataCollatorForLanguageModeling: A collator that pads batches and returns
                PyTorch tensors, with no token masking applied.
        """
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )

    def tokenize(self, batch: dict) -> dict:
        """Tokenizes a batch of raw text examples.

        Wraps the tokenizer call with fixed padding and truncation settings.
        Sequences shorter than `max_len` are right-padded; longer sequences
        are truncated to `max_len`.

        Args:
            batch (dict): A batch dictionary with a ``'text'`` key mapping to a list
                of raw string examples, as produced by `datasets.DatasetDict.map`.

        Returns:
            dict: A dictionary containing tokenizer outputs — at minimum
                ``'input_ids'`` and ``'attention_mask'`` — each padded or truncated
                to `self.max_len`.
        """
        return self.tokenizer(
            batch["text"],
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )

    def make_cache_dir(self, cache_dir: str = None) -> str:
        """Creates the cache directory if it does not exist and returns its path.

        If ``cache_dir`` is ``None``, the directory defaults to ``'cache'`` relative
        to the current working directory. The directory is created recursively with
        `os.makedirs`, so intermediate parent directories are also created as needed.

        Args:
            cache_dir (str, optional): Desired path for the cache directory.
                Defaults to ``'cache'`` if not provided.

        Returns:
            str: The resolved path to the cache directory.
        """
        if cache_dir is None:
            cache_dir = 'cache'
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def return_arguments(self, args: str | dict) -> dict:
        """Resolves configuration arguments from either a dict or a YAML file path.

        Allows training and model arguments to be specified inline as Python
        dictionaries or externalized to YAML config files for reproducibility.
        If a file path is provided, the file is parsed with `yaml.safe_load`.

        Args:
            args (str | dict): Either a dictionary of arguments to use directly,
                or a string path to a YAML file containing the arguments.

        Returns:
            dict: The resolved argument dictionary.

        Raises:
            ValueError: If ``args`` is a string but cannot be opened or parsed
                as a valid YAML file.
        """
        if isinstance(args, dict):
            return args
        else:
            try:
                with open(args, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                raise ValueError("Argument should be a dict or path to a YAML file.")

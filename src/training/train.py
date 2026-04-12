import argparse

from pretrainer import LMTrainer
from datasets import load_dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help="Pretraining script")
    p.add_argument(
        "--hf_model", 
        type=str, 
        default="gpt2", 
        help="HuggingFace model name or path (e.g. 'gpt2', 'gpt2-medium')."
    )
    p.add_argument(
        "--hf_dataset",
        type=str,
        default="BabyLM-community/BabyLM-2026-Strict",
        help="HuggingFace dataset identifier to load for training."
    )
    p.add_argument(
        "--training_args",
        type=str,
        default="configs/training/train.yaml",
        help="Path to YAML file containing HuggingFace TrainingArguments."
    )
    p.add_argument(
        "--model_args",
        type=str,
        default="configs/models/gpt2_small.yaml",
        help="Path to YAML file containing model architecture config."
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory for caching downloaded models and datasets."
    )
    args = p.parse_args()

    ds = load_dataset(args.hf_dataset, cache_dir=args.cache_dir)

    trainer = LMTrainer(
        hf_model=args.hf_model,
        dataset=ds,
        training_args=args.training_args,
        model_args=args.model_args,
        cache_dir=args.cache_dir
    )

    trainer.train()

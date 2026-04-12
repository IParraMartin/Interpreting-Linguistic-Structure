import argparse

from pretrainer import LMTrainer
from datasets import load_dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model", type=str, default="gpt2", required=True)
    p.add_argument("--hf_dataset", type=str, default="BabyLM-community/BabyLM-2026-Strict", required=True)
    p.add_argument("--training_args", type=str, default="configs/training/train.yaml", required=True)
    p.add_argument("--model_args", type=str, default="configs/models/gpt2_small.yaml", required=True)
    p.add_argument("--cache_dir", type=str, default="cache", required=True)
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

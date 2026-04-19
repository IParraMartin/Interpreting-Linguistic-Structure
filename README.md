# Interpreting-Linguistic-Structure

## How to train
Run this command on the terminal:

```command
python src/training/train.py \
--hf_model gpt2 \
--hf_dataset BabyLM-community/BabyLM-2026-Strict \
--training_args configs/training/TRAIN-CONFIG.yaml \
--model_args configs/models/MODEL-CONFIG.yaml \
--cache_dir cache
```

For multiple GPUs:
```command
TBD
```
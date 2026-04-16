import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer


# Pretrained models here:
# https://decoderesearch.github.io/SAELens/latest/pretrained_saes/

HF_MODEL = "gpt2"
RELEASE = "gpt2-small-res-jb"
ID = "blocks.6.hook_resid_pre"
HIDDEN_STATE_L = 6
DEVICE = "cpu"

text = "The house in the forest"

sae = SAE.from_pretrained(RELEASE, ID, device=DEVICE)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL, cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, cache_dir="cache")

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

acts = out.hidden_states[HIDDEN_STATE_L]
feature_acts = sae.encode(acts)

# Check which features are active
active_features = (feature_acts > 0).sum(dim=-1)
print(f"Active feats (N): {len(active_features.squeeze().numpy())}")
print(f"Active feats (idx): {active_features}")
print(f"Average L0: {active_features.float().mean().item()}")


if __name__ == "__main__":
    pass
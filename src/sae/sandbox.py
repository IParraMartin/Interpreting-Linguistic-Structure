from functools import partial

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


# Pretrained models here:
# https://decoderesearch.github.io/SAELens/latest/pretrained_saes/

HF_MODEL = "gpt2"
RELEASE = "gpt2-small-res-jb"
ID = "blocks.6.hook_resid_pre"
HIDDEN_STATE_L = 6
DEVICE = "cpu"

pair = {"sentence_good": "the dog barks", "sentence_bad": "the dog bark"}


# DUMMY EXAMPLE
text = "The dog barks"

# load models
sae = SAE.from_pretrained(RELEASE, ID, device=DEVICE)
model = HookedTransformer.from_pretrained(HF_MODEL, cache_dir="cache")

# tokenize
tokens = model.to_tokens(text)

# get acts
with torch.no_grad():
    _, cache = model.run_with_cache(tokens)
acts = cache["blocks.6.hook_resid_pre"]

# Check which features are active
active_features = (acts > 0).sum(dim=-1).cpu()
print(f"Active feats (N): {len(active_features.squeeze().numpy())}")
print(f"Active feats (idx): {active_features}")
print(f"Average L0: {active_features.float().mean().item()}")

# # Ablation
# def ablate_feature(sae_acts, hook, feature_id):
#     sae_acts[:, :, feature_id] = 0.0
#     return sae_acts

# # Ablate feature 1000 during forward pass
# logits = model.run_with_hooks_with_saes(
#     inputs,
#     saes=[sae],
#     fwd_hooks=[
#         ("blocks.12.hook_resid_post.hook_sae_acts_post",
#          partial(ablate_feature, feature_id=1000))
#     ]
# )


# if __name__ == "__main__":
#     pass
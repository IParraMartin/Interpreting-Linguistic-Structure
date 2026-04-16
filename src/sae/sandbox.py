from sae_lens import SAE
import torch

# https://decoderesearch.github.io/SAELens/latest/pretrained_saes/

RELEASE = "gpt2-small-res-jb"
ID = "blocks.6.hook_resid_pre"
DEVICE = "cpu"

sae = SAE.from_pretrained(RELEASE, ID, device=DEVICE)


activations = torch.randn(1, 5, 768, device="cpu")
feature_acts = sae.encode(activations)

# Check which features are active
active_features = (feature_acts > 0).sum(dim=-1)
print(f"Active feats (N): {len(active_features.squeeze().numpy())}")
print(f"Active feats (idx): {active_features}")
print(f"Average L0: {active_features.float().mean().item()}")




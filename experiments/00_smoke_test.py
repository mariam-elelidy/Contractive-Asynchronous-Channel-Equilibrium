import sys, time
sys.path.insert(0, "/home/claude/cace")
import numpy as np
import torch
from src.common import load_patients, make_splits, compute_value_norm_stats, AsyncBatcher, HourlyBatcher
from src.models import CACE, LinearSSM, SmallNeuralCDE, GRUD

patients, channels, static_params = load_patients()
print(f"{len(patients)} patients, {len(channels)} channels")
value_mean, value_std = compute_value_norm_stats(patients[:200], len(channels))

async_batcher = AsyncBatcher(patients, len(channels), value_mean, value_std, max_events=100)
hourly_batcher = HourlyBatcher(patients, len(channels), value_mean, value_std)

idx = list(range(8))
ab = async_batcher.get_batch(idx)
hb = hourly_batcher.get_batch(idx, rng=np.random.default_rng(0))
print("async batch shapes:", {k: v.shape for k, v in ab.items()})
print("hourly batch shapes:", {k: v.shape for k, v in hb.items()})

for name, model, batch in [
    ("CACE", CACE(len(channels)), ab),
    ("LinearSSM", LinearSSM(len(channels)), ab),
    ("SmallNeuralCDE", SmallNeuralCDE(len(channels)), ab),
    ("GRUD", GRUD(len(channels)), hb),
]:
    t0 = time.time()
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch["labels"])
    loss.backward()
    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:16s} out.shape={tuple(out.shape)} loss={loss.item():.4f} "
          f"params={n_params:,} fwd+bwd_time={dt:.3f}s")

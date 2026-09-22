"""
experiments/03_h1_order_invariance.py (batched, fast version)
================================================================
Same experiment as before, but batched across all candidate test patients
at once instead of looping one patient at a time (the per-patient version
was too slow for the Neural CDE baseline at scale).
"""
import sys, json
sys.path.insert(0, "/home/claude/cace")
import numpy as np
import torch

from src.common import load_patients, make_splits, compute_value_norm_stats, AsyncBatcher, HourlyBatcher
from src.models import CACE, LinearSSM, SmallNeuralCDE, GRUD

SEED = 0
torch.manual_seed(SEED)

patients, channels, static_params = load_patients()
n_channels = len(channels)
train_idx, val_idx, test_idx = make_splits(patients, seed=SEED)
value_mean, value_std = compute_value_norm_stats([patients[i] for i in train_idx], n_channels)
async_batcher = AsyncBatcher(patients, n_channels, value_mean, value_std)

models = {}
for name, cls in [("CACE", CACE), ("LinearSSM", LinearSSM), ("SmallNeuralCDE", SmallNeuralCDE)]:
    m = cls(n_channels)
    m.load_state_dict(torch.load(f"/home/claude/cace/outputs/{name}_checkpoint.pt"))
    m.eval()
    models[name] = m

rng = np.random.default_rng(123)

candidates = []
for idx in test_idx:
    times = patients[idx]["events"][:, 0]
    _, counts = np.unique(times, return_counts=True)
    if (counts > 1).sum() >= 1:
        candidates.append(idx)
print(f"{len(candidates)}/{len(test_idx)} test patients have >=1 same-timestamp tie group", flush=True)

N_TRIALS = 5
BATCH_SIZE = 100
summary = {}
raw = {}

for name, model in models.items():
    print(f"=== {name} ===", flush=True)
    all_deltas = []
    per_patient_max = []
    for start in range(0, len(candidates), BATCH_SIZE):
        idx_chunk = candidates[start:start + BATCH_SIZE]
        base_batch = async_batcher.get_batch(idx_chunk)
        with torch.no_grad():
            base_out = model(base_batch).numpy()
        chunk_deltas = np.zeros((len(idx_chunk), N_TRIALS))
        for trial in range(N_TRIALS):
            shuf_batch = async_batcher.get_batch(idx_chunk, shuffle_ties=True, rng=rng)
            with torch.no_grad():
                shuf_out = model(shuf_batch).numpy()
            chunk_deltas[:, trial] = np.abs(shuf_out - base_out)
        all_deltas.append(chunk_deltas)
        per_patient_max.extend(chunk_deltas.max(axis=1).tolist())
        print(f"  processed {start+len(idx_chunk)}/{len(candidates)}", flush=True)
    all_deltas = np.concatenate(all_deltas, axis=0)
    flat = all_deltas.flatten()
    summary[name] = {
        "n_patients": len(candidates), "n_trials_total": len(flat),
        "mean_abs_delta": float(flat.mean()),
        "max_abs_delta": float(flat.max()),
        "median_abs_delta": float(np.median(flat)),
        "frac_exactly_invariant_1e-5": float((flat < 1e-5).mean()),
    }
    raw[name] = {"per_patient_max_delta": per_patient_max}
    print(json.dumps(summary[name], indent=2), flush=True)

summary["GRUD"] = {"note": "not applicable in native hourly-binned format -- "
                           "same-timestamp cross-channel order has no representation "
                           "in a fixed hourly-bin input (values are overwritten by "
                           "construction), so the manipulation is not comparable."}

with open("/home/claude/cace/outputs/metrics/h1_order_invariance.json", "w") as f:
    json.dump(summary, f, indent=2)
with open("/home/claude/cace/outputs/predictions/h1_raw.json", "w") as f:
    json.dump(raw, f, indent=2)
print("Saved h1_order_invariance.json", flush=True)

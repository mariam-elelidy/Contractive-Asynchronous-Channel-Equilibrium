"""
experiments/04_h2_robustness.py
==================================
H2 (pre-registered): holding total information content fixed, CACE's
predictive performance degrades LESS than the baselines' as (a) observation
density is reduced (random event dropout) and (b) channel-arrival jitter is
injected (Gaussian noise added to event timestamps), because CACE decouples
within-channel dynamics from cross-channel scheduling.

Falsification: H2 is falsified if CACE's AUROC degradation curve is not
measurably shallower than the baselines' (i.e., statistically indistinguishable
or steeper).

Severity levels are fixed in advance: drop_frac in {0.0, 0.1, 0.2, 0.3, 0.4,
0.5}; jitter_std (minutes) in {0, 15, 30, 60, 120}. All four TRAINED models
(from experiment 02, no retraining) are evaluated on the SAME corrupted test
sets at each severity level.
"""
import sys, json
sys.path.insert(0, "/home/claude/cace")
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.common import load_patients, make_splits, compute_value_norm_stats, AsyncBatcher, HourlyBatcher
from src.models import CACE, LinearSSM, SmallNeuralCDE, GRUD

SEED = 0
DROP_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
JITTER_STDS = [0, 15, 30, 60, 120]  # minutes
BATCH_SIZE = 64

patients, channels, static_params = load_patients()
n_channels = len(channels)
train_idx, val_idx, test_idx = make_splits(patients, seed=SEED)
value_mean, value_std = compute_value_norm_stats([patients[i] for i in train_idx], n_channels)
async_batcher = AsyncBatcher(patients, n_channels, value_mean, value_std)
hourly_batcher = HourlyBatcher(patients, n_channels, value_mean, value_std)

models = {}
for name, cls in [("CACE", CACE), ("LinearSSM", LinearSSM), ("SmallNeuralCDE", SmallNeuralCDE), ("GRUD", GRUD)]:
    m = cls(n_channels)
    m.load_state_dict(torch.load(f"/home/claude/cace/outputs/{name}_checkpoint.pt"))
    m.eval()
    models[name] = m


def eval_auroc(model, batcher, is_hourly, drop_frac=0.0, jitter_std=0.0, seed=SEED):
    rng = np.random.default_rng(seed)
    all_logits, all_labels = [], []
    for start in range(0, len(test_idx), BATCH_SIZE):
        idx = test_idx[start:start + BATCH_SIZE]
        batch = batcher.get_batch(idx, drop_frac=drop_frac, jitter_std=jitter_std, rng=rng)
        with torch.no_grad():
            logits = model(batch)
        all_logits.append(logits.numpy()); all_labels.append(batch["labels"].numpy())
    all_logits = np.concatenate(all_logits); all_labels = np.concatenate(all_labels)
    return roc_auc_score(all_labels, all_logits)


results = {"dropout": {}, "jitter": {}}

print("=== Dropout sweep ===")
for name, model in models.items():
    is_hourly = (name == "GRUD")
    batcher = hourly_batcher if is_hourly else async_batcher
    curve = []
    for frac in DROP_FRACS:
        auroc = eval_auroc(model, batcher, is_hourly, drop_frac=frac)
        curve.append(auroc)
        print(f"  {name:16s} drop_frac={frac:.1f} AUROC={auroc:.4f}")
    results["dropout"][name] = curve

print("=== Jitter sweep ===")
for name, model in models.items():
    is_hourly = (name == "GRUD")
    batcher = hourly_batcher if is_hourly else async_batcher
    curve = []
    for jit in JITTER_STDS:
        auroc = eval_auroc(model, batcher, is_hourly, jitter_std=jit)
        curve.append(auroc)
        print(f"  {name:16s} jitter_std={jit:4d}min AUROC={auroc:.4f}")
    results["jitter"][name] = curve

results["drop_fracs"] = DROP_FRACS
results["jitter_stds"] = JITTER_STDS

with open("/home/claude/cace/outputs/metrics/h2_robustness.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved h2_robustness.json")

print("\n=== Relative degradation (dropout 0.0 -> 0.5) ===")
for name in models:
    c = results["dropout"][name]
    rel_drop = (c[0] - c[-1]) / c[0]
    print(f"  {name:16s} AUROC {c[0]:.4f} -> {c[-1]:.4f}  relative_degradation={rel_drop:.4f}")

print("\n=== Relative degradation (jitter 0 -> 120min) ===")
for name in models:
    c = results["jitter"][name]
    rel_drop = (c[0] - c[-1]) / c[0]
    print(f"  {name:16s} AUROC {c[0]:.4f} -> {c[-1]:.4f}  relative_degradation={rel_drop:.4f}")

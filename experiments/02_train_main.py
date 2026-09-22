"""
experiments/02_train_main.py
==============================
Trains all four models (CACE, LinearSSM, SmallNeuralCDE, GRUD) on real
in-hospital mortality prediction from the first 48h of ICU data, with
hyperparameters fixed in advance in PREREGISTRATION.md. This establishes
construct validity (do the models learn the real task at all?) -- it is
NOT itself one of the pre-registered structural hypotheses (H1/H2/H3).

Saves trained model checkpoints (needed for H1/H2) and a results table.
"""
import sys, time, json
sys.path.insert(0, "/home/claude/cace")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from src.common import load_patients, make_splits, compute_value_norm_stats, AsyncBatcher, HourlyBatcher
from src.models import CACE, LinearSSM, SmallNeuralCDE, GRUD

SEED = 0
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
WD = 1e-5

torch.manual_seed(SEED)
np.random.seed(SEED)

patients, channels, static_params = load_patients()
n_channels = len(channels)
train_idx, val_idx, test_idx = make_splits(patients, seed=SEED)
print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

value_mean, value_std = compute_value_norm_stats([patients[i] for i in train_idx], n_channels)

async_batcher = AsyncBatcher(patients, n_channels, value_mean, value_std)
hourly_batcher = HourlyBatcher(patients, n_channels, value_mean, value_std)

train_labels = np.array([patients[i]["label"] for i in train_idx])
pos_weight = torch.tensor((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))
print(f"pos_weight (for BCE) = {pos_weight.item():.3f}")


def run_epoch(model, batcher, idx_list, train=True, is_hourly=False, rng=None):
    if train:
        model.train()
    else:
        model.eval()
    rng = rng or np.random.default_rng(SEED)
    order = np.array(idx_list)
    if train:
        rng.shuffle(order)
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        if is_hourly:
            batch = batcher.get_batch(batch_idx, rng=rng)
        else:
            batch = batcher.get_batch(batch_idx)
        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        total_loss += loss.item() * len(batch_idx)
        n_batches += 1
        all_logits.append(logits.detach().numpy())
        all_labels.append(batch["labels"].numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_logits)
    auprc = average_precision_score(all_labels, all_logits)
    return total_loss / len(order), auroc, auprc


results = {}
for name, model, batcher, is_hourly in [
    ("CACE", CACE(n_channels), async_batcher, False),
    ("LinearSSM", LinearSSM(n_channels), async_batcher, False),
    ("SmallNeuralCDE", SmallNeuralCDE(n_channels), async_batcher, False),
    ("GRUD", GRUD(n_channels), hourly_batcher, True),
]:
    print(f"\n=== Training {name} ===")
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    history = []
    for epoch in range(EPOCHS):
        tr_loss, tr_auroc, tr_auprc = run_epoch(model, batcher, train_idx, train=True, is_hourly=is_hourly, rng=rng)
        val_loss, val_auroc, val_auprc = run_epoch(model, batcher, val_idx, train=False, is_hourly=is_hourly, rng=rng)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_auroc": tr_auroc,
                         "val_loss": val_loss, "val_auroc": val_auroc})
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  train_loss={tr_loss:.4f} train_auroc={tr_auroc:.4f} "
              f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}")
    train_time = time.time() - t0
    test_loss, test_auroc, test_auprc = run_epoch(model, batcher, test_idx, train=False, is_hourly=is_hourly, rng=rng)
    print(f"  FINAL TEST: loss={test_loss:.4f} auroc={test_auroc:.4f} auprc={test_auprc:.4f} "
          f"(train_time={train_time:.1f}s)")
    n_params = sum(p.numel() for p in model.parameters())
    results[name] = {
        "test_loss": test_loss, "test_auroc": test_auroc, "test_auprc": test_auprc,
        "n_params": n_params, "train_time_sec": train_time, "history": history,
    }
    torch.save(model.state_dict(), f"/home/claude/cace/outputs/{name}_checkpoint.pt")

with open("/home/claude/cace/outputs/metrics/main_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== SUMMARY ===")
for name, r in results.items():
    print(f"{name:16s} AUROC={r['test_auroc']:.4f} AUPRC={r['test_auprc']:.4f} "
          f"params={r['n_params']:,} train_time={r['train_time_sec']:.1f}s")

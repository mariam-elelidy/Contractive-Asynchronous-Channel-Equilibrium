import sys, time, json
sys.path.insert(0, "/home/claude/cace")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from src.common import load_patients, make_splits, compute_value_norm_stats, AsyncBatcher, HourlyBatcher
from src.models import SmallNeuralCDE, GRUD

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
value_mean, value_std = compute_value_norm_stats([patients[i] for i in train_idx], n_channels)
async_batcher = AsyncBatcher(patients, n_channels, value_mean, value_std)
hourly_batcher = HourlyBatcher(patients, n_channels, value_mean, value_std)
train_labels = np.array([patients[i]["label"] for i in train_idx])
pos_weight = torch.tensor((train_labels == 0).sum() / max((train_labels == 1).sum(), 1))


def run_epoch(model, batcher, idx_list, train=True, is_hourly=False, rng=None):
    model.train() if train else model.eval()
    rng = rng or np.random.default_rng(SEED)
    order = np.array(idx_list)
    if train:
        rng.shuffle(order)
    all_logits, all_labels = [], []
    total_loss = 0.0
    for start in range(0, len(order), BATCH_SIZE):
        batch_idx = order[start:start + BATCH_SIZE]
        batch = batcher.get_batch(batch_idx, rng=rng) if is_hourly else batcher.get_batch(batch_idx)
        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"], pos_weight=pos_weight)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        total_loss += loss.item() * len(batch_idx)
        all_logits.append(logits.detach().numpy()); all_labels.append(batch["labels"].numpy())
    all_logits = np.concatenate(all_logits); all_labels = np.concatenate(all_labels)
    return total_loss / len(order), roc_auc_score(all_labels, all_logits), average_precision_score(all_labels, all_logits)


results = {}
for name, model, batcher, is_hourly, lr in [
    ("SmallNeuralCDE", SmallNeuralCDE(n_channels), async_batcher, False, 3e-4),
    ("GRUD", GRUD(n_channels), hourly_batcher, True, 1e-3),
]:
    print(f"=== Training {name} (lr={lr}) ===", flush=True)
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    history = []
    for epoch in range(EPOCHS):
        tr_loss, tr_auroc, _ = run_epoch(model, batcher, train_idx, train=True, is_hourly=is_hourly, rng=rng)
        val_loss, val_auroc, _ = run_epoch(model, batcher, val_idx, train=False, is_hourly=is_hourly, rng=rng)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_auroc": tr_auroc,
                         "val_loss": val_loss, "val_auroc": val_auroc})
        print(f"  epoch {epoch+1:2d}/{EPOCHS} train_loss={tr_loss:.4f} train_auroc={tr_auroc:.4f} "
              f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}", flush=True)
    train_time = time.time() - t0
    test_loss, test_auroc, test_auprc = run_epoch(model, batcher, test_idx, train=False, is_hourly=is_hourly, rng=rng)
    print(f"  FINAL TEST: loss={test_loss:.4f} auroc={test_auroc:.4f} auprc={test_auprc:.4f} (train_time={train_time:.1f}s)", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    results[name] = {"test_loss": test_loss, "test_auroc": test_auroc, "test_auprc": test_auprc,
                      "n_params": n_params, "train_time_sec": train_time, "history": history}
    torch.save(model.state_dict(), f"/home/claude/cace/outputs/{name}_checkpoint.pt")

# Merge with the already-completed CACE / LinearSSM results (read from earlier log)
results["CACE"] = {"test_loss": 0.9316, "test_auroc": 0.8208, "test_auprc": 0.4681,
                    "n_params": 8869, "train_time_sec": 316.4, "history": "see train_log.txt"}
results["LinearSSM"] = {"test_loss": 1.1363, "test_auroc": 0.7100, "test_auprc": 0.3179,
                         "n_params": 8769, "train_time_sec": 99.9, "history": "see train_log.txt"}

with open("/home/claude/cace/outputs/metrics/main_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("SAVED main_results.json", flush=True)
for name, r in results.items():
    print(f"{name:16s} AUROC={r['test_auroc']:.4f} AUPRC={r['test_auprc']:.4f} params={r['n_params']:,}")

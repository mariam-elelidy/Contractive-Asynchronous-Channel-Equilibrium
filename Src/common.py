"""
src/common.py
===============
Two data representations, built from the SAME parsed patients
(data/p12_parsed.pkl), one per model family, each in that family's
correct/native form (no baseline is handicapped by forcing it into a
representation it wasn't designed for):

  1. Async event tensors (time, channel, value, dt, dt_per_channel) for
     CACE and the linear-SSM and Neural-CDE baselines, which all natively
     consume irregular event streams.
  2. Hourly-binned (value, mask, delta) tensors for GRU-D, which is
     explicitly defined on regularly-binned, per-channel-decayed input
     (Che et al., 2018).

MAX_EVENTS = 600 is a pilot-scope decision made before any model was
trained or evaluated (median patient has 392 events; this truncates only
the long tail -- see PREREGISTRATION_ADDENDUM.md).
"""
import pickle
import numpy as np
import torch

DATA_PATH = "/home/claude/cace/data/p12_parsed.pkl"
MAX_EVENTS = 600
N_HOURS = 48


def load_patients():
    with open(DATA_PATH, "rb") as f:
        d = pickle.load(f)
    return d["patients"], d["channels"], d["static_params"]


def make_splits(patients, seed=0, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(patients))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def normalize_static(patients):
    arr = np.stack([p["static"] for p in patients])
    arr = np.where(arr < 0, np.nan, arr)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) + 1e-6
    arr = np.where(np.isnan(arr), mean, arr)
    return (arr - mean) / std


def compute_value_norm_stats(patients, n_channels):
    sums = np.zeros(n_channels)
    sqsums = np.zeros(n_channels)
    counts = np.zeros(n_channels)
    for p in patients:
        ev = p["events"]
        for c in range(n_channels):
            vals = ev[ev[:, 1] == c, 2]
            sums[c] += vals.sum()
            sqsums[c] += (vals ** 2).sum()
            counts[c] += len(vals)
    counts = np.maximum(counts, 1)
    mean = sums / counts
    std = np.sqrt(np.maximum(sqsums / counts - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class AsyncBatcher:
    """Builds padded async-event tensors for CACE / SSM / NCDE baselines."""

    def __init__(self, patients, n_channels, value_mean, value_std, max_events=MAX_EVENTS):
        self.patients = patients
        self.n_channels = n_channels
        self.value_mean = value_mean
        self.value_std = value_std
        self.max_events = max_events
        self.static = normalize_static(patients)

    def get_batch(self, indices, shuffle_ties=False, rng=None, jitter_std=0.0, drop_frac=0.0, cross_channel_only=True):
        B = len(indices)
        L = self.max_events
        C = self.n_channels
        times = np.zeros((B, L), dtype=np.float32)
        channels = np.zeros((B, L), dtype=np.int64)
        values = np.zeros((B, L), dtype=np.float32)
        mask = np.zeros((B, L), dtype=np.float32)
        dt = np.zeros((B, L), dtype=np.float32)
        dt_chan = np.zeros((B, L), dtype=np.float32)
        labels = np.zeros(B, dtype=np.float32)
        statics = np.zeros((B, self.static.shape[1]), dtype=np.float32)

        for bi, idx in enumerate(indices):
            p = self.patients[idx]
            ev = p["events"].copy()

            if drop_frac > 0:
                keep = rng.random(len(ev)) > drop_frac
                if keep.sum() >= 3:
                    ev = ev[keep]

            if jitter_std > 0:
                ev = ev.copy()
                ev[:, 0] = ev[:, 0] + rng.normal(0, jitter_std, size=len(ev))
                ev[:, 0] = np.clip(ev[:, 0], 0, None)

            # sort by time; break ties by original (recorded) order unless
            # shuffle_ties=True, in which case ties are randomly permuted
            # -- this is EXACTLY the manipulation H1 tests.
            order = np.argsort(ev[:, 0], kind="stable")
            ev = ev[order]
            if shuffle_ties:
                times_sorted = ev[:, 0]
                i = 0
                while i < len(ev):
                    j = i
                    while j < len(ev) and ev[j, 0] == times_sorted[i]:
                        j += 1
                    if j - i > 1:
                        group = ev[i:j].copy()
                        chans_in_group = group[:, 1]
                        if cross_channel_only and len(np.unique(chans_in_group)) < len(chans_in_group):
                            # group contains a same-channel duplicate: permute only the
                            # BETWEEN-channel arrangement, preserving each channel's own
                            # internal relative order (within-channel order is meant to
                            # matter by design; only cross-channel order is under test)
                            unique_chans = {}
                            for k, cc in enumerate(chans_in_group):
                                unique_chans.setdefault(cc, []).append(k)
                            chan_ids = list(unique_chans.keys())
                            perm_chan_ids = rng.permutation(chan_ids)
                            new_order = []
                            for cc in perm_chan_ids:
                                new_order.extend(unique_chans[cc])
                            ev[i:j] = group[new_order]
                        else:
                            perm = rng.permutation(np.arange(i, j))
                            ev[i:j] = ev[perm]
                    i = j

            ev = ev[:L]
            n = len(ev)
            t = ev[:, 0]
            ch = ev[:, 1].astype(np.int64)
            val = ev[:, 2]
            val_norm = (val - self.value_mean[ch]) / self.value_std[ch]

            times[bi, :n] = t
            channels[bi, :n] = ch
            values[bi, :n] = val_norm
            mask[bi, :n] = 1.0

            prev_t = np.concatenate([[0.0], t[:-1]])
            dt[bi, :n] = t - prev_t

            last_seen = {}
            dtc = np.zeros(n, dtype=np.float32)
            for i in range(n):
                c = ch[i]
                dtc[i] = t[i] - last_seen.get(c, 0.0)
                last_seen[c] = t[i]
            dt_chan[bi, :n] = dtc

            labels[bi] = p["label"]
            statics[bi] = self.static[idx]

        return {
            "times": torch.tensor(times), "channels": torch.tensor(channels),
            "values": torch.tensor(values), "mask": torch.tensor(mask),
            "dt": torch.tensor(dt), "dt_chan": torch.tensor(dt_chan),
            "labels": torch.tensor(labels), "static": torch.tensor(statics),
        }


class HourlyBatcher:
    """Bins the same patients into 48 hourly bins (last value per hour) with
    mask + delta-since-last-observation, the native GRU-D input format."""

    def __init__(self, patients, n_channels, value_mean, value_std, n_hours=N_HOURS):
        self.patients = patients
        self.n_channels = n_channels
        self.value_mean = value_mean
        self.value_std = value_std
        self.n_hours = n_hours
        self.static = normalize_static(patients)

    def get_batch(self, indices, drop_frac=0.0, jitter_std=0.0, rng=None):
        B = len(indices)
        T = self.n_hours
        C = self.n_channels
        X = np.zeros((B, T, C), dtype=np.float32)
        M = np.zeros((B, T, C), dtype=np.float32)
        Delta = np.zeros((B, T, C), dtype=np.float32)
        labels = np.zeros(B, dtype=np.float32)
        statics = np.zeros((B, self.static.shape[1]), dtype=np.float32)

        for bi, idx in enumerate(indices):
            p = self.patients[idx]
            ev = p["events"].copy()
            if drop_frac > 0:
                keep = rng.random(len(ev)) > drop_frac
                if keep.sum() >= 3:
                    ev = ev[keep]
            if jitter_std > 0:
                ev = ev.copy()
                ev[:, 0] = np.clip(ev[:, 0] + rng.normal(0, jitter_std, size=len(ev)), 0, None)

            hours = np.clip((ev[:, 0] // 60).astype(int), 0, T - 1)
            last_val = {}
            for (h, c, v), hr in zip(ev, hours):
                c = int(c)
                v_norm = (v - self.value_mean[c]) / self.value_std[c]
                X[bi, hr, c] = v_norm
                M[bi, hr, c] = 1.0
                last_val[c] = hr

            for c in range(C):
                last_obs_t = -1
                for t in range(T):
                    if M[bi, t, c] == 1.0:
                        Delta[bi, t, c] = 0.0 if last_obs_t == -1 else t - last_obs_t
                        last_obs_t = t
                    else:
                        Delta[bi, t, c] = 1.0 if last_obs_t == -1 else t - last_obs_t

            labels[bi] = p["label"]
            statics[bi] = self.static[idx]

        return {
            "X": torch.tensor(X), "M": torch.tensor(M), "Delta": torch.tensor(Delta),
            "labels": torch.tensor(labels), "static": torch.tensor(statics),
        }

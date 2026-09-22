"""
src/data.py
============
Parses the RAW PhysioNet/CinC 2012 per-patient files (set-a: 4000 real ICU
stays) into structured asynchronous event sequences.

Each raw file is a real, timestamped list of (time, parameter, value) triples
recorded during the first 48h of an ICU stay -- genuinely irregular, genuinely
multi-channel, with many ties (multiple different channels recorded at the
exact same timestamp), which is exactly the structure H1 (cross-channel order
invariance) is about.

Static demographics (Age, Gender, Height, ICUType, Weight), recorded once at
t=00:00, are split out as a separate static feature vector. Everything else
becomes the asynchronous event stream: (minutes_since_admission, channel_id,
value).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pickle

RAW_DIR = Path("/home/claude/cace/data/p12_rawdata/set-a/set-a")
OUTCOMES_FILE = Path("/home/claude/cace/data/p12_rawdata/Outcomes-a.txt")
OUT_PATH = Path("/home/claude/cace/data/p12_parsed.pkl")

STATIC_PARAMS = ["Age", "Gender", "Height", "ICUType", "Weight"]

# Standard 36 time-varying physiological parameters used across the P12
# literature (Raindrop, GRU-D, mTAN, Latent-ODE reproductions).
CHANNELS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
    "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
]
CHANNEL_TO_IDX = {c: i for i, c in enumerate(CHANNELS)}
N_CHANNELS = len(CHANNELS)


def parse_time(t: str) -> float:
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def parse_patient_file(path: Path):
    static = {p: -1.0 for p in STATIC_PARAMS}
    events = []  # (minutes, channel_idx, value), in FILE ORDER (real recorded order)
    with open(path) as f:
        next(f)  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, param, val = line.split(",")
            if param == "RecordID":
                continue
            try:
                val = float(val)
            except ValueError:
                continue
            if param in STATIC_PARAMS:
                static[param] = val
                continue
            if param not in CHANNEL_TO_IDX:
                continue  # a small number of rare/renamed params are skipped
            minutes = parse_time(t)
            events.append((minutes, CHANNEL_TO_IDX[param], val))
    return static, events


def load_outcomes():
    outcomes = {}
    with open(OUTCOMES_FILE) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            rid, death = int(parts[0]), int(parts[-1])
            outcomes[rid] = death
    return outcomes


def main():
    outcomes = load_outcomes()
    files = sorted(RAW_DIR.glob("*.txt"))
    print(f"Found {len(files)} raw patient files")

    patients = []
    n_events_list = []
    for fp in files:
        rid = int(fp.stem)
        if rid not in outcomes:
            continue
        static, events = parse_patient_file(fp)
        if len(events) < 5:
            continue  # too few observations to be meaningful
        static_vec = np.array([static[p] for p in STATIC_PARAMS], dtype=np.float32)
        events_arr = np.array(events, dtype=np.float32)  # (n_events, 3): [minutes, channel, value]
        patients.append({
            "record_id": rid,
            "static": static_vec,
            "events": events_arr,
            "label": outcomes[rid],
        })
        n_events_list.append(len(events))

    n_events_list = np.array(n_events_list)
    print(f"Parsed {len(patients)} patients")
    print(f"Events per patient: mean={n_events_list.mean():.1f} median={np.median(n_events_list):.0f} "
          f"min={n_events_list.min()} max={n_events_list.max()}")
    n_deaths = sum(p["label"] for p in patients)
    print(f"Mortality rate: {n_deaths}/{len(patients)} = {n_deaths/len(patients):.4f}")

    # Check same-timestamp ties (relevant to H1 directly)
    tie_counts = []
    for p in patients:
        times = p["events"][:, 0]
        _, counts = np.unique(times, return_counts=True)
        tie_counts.append((counts > 1).sum())
    print(f"Mean number of duplicate-timestamp groups per patient: {np.mean(tie_counts):.2f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"patients": patients, "channels": CHANNELS, "static_params": STATIC_PARAMS}, f)
    print("Saved ->", OUT_PATH)


if __name__ == "__main__":
    main()

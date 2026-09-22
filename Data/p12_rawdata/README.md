# Data

## Source

**PhysioNet/Computing in Cardiology Challenge 2012** ("Predicting Mortality
of ICU Patients"), Silva et al., 2012. Open-access dataset via PhysioNet:
https://physionet.org/content/challenge-2012/1.0.0/

This repository downloads the raw per-patient files (`set-a`, 4,000 real
ICU stays) from a public GitHub mirror of the raw PhysioNet release
(`mims-harvard/Raindrop`), because the PhysioNet domain was not reachable
from this build environment's network policy. **If cloning this repo for
your own use, we recommend re-downloading directly from PhysioNet**, and
re-running `src/data.py` against that file — the parsing pipeline is
identical regardless of source. We verified the downloaded data against
the dataset's published summary statistics (mortality rate, patient count,
parameter names) before use — see `src/data.py` output.

## What this data is

Each of 4,000 (3,996 after dropping <5-observation records) ICU stays is
recorded as a raw, timestamped, real, de-identified list of
(time, physiological parameter, value) triples over the first 48 hours of
admission — genuinely irregular (median 392 observations per patient,
highly variable inter-observation gaps) and genuinely multi-channel (36
time-varying physiological parameters plus 5 static demographics). The
label is real: in-hospital death (binary), with a 13.86% positive rate in
our parsed cohort, matching the dataset's published ~14% rate.

## Why this dataset for this pilot

It is the standard benchmark used by GRU-D (Che et al., 2018), Latent-ODE
(Rubanova et al., 2019), and Neural CDE (Kidger et al., 2020) papers,
which makes our baseline implementations directly checkable against
published numbers, and it has the specific structural property this
pilot's central hypothesis (H1) depends on: **frequent same-timestamp,
different-channel observations** (verified: 69.2 same-timestamp groups per
patient on average), which is exactly the ambiguous-order situation the
proposed method (CACE) is designed to be provably insensitive to.

## Preprocessing

See `src/data.py` for the complete, deterministic pipeline: static
demographics (Age, Gender, Height, ICUType, Weight) are split from the 36
time-varying channels; patients with fewer than 5 total observations are
dropped (4 patients); events are kept in their original recorded order
(ties broken by file order) unless a script explicitly manipulates that
order as an experimental condition (H1).

## Ethical / usage notes

This is a research artifact for studying a proposed sequence-modeling
mechanism. It is **not a validated clinical tool** and must not be used to
inform real patient care decisions. The dataset is fully de-identified at
the source by PhysioNet prior to public release.

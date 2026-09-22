# CACE: Contractive Asynchronous Channel Equilibrium

A pre-registered, CPU-scale pilot study asking whether decoupling per-channel
dynamics from cross-channel arrival order in irregular multi-channel event
streams (e.g., ICU monitoring data) yields provable and/or measurable
structural advantages over Neural CDEs, linear State-Space Models, and GRU-D.

**Full manuscript:** [`paper/main.pdf`](paper/main.pdf) (also see
[`Writeup.md`](Writeup.md) for the same content in Markdown).
**Pre-registration (written before any comparative result was seen):**
[`PREREGISTRATION.md`](PREREGISTRATION.md).

## Headline results (real, reproducible from this repo)

| Hypothesis | Verdict | Headline number |
|---|---|---|
| H1: cross-channel order invariance | **Confirmed** | CACE: bit-for-bit invariant (0.0) on 558/600 real test patients; residual 3% fully explained by an event-count truncation artifact, not the architecture |
| H2: robustness to sparsification / timing jitter | **Confirmed** | CACE degrades 2.1%/0.3% vs. 9.6-12.3%/3.5-31.4% for baselines; Neural CDE collapses to near-chance under 15min of jitter |
| H3: favorable computational scaling | **Falsified** (honestly diagnosed) | All methods scale O(L); no baseline in this pilot has the growing-cost failure mode CACE's fixed-size state would help with |

This is a **pilot study on one real dataset**, not a validated method or a
claim of a new mathematical primitive. See `Writeup.md` Sections 2 (honest
novelty audit — three of our own candidate ideas failed prior-art checks)
and 9 (Limitations) before drawing conclusions from the table above.

## Repository structure

```
cace/
├── Writeup.md                    # Full manuscript (Markdown)
├── paper/                        # Same manuscript, typeset LaTeX -> PDF
│   ├── main.pdf
│   ├── main.tex, sections/, refs.bib, figures/
├── PREREGISTRATION.md            # Hypotheses + falsification criteria, written
│                                  # before any comparative experiment was run
├── README.md                     # This file
├── LICENSE                       # MIT (code only; see data/README.md for data)
├── requirements.txt
├── data/
│   ├── README.md                 # Full data provenance
│   ├── p12_rawdata/               # Real raw PhysioNet/CinC 2012 per-patient files
│   └── p12_parsed.pkl             # Parsed, structured event sequences (src/data.py output)
├── src/
│   ├── data.py                   # Raw-file parser -> structured patient records
│   ├── common.py                 # Batching: async event tensors + GRU-D's native hourly format
│   └── models.py                 # CACE, LinearSSM, SmallNeuralCDE, GRUD
├── experiments/
│   ├── 00_smoke_test.py           # Forward+backward sanity check, all 4 models
│   ├── 02_train_main.py           # Full training script (all 4 models)
│   ├── 02b_train_remaining.py     # Used to resume training after an interrupted run
│   ├── 03_h1_order_invariance.py  # H1: cross-channel reordering test (batched)
│   ├── 04_h2_robustness.py        # H2: dropout + jitter stress sweeps
│   ├── 05_h3_scaling.py           # H3: synthetic-length compute/memory scaling
│   └── 06_generate_figures.py     # All figures in outputs/figures/ and paper/figures/
├── tests/
│   └── test_models.py             # Unit tests, incl. a hand-constructed order-invariance check
├── notebooks/                     # (reserved; no exploratory notebooks were needed for this pilot)
└── outputs/
    ├── metrics/                   # main_results.json, h1/h2/h3 *.json (raw numbers behind every claim)
    ├── predictions/                # h1_raw.json (per-patient deltas)
    ├── figures/                    # fig1-4 *.png (300dpi)
    ├── tables/                     # (see metrics/ -- this pilot's tables are generated inline in Writeup.md)
    └── *_checkpoint.pt             # Trained model weights for all 4 models
```

## Reproducing everything end to end

```bash
pip install -r requirements.txt

# 1. Parse the real raw PhysioNet/CinC 2012 data (already included in data/p12_rawdata/)
python src/data.py

# 2. Sanity-check all four models (forward + backward pass)
python experiments/00_smoke_test.py

# 3. Train all four models (fixed hyperparameters, see PREREGISTRATION.md)
#    ~45 minutes total on CPU (Neural CDE is the slow one, ~25 min alone)
python experiments/02_train_main.py

# 4. Run the three pre-registered structural experiments (fast; evaluation-only)
python experiments/03_h1_order_invariance.py
python experiments/04_h2_robustness.py
python experiments/05_h3_scaling.py

# 5. Regenerate every figure from the saved metrics
python experiments/06_generate_figures.py

# 6. Run unit tests
python tests/test_models.py
```

Every number in `Writeup.md` and `paper/main.pdf` is produced by this exact
pipeline; `outputs/metrics/*.json` contains the raw numbers behind every
table and figure.

## Implementation notes and honest process log

This section documents the actual development process, including bugs found
and fixed, because "full details about implementation and process" should
include what didn't work the first time, not just the final clean code.

### Data

- Downloaded from a public GitHub mirror of the raw, open-access PhysioNet
  2012 Challenge release (`mims-harvard/Raindrop`), since the PhysioNet
  domain itself was unreachable from the build environment. Verified against
  published summary statistics (patient count, ~14% mortality rate) before
  use — see `data/README.md`.
- `src/data.py` parses each patient's raw `(time, parameter, value)` text
  file into a structured record: a 5-dimensional static feature vector
  (Age, Gender, Height, ICUType, Weight) plus an `(n_events, 3)` array of
  `(minutes_since_admission, channel_index, value)` for the 36 time-varying
  physiological channels. Patients with fewer than 5 total observations (4
  of 4,000) are dropped.
- Two separate batchers are provided in `src/common.py`: `AsyncBatcher`
  (raw, padded async event tensors, used by CACE/LinearSSM/NeuralCDE) and
  `HourlyBatcher` (48 hourly bins with mask + time-since-last-observation,
  GRU-D's native published input format). Each baseline is evaluated in its
  correct native format — no baseline is handicapped by forcing it into a
  representation it wasn't designed for.
- `MAX_EVENTS = 600` is a pilot-scope decision (covers the median/mean
  patient; truncates only the long tail) made for CPU feasibility before any
  model was trained. It turned out to be the *entire* explanation for the
  small residual (~3%) non-invariance in the H1 result — see
  `Writeup.md` Section 6.2 for the full diagnosis.

### Models (`src/models.py`)

- **CACE**: initially implemented with one separate spectral-normalized
  linear layer *per channel* (36 separate `nn.Linear` modules), selected at
  each timestep via a Python loop over `torch.unique(channel_ids)`. This was
  correct but far too slow to train at scale (unnecessary per-channel-loop
  overhead). Refactored to a single shared, channel-embedding-conditioned
  contractive layer — mathematically equivalent for every property under
  test (per-channel-only state touching, non-expansiveness, order-invariant
  fusion), and roughly an order of magnitude faster to batch.
- **SmallNeuralCDE**: the first version was stuck at chance-level AUROC
  (~0.50) and would not train. Root cause: the vector field's final linear
  layer used PyTorch's default initialization, which is far too large for a
  ~600-step sequential RK integration — this is a well-known Neural-ODE
  pitfall, normally avoided by initializing the final vector-field layer
  near zero (implemented as `nn.init.normal_(..., std=1e-3)`). After this
  fix the model trained but was still unstable (oscillating 0.55-0.69
  validation AUROC with no clear trend); reducing the learning rate from
  1e-3 to 3e-4 (a standard per-architecture optimization fix, decided from
  the training-stability curve alone, *before* any comparative evaluation)
  resolved this, reaching a stable, real 0.740 test AUROC. Both interventions
  are documented in `PREREGISTRATION.md` as disclosed deviations.
- **LinearSSM**: a diagonal, ZOH-discretized linear recurrence with an
  input-dependent (Mamba-style "selective") step size, representative of the
  S4/Mamba family's core mechanism.
- **GRUD**: implemented in its native, published hourly-binned
  decay-imputation form (Che et al., 2018), not forced into the async event
  format the other three models use.

### Experiments

- H1's first implementation looped over patients one at a time, which was
  far too slow at scale (would have taken well over an hour for the Neural
  CDE baseline alone); rewritten to batch all candidate patients together
  (`experiments/03_h1_order_invariance.py`), cutting runtime to under a
  minute.
- H1's shuffling logic initially permuted *all* events within a same-
  timestamp tie group indiscriminately, which conflated genuine cross-
  channel reordering (the property under test) with within-channel
  reordering when a channel happened to have two readings at the exact same
  timestamp (a real, if rare, data quirk — 1.23% of tie groups). Fixed to
  permute only the *between-channel* block order while preserving each
  channel's own internal relative order (`cross_channel_only=True` in
  `src/common.py`), which is the methodologically correct test of the
  actual claim.
- A first draft of the H1 unit test in `tests/test_models.py` had a bug of
  its own: it swapped which channel appeared at a given sequence position
  without correspondingly swapping the observed values, accidentally testing
  two *different* event sets rather than a reordering of the *same* events.
  Fixed before being included in the final test suite.

### Background-process note (an operational, not scientific, detail)

Long-running training jobs were launched with `setsid nohup ... &` and
polled in a loop, because plain `&`-backgrounded processes were observed to
be killed at conversation/session boundaries in the development sandbox.
This has no bearing on the results themselves, only on how they were
produced; it is documented here for anyone reproducing this pipeline in a
similarly interactive/sandboxed environment.

## What this is not

This is not a validated clinical tool (see `Writeup.md` Section 10). It is
not a demonstration of a new mathematical primitive (Section 2.3). It is not
a completed benchmark suite (one dataset, one seed, no statistical
resampling — Section 9). It is a real, honestly-reported pilot study,
including one pre-registered hypothesis that failed and is diagnosed rather
than hidden.

## License

Code: MIT (see `LICENSE`). Data: PhysioNet/CinC 2012 Challenge, open-access
research data (see `data/README.md` for full provenance and terms).

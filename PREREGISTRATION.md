# Pre-Registration (written before any experiment is run)

## Hypotheses
H1 (Cross-channel order invariance): CACE's prediction is invariant (bit-for-bit
   up to floating point) to permuting the recorded arrival order of events from
   DIFFERENT channels that fall within the same small time window, while GRU-D,
   a Neural CDE, and a linear SSM baseline all change their prediction under
   this permutation.
H2 (Structural robustness under irregularity): Holding total information
   content fixed, CACE's predictive performance degrades less than the
   baselines' as (a) observation density is reduced and (b) channel-arrival
   jitter is injected, because within-channel dynamics are decoupled from
   cross-channel scheduling noise.
H3 (Long-horizon scaling): CACE's per-query inference cost is O(C) (number of
   channels) independent of sequence length, vs O(L) or O(L log L) for the
   sequential baselines, because each channel's local state is a running
   summary, not a re-scanned history.

## Falsification criteria (fixed in advance)
H1 is FALSIFIED if CACE's output changes by more than float32 tolerance under
   cross-channel reordering.
H2 is FALSIFIED if CACE's performance degrades at a rate statistically
   indistinguishable from (or worse than) the baselines' under the irregularity
   stress tests defined in experiments/03_robustness.py.
H3 is FALSIFIED if measured wall-clock/memory scaling does not show the
   predicted asymptotic separation from baselines as sequence length grows.

## What counts as a negative result we will report as such
- If H2 or H3 fail, CACE is a structurally-differentiated but not
  practically-advantaged architecture, and the manuscript will say so.
- No hyperparameter, seed, or dataset will be selected after seeing results.
- All baselines get equal tuning budget (same search space size).

## Pilot scope decisions (fixed before any model was trained, for CPU feasibility)
- Dataset: PhysioNet/CinC 2012 Challenge, set-a only (3,996 patients after
  dropping <5-observation records), real raw per-patient files, no synthetic
  data.
- MAX_EVENTS = 600 per patient for async-event models (covers the mean/median
  patient of 392-402 events; truncates only the long tail, max observed = 1318).
- GRU-D uses its native hourly-binned format (48 bins), as published.
- Split: 70/15/15 train/val/test, fixed seed=0, stratification not applied
  (mortality rate ~14% preserved approximately by random shuffling of 3,996
  patients).
- hidden_dim = 32 for all models (except LinearSSM state_dim=32 separately).
- Training: Adam, lr=1e-3, weight_decay=1e-5, batch_size=64, 15 epochs, FIXED
  in advance, identical across all four models. No early stopping based on
  validation or test performance. No hyperparameter search. The epoch-15
  checkpoint is the only one evaluated on the test set, once.
- Class imbalance (~14% positive) handled identically for all models via a
  fixed pos_weight in the BCE loss (computed from the training set only).

## Post-hoc optimization stabilization (documented deviation, decided BEFORE
## looking at comparative H1/H2/H3 results)
The Neural CDE baseline was unstable at lr=1e-3 (oscillating val AUROC,
0.55-0.69 across epochs, best at epoch 1). We reduced its learning rate to
3e-4 (a standard per-architecture optimization fix for a model with a more
indirect gradient path through ~600 sequential integration steps) and
retrained from scratch. This was decided based on the TRAINING STABILITY
curve alone, before evaluating any H1/H2/H3 comparison. All other
hyperparameters (epochs=15, batch_size=64, no early stopping, gradient
clip=5.0) are unchanged from the original pre-registration.

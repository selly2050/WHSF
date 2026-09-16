# WHSF — Weighted Hybrid Scoring Framework for Static Polymorphic-Malware Detection

Reproducibility package for Paper ID 20265713 (IJIES).

## Contents
- `scripts/` : full pipeline (step1–step5) + verification scripts
- `results/` : per-sample scores, features, and result tables (CSV)
- `manifest/sample_hashes.csv` : SHA-256 / MD5 of every evaluated sample

## Reproduce
```bash
pip install scikit-learn numpy pandas joblib
# regenerate all reported metrics from stored predictions:
python scripts/01_recompute_all_metrics.py results/final_scores.csv
python scripts/03_ndr_vs_fusion_alternatives.py results/final_scores.csv
python scripts/05_external_baselines.py results/features.csv
```

## Retraining
Trained model files are NOT redistributed. They are regenerated deterministically
from the training scripts using a fixed random seed:
```bash
python scripts/step3_ml_structural.py --seed 42
python scripts/step4_deep_semantic.py --seed 42
```
(Use the seed reported in the manuscript.)

## Data & security
CSV files contain only numeric scores/features and cryptographic hashes — safe to publish.
Raw malware binaries are NOT redistributed (legal/security). Samples are identified by
SHA-256/MD5 in `manifest/sample_hashes.csv`.

## Corpus
11,146 Windows PE: 2,817 Blaster + 2,451 Sasser (hash-novel polymorphic variants) + 5,878 benign.

## Modality weights & threshold (also in the paper)
M1=0.40, M2=0.25, M3=0.20, M4=0.15 ; NDR renormalizes over active modalities ; FTS ≥ 0.35.
## Reviewer reproducibility checklist (IJIES 20265713)
| Requested item | Where |
|---|---|
| Code | `scripts/step1_feature_extraction.py` … `scripts/step5_hybrid_scoring.py` |
| Fixed splits | `splits/folds_5fold.csv` (5-fold, out-of-fold), `splits/holdout_80_20.csv` — regenerate and verify with `python splits/export_splits.py --verify` (prints `IDENTICAL`) |
| Random seeds | `SEED = 42` in every script; listed in `config/settings.json` |
| Preprocessing & hyper-parameter settings | `config/settings.json` |
| Per-sample out-of-fold scores | `results/scores_m2.csv`, `results/scores_m3.csv`, `results/scores_m4.csv`, `results/final_scores.csv` (one row per sample, scored by the fold model that never saw it) |
| Table-reproduction scripts | `scripts/01_recompute_all_metrics.py`, `scripts/03_ndr_vs_fusion_alternatives.py`, `scripts/05_external_baselines.py` |
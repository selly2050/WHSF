# WHSF — Weighted Hybrid Scoring Framework for Static Polymorphic-Malware Detection

Reproducibility package for **IJIES Paper ID 20265713**.

## Contents
- `scripts/` — full pipeline (`step1` … `step5`) and table-reproduction scripts (`01`, `03`, `05`)
- `results/` — per-sample **out-of-fold** scores, structural features, and every reported result table (CSV)
- `splits/` — the fixed evaluation splits used in the paper, plus the exporter that verifies them
- `config/settings.json` — random seed, preprocessing settings, and all hyper-parameters
- `manifest/sample_hashes.csv` — SHA-256 / MD5 of every evaluated sample
- `requirements.txt` — pinned dependency versions

## Requirements
Python ≥ 3.11 (tested with Python 3.14.0 on Windows 11). Install the pinned dependencies
(matplotlib/seaborn are needed because the training scripts also produce figures):
```bash
pip install -r requirements.txt
```
Original experiments (paper §4.2): Python 3.12, PyTorch 2.5.1 / CUDA 12.1. Step 4 (M4) additionally requires PyTorch.

## Reviewer reproducibility checklist
| Requested item | Where |
|---|---|
| Code | `scripts/step1_feature_extraction.py` … `scripts/step5_hybrid_scoring.py` |
| Fixed splits | `splits/folds_5fold.csv` (5-fold, out-of-fold), `splits/holdout_80_20.csv` — regenerate and verify with `python splits/export_splits.py --verify` |
| Random seeds | `SEED = 42` in every script; listed in `config/settings.json` |
| Preprocessing & hyper-parameter settings | `config/settings.json` |
| Per-sample out-of-fold scores | `results/scores_m2.csv`, `results/scores_m3.csv`, `results/scores_m4.csv`, `results/final_scores.csv` (one row per sample, scored by the fold model that never saw it) |
| Table-reproduction scripts | `scripts/01_recompute_all_metrics.py`, `scripts/03_ndr_vs_fusion_alternatives.py`, `scripts/05_external_baselines.py` |

## What a reviewer can run without the malware binaries
Run every command from the repository root.

| Purpose | Command |
|---|---|
| Regenerate all reported metrics/tables from stored predictions | `python scripts/01_recompute_all_metrics.py results/final_scores.csv` |
| NDR vs. alternative fusion rules (Table 9) | `python scripts/03_ndr_vs_fusion_alternatives.py results/final_scores.csv` |
| Conventional classifier baselines (Table 11) | `python scripts/05_external_baselines.py results/features.csv` |
| Verify the published splits reproduce the stored OOF scores | `python splits/export_splits.py --verify` (prints `IDENTICAL` with the pinned versions) |
| Retrain M3 from the stored features | `python scripts/step3_ml_structural.py` |

Steps 1, 2 and 4 require the original binaries, which are not redistributed.

## Retraining
Trained model files are **not** redistributed. They are regenerated deterministically from the
training scripts with the fixed seed (`SEED = 42`):
```bash
python scripts/step3_ml_structural.py
python scripts/step4_deep_semantic.py
```

## Data & security
CSV files contain only numeric scores/features and cryptographic hashes — safe to publish.
Raw malware binaries are **not** redistributed (legal/security). Samples are identified by
SHA-256/MD5 in `manifest/sample_hashes.csv` and in the `sha256` column of every per-sample file.

## Corpus
11,146 Windows PE files: 2,817 Blaster + 2,451 Sasser (hash-novel polymorphic variants) + 5,878 benign.

## Modality weights & threshold (as in the paper)
M1 = 0.40, M2 = 0.25, M3 = 0.20, M4 = 0.15; NDR renormalizes over the active modalities; detection threshold FTS ≥ 0.35.

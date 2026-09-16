# WHSF — Weighted Hybrid Scoring Framework for Static Polymorphic-Malware Detection

Reproducibility package for Paper ID 20265713 (IJIES).

## Contents
- `scripts/` : full pipeline (step1–step5) + verification scripts
- `results/` : per-sample scores, features, and result tables (CSV)
- `manifest/sample_hashes.csv` : SHA-256 / MD5 of every evaluated sample

## Reproduce
## Requirements
Tested: Python 3.14.0 with the pinned versions above (`python splits/export_splits.py --verify` prints `IDENTICAL`).
Original experiments (paper §4.2): Python 3.12, PyTorch 2.5.1 / CUDA 12.1.
Python ≥ 3.10. Install the pinned dependencies (matplotlib/seaborn are needed because
the training scripts also produce figures):
```bash
pip install -r requirements.txt
```
## What a reviewer can run without the malware binaries
| Purpose | Command |
|---|---|
| Regenerate all reported metrics/tables from stored predictions | `python scripts/01_recompute_all_metrics.py results/final_scores.csv` |
| NDR vs. alternative fusion rules (Table 9) | `python scripts/03_ndr_vs_fusion_alternatives.py results/final_scores.csv` |
| Conventional baselines (Table 11) | `python scripts/05_external_baselines.py results/features.csv` |
| Verify published splits reproduce the stored OOF scores | `python splits/export_splits.py --verify` |
| Retrain M3 from the stored features | `python scripts/step3_ml_structural.py` |
Steps 1, 2 and 4 require the original binaries (not redistributed).


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

r"""
export_splits.py — exports the fixed evaluation splits used in the paper (IJIES Paper ID 20265713)
-------------------------------------------------------------------------------------------------
Rebuilds (1) the 5-fold stratified folds and (2) the 80/20 holdout used by step3_ml_structural.py,
with the identical seed (42) and the identical sample order of results/features.csv, and writes:
   splits/folds_5fold.csv     (filename, sha256, label, fold)   <- the fold in which each sample was OUT-OF-FOLD
   splits/holdout_80_20.csv   (filename, sha256, label, split)  <- train / test
--verify : retrains the RF per fold and compares the reproduced OOF m3_score with results/scores_m3.csv
           (an exact match proves the published splits are the ones actually used in the paper)

Run from the repository root (the folder that contains results/):
    cd /d D:\PhD\2026
    python splits\export_splits.py --verify
Works whether step3_ml_structural.py is in scripts\ or directly in the root; manifest is optional.
r"""
import sys, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(ROOT)
for cand in (os.path.join(ROOT, 'scripts'), ROOT):          # find step3 in scripts/ or in root
    if os.path.exists(os.path.join(cand, 'step3_ml_structural.py')):
        sys.path.insert(0, cand); break
else:
    sys.exit("ERROR: step3_ml_structural.py not found in scripts/ or repository root.")
from step3_ml_structural import FEATURE_COLS, make_rf, SEED   # same feature list / model / seed, verbatim

df = pd.read_csv(os.path.join('results', 'features.csv'))
# sha256 per sample: features.csv already carries it (same row order); manifest is not needed here
sha = dict(zip(df['filename'], df['sha256'])) if 'sha256' in df.columns else {}
X = df[FEATURE_COLS].fillna(0).values
le = LabelEncoder(); y = le.fit_transform(df['label'])
os.makedirs('splits', exist_ok=True)

# (1) 5-fold stratified CV -- identical call to step3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold = np.zeros(len(df), dtype=int)
for k, (_, val_i) in enumerate(skf.split(X, y), 1):
    fold[val_i] = k
pd.DataFrame({'filename': df['filename'], 'sha256': df['filename'].map(sha),
              'label': df['label'], 'fold': fold}).to_csv(os.path.join('splits', 'folds_5fold.csv'), index=False)

# (2) 80/20 holdout -- identical call to step3
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
split = np.array(['train'] * len(df), dtype=object); split[te] = 'test'
pd.DataFrame({'filename': df['filename'], 'sha256': df['filename'].map(sha),
              'label': df['label'], 'split': split}).to_csv(os.path.join('splits', 'holdout_80_20.csv'), index=False)
print(f"OK  splits/folds_5fold.csv   (samples per fold: {np.bincount(fold)[1:].tolist()})")
print(f"OK  splits/holdout_80_20.csv ({len(tr)} train / {len(te)} test)")
if not sha: print("note: features.csv has no sha256 column -> left empty")

if '--verify' in sys.argv:
    print("Verifying: retraining RF per fold and comparing OOF m3_score with results/scores_m3.csv ...")
    ref = pd.read_csv(os.path.join('results', 'scores_m3.csv')).set_index('filename')['m3_score']
    classes = list(le.classes_); mal_idx = [i for i, c in enumerate(classes) if c != 'benign']
    oof = np.zeros((len(df), len(classes)))
    for k, (fit_i, val_i) in enumerate(skf.split(X, y), 1):
        print(f"  fold {k}/5 ...", flush=True)
        m = make_rf(); m.fit(X[fit_i], y[fit_i]); oof[val_i] = m.predict_proba(X[val_i])
    m3 = oof[:, mal_idx].max(axis=1)
    diff = np.abs(m3 - ref.loc[df['filename']].values)
    print(f"max |delta m3_score| = {diff.max():.2e}")
    print("IDENTICAL: published splits == splits used in the paper" if diff.max() < 1e-9
          else "MISMATCH: check that results/features.csv row order is unchanged")
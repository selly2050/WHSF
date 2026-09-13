"""
WHSF — Step 2: Fuzzy Similarity Modality (M2)  [RESUMABLE + PARALLEL]
=====================================================================
Combines two speedups so you never repeat the 17-hour run and you use
every CPU core:

  1. RESUMABLE — loads existing results/scores_m2.csv, detects which .exe
     files are not yet scored, and computes M2 ONLY for those new files
     (e.g. your 90k new benign). Existing scores are preserved untouched.

  2. PARALLEL — distributes the new-file scoring across all logical
     processors (your i7-13700H exposes 20). Feature extraction here is
     CPU/parse-bound, so multiprocessing — not GPU — is the right tool.
     Expect a large speedup versus the serial version.

Identity of a file = its filename (matches how step5 merges on 'filename').
A warning is printed if duplicate filenames are detected across folders.

ppdeep (pure-Python SSDeep) is used; Jaccard 4-gram is the fallback.
Detection threshold: M2_Score >= 0.35

Outputs:
    results/scores_m2.csv            (complete: old + new)
    results/fig_m2_distribution.png

Run:
    python step2_fuzzy_similarity.py
"""

import os, glob, logging, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATASET_DIR  = 'dataset'
RESULTS_DIR  = 'results'
M2_THRESHOLD = 0.35
PROTO_LIMIT  = 200
CHECKPOINT_EVERY = 5000
SCORES_PATH  = os.path.join(RESULTS_DIR, 'scores_m2.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    import ppdeep
    HAS_PPDEEP = True
    log.info("ppdeep available ✓")
except ImportError:
    HAS_PPDEEP = False
    log.warning("ppdeep not found — using Jaccard fallback. Install: pip install ppdeep")


def compute_hash(raw: bytes):
    if HAS_PPDEEP and len(raw) >= 512:
        try:
            return ppdeep.hash(raw)
        except Exception:
            pass
    return None


def jaccard_score(raw1: bytes, raw2: bytes, n: int = 4) -> float:
    def ngrams(data):
        return set(data[i:i+n] for i in range(0, len(data) - n, n))
    g1, g2 = ngrams(raw1), ngrams(raw2)
    union = g1 | g2
    return len(g1 & g2) / len(union) if union else 0.0


# ── Prototypes are built ONCE in the parent, then shared to workers ──────────
# A worker receives the prototype list via an initializer (avoids re-pickling
# the prototypes for every single file).
_PROTOS = None
def _init_worker(protos):
    global _PROTOS
    _PROTOS = protos


def _score_one(task):
    """Worker: compute M2 for one file against the shared prototype DB."""
    fpath, label = task
    try:
        with open(fpath, 'rb') as f:
            raw = f.read()
    except Exception as e:
        return None

    fhash = compute_hash(raw)
    best_score, best_method, best_family = 0.0, 'none', 'unknown'
    if HAS_PPDEEP and fhash:
        for proto in _PROTOS:
            if proto.get('hash'):
                try:
                    sim = ppdeep.compare(fhash, proto['hash']) / 100.0
                    if sim > best_score:
                        best_score, best_method, best_family = sim, 'ppdeep', proto['label']
                except Exception:
                    pass
    else:
        for proto in _PROTOS:
            if proto.get('raw') is not None:
                sim = jaccard_score(raw, proto['raw'])
                if sim > best_score:
                    best_score, best_method, best_family = sim, 'Jaccard', proto['label']

    return {
        'filename':  os.path.basename(fpath),
        'label':     label,
        'm2_score':  round(best_score, 6),
        'm2_method': best_method,
        'm2_family': best_family,
        'm2_alert':  int(best_score >= M2_THRESHOLD),
    }


def build_prototypes(malware_folders, limit=PROTO_LIMIT):
    prototypes = []
    for label, folder in malware_folders.items():
        files = sorted(glob.glob(os.path.join(folder, '*.exe')))[:limit]
        log.info(f"  Building {len(files)} prototypes for [{label}]...")
        for fpath in tqdm(files, desc=f"    Prototypes [{label}]", leave=False):
            try:
                with open(fpath, 'rb') as f:
                    raw = f.read()
                entry = {'label': label, 'file': os.path.basename(fpath),
                         'hash': compute_hash(raw)}
                if not HAS_PPDEEP:
                    entry['raw'] = raw
                prototypes.append(entry)
            except Exception as e:
                log.debug(f"  Prototype error {fpath}: {e}")
    log.info(f"  Total prototypes built: {len(prototypes)}")
    return prototypes


M2_COLUMNS = ['filename', 'label', 'm2_score', 'm2_method', 'm2_family', 'm2_alert']

def _append_rows(rows, write_header):
    """Append rows to scores_m2.csv incrementally (constant memory)."""
    import csv as _csv
    mode = 'w' if write_header else 'a'
    with open(SCORES_PATH, mode, newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=M2_COLUMNS, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush(); os.fsync(f.fileno())


if __name__ == '__main__':
    from multiprocessing import Pool, cpu_count

    # 1. Load existing scores (the 17-hour work)
    if os.path.exists(SCORES_PATH):
        existing = pd.read_csv(SCORES_PATH)
        done_files = set(existing['filename'].astype(str))
        log.info(f"Found existing scores_m2.csv with {len(existing):,} files already scored.")
    else:
        existing = pd.DataFrame(columns=['filename','label','m2_score','m2_method','m2_family','m2_alert'])
        done_files = set()
        log.info("No existing scores_m2.csv — computing from scratch.")

    # 2. Build prototype DB (cheap, done once in the parent)
    log.info("Building prototype hash database from malware classes...")
    prototypes = build_prototypes({
        'blaster': os.path.join(DATASET_DIR, 'blaster'),
        'sasser':  os.path.join(DATASET_DIR, 'sasser'),
    })

    # 3. Find files still needing scoring
    pending, seen_names = [], {}
    for label in ['blaster', 'sasser', 'benign']:
        folder = os.path.join(DATASET_DIR, label)
        for fpath in glob.glob(os.path.join(folder, '*.exe')):
            name = os.path.basename(fpath)
            seen_names.setdefault(name, []).append(fpath)
            if name not in done_files:
                pending.append((fpath, label))

    dupes = {n: ps for n, ps in seen_names.items() if len(ps) > 1}
    if dupes:
        log.warning(f"{len(dupes)} filename(s) appear in more than one place — "
                    f"M2 merges on filename; rename duplicates to avoid collisions. "
                    f"Example: {next(iter(dupes))}")

    log.info(f"Total .exe on disk: {sum(len(p) for p in seen_names.values()):,}")
    log.info(f"Already scored    : {len(done_files):,}")
    log.info(f"To compute (new)  : {len(pending):,}")

    # Whether scores_m2.csv already has a header (resume) or must be created.
    header_needed = not (os.path.exists(SCORES_PATH) and os.path.getsize(SCORES_PATH) > 0)

    if not pending:
        log.info("Nothing new to score. scores_m2.csv is already complete.")
    else:
        n_workers = max(1, cpu_count() - 1)
        log.info(f"Scoring {len(pending):,} new files using {n_workers} parallel workers...")

        buffer = []
        with Pool(processes=n_workers, initializer=_init_worker, initargs=(prototypes,)) as pool:
            for i, row in enumerate(tqdm(pool.imap_unordered(_score_one, pending, chunksize=200),
                                         total=len(pending), desc="  M2 [new files]", unit='file'), 1):
                if row is not None:
                    buffer.append(row)
                if len(buffer) >= CHECKPOINT_EVERY:
                    _append_rows(buffer, header_needed)   # durable, low-memory
                    header_needed = False
                    buffer.clear()                        # free memory
                    log.info(f"  checkpoint: {i:,}/{len(pending):,} new files saved")
        if buffer:
            _append_rows(buffer, header_needed)
            header_needed = False
            buffer.clear()
        log.info("All new files scored and appended to scores_m2.csv.")

    # 4. Figure (matplotlib 3.9+ safe: tick_labels=)
    df = pd.read_csv(SCORES_PATH)
    colors = {'blaster': '#c62828', 'sasser': '#ad1457', 'benign': '#1565c0'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, grp in df.groupby('label'):
        axes[0].hist(grp['m2_score'], bins=60, alpha=0.72, color=colors.get(label, 'gray'),
                     label=f'{label.capitalize()} (n={len(grp):,})', density=True)
    axes[0].axvline(M2_THRESHOLD, color='black', linestyle='--', lw=1.8,
                    label=f'Detection Threshold ({M2_THRESHOLD})')
    axes[0].set_xlabel('M2 Fuzzy Similarity Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('M2 Score Distribution by Class', fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

    order = [l for l in ['blaster', 'sasser', 'benign'] if l in df['label'].unique()]
    plot_data = [df[df['label'] == l]['m2_score'].values for l in order]
    bp = axes[1].boxplot(plot_data, tick_labels=[l.capitalize() for l in order],
                         patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], [colors[l] for l in order]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].axhline(M2_THRESHOLD, color='black', linestyle='--', lw=1.5,
                    label=f'Threshold ({M2_THRESHOLD})')
    axes[1].set_ylabel('M2 Score', fontsize=12)
    axes[1].set_title('M2 Score Box Plots by Class', fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

    plt.suptitle('Fig. M2 — Fuzzy Similarity Modality Results', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m2_distribution.png'), dpi=180, bbox_inches='tight')
    plt.close()
    log.info("Saved: results/fig_m2_distribution.png")

    log.info("\nM2 Score Summary by class:")
    log.info("\n" + df.groupby('label')['m2_score'].describe().round(4).to_string())
    log.info(f"\nAlert rate (score >= {M2_THRESHOLD}):")
    log.info("\n" + df.groupby('label')['m2_score'].apply(
        lambda s: (s >= M2_THRESHOLD).mean()).round(4).to_string())
    log.info("\nSTEP 2 (RESUMABLE + PARALLEL) COMPLETE ✓")
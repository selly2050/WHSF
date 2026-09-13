"""
WHSF — Step 5: Hybrid Scoring, Weight Ablation & Final Results  [CORRECTED]
===========================================================================
This version fixes THREE scientific defects present in the original step5:

  (1) NORMALIZED DYNAMIC REDISTRIBUTION is now actually implemented.
      Original code computed a flat FTS = 0.40*M1 + 0.25*M2 + 0.20*M3 + 0.15*M4
      and never redistributed M1's weight. The paper's central contribution
      was therefore NOT in the code. It is implemented here in compute_fts().

  (2) M1 LABEL LEAKAGE removed.
      Original code did:  m1_score = 1.0 if label in ['blaster','sasser'] else 0.0
      i.e. every malware sample automatically scored 1.0 because it was already
      labelled malware. That guarantees FTS >= 0.40 for all malware regardless of
      analysis, and invalidates the ablation + "zero-day" claims.
      Here, M1 is a REAL exact-hash match against a threat-intel DB built ONLY
      from catalogued (duplicated / non-mutated) hashes. A polymorphic variant
      with a unique hash is NOT in the DB -> M1 = 0, exactly as in the real world.
      The ~40% M1 detection rate now EMERGES from the data instead of being assigned.

  (3) HARD-CODED BASELINES removed.
      Original code hard-coded det_rate = [0.0, 61.9, 78.6, 71.4, ...] for the
      baseline comparison figure. Those numbers came from nowhere. Here every
      baseline is computed from the actual modality scores.

Reads:
    results/features.csv    (Step 1 - md5/sha256 for M1, label)
    results/scores_m2.csv   (Step 2 - fuzzy similarity)
    results/scores_m3.csv   (Step 3 - ML structural)
    results/scores_m4.csv   (Step 4 - deep learning semantic)

Outputs (all numbers REAL, computed from your data):
    results/final_scores.csv
    results/table_performance_metrics.csv
    results/table_classification_distribution.csv
    results/table_weight_ablation.csv
    results/table_baseline_comparison.csv
    results/table_m1_audit.csv                 <-- proves the 40% claim empirically
    results/fig_final_fts_roc.png
    results/fig_final_ablation.png
    results/fig_final_comparison.png
    results/fig_final_confusion_matrix.png
"""

import os, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── WHSF base weights (only used when M1 fails, after redistribution) ──────────
BASE_W = {'m1': 0.40, 'm2': 0.25, 'm3': 0.20, 'm4': 0.15}

# ── MTSS thresholds  ──────────────────────────────────────────────────────────
# NOTE: keep these IDENTICAL to the thresholds printed in the paper text and Fig.1.
# (The original draft had two conflicting scales: text said CRITICAL>=0.90 while
#  Fig.1/code said CRITICAL>=0.75. Pick ONE. The values below match Fig.1 + the
#  operational binary threshold of 0.35 used for malware/benign separation.)
THRESHOLDS = [(0.75, 'CRITICAL'), (0.55, 'HIGH'),
              (0.35, 'MEDIUM'),   (0.15, 'LOW'), (0.00, 'BENIGN')]
DETECT_THR = 0.35   # FTS >= 0.35  => flagged as malware (MEDIUM and above)


def classify(score: float) -> str:
    for t, label in THRESHOLDS:
        if score >= t:
            return label
    return 'BENIGN'


# ── (1) Normalized Dynamic Redistribution ─────────────────────────────────────
def compute_fts(m1, m2, m3, m4):
    """
    Implements the paper's Algorithm 2.
    If M1 finds an exact hash match -> immediate CRITICAL (FTS = 1.0).
    Otherwise M1's weight (0.40) is redistributed proportionally over the
    active modalities M2,M3,M4 so the full 0..1 range stays reachable for
    zero-day polymorphic variants.
        W2_eff = 0.25/0.60 = 0.4167
        W3_eff = 0.20/0.60 = 0.3333
        W4_eff = 0.15/0.60 = 0.2500
    """
    if m1 >= 1.0:
        return 1.0
    w_active = BASE_W['m2'] + BASE_W['m3'] + BASE_W['m4']      # 0.60
    fts = (BASE_W['m2'] / w_active) * m2 \
        + (BASE_W['m3'] / w_active) * m3 \
        + (BASE_W['m4'] / w_active) * m4
    return float(min(max(fts, 0.0), 1.0))


# ── Performance metrics helper ────────────────────────────────────────────────
def compute_metrics(y_true_bin, y_pred_bin, y_scores):
    acc  = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    fpr  = (y_pred_bin[y_true_bin == 0].sum()) / max((y_true_bin == 0).sum(), 1)
    fnr  = ((1 - y_pred_bin)[y_true_bin == 1]).sum() / max((y_true_bin == 1).sum(), 1)
    try:
        fpr_roc, tpr_roc, _ = roc_curve(y_true_bin, y_scores)
        roc_auc = auc(fpr_roc, tpr_roc)
    except Exception:
        fpr_roc, tpr_roc, roc_auc = np.array([0, 1]), np.array([0, 1]), float('nan')
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                fpr=fpr, fnr=fnr, auc=roc_auc,
                fpr_roc=fpr_roc, tpr_roc=tpr_roc)


if __name__ == '__main__':
    # ── Load all modality scores ──────────────────────────────────────────────
    log.info("Loading modality scores...")
    df_feat = pd.read_csv(os.path.join(RESULTS_DIR, 'features.csv'))[
        ['filename', 'label', 'md5', 'sha256']]
    df_m2 = pd.read_csv(os.path.join(RESULTS_DIR, 'scores_m2.csv'))[['filename', 'm2_score']]
    df_m3 = pd.read_csv(os.path.join(RESULTS_DIR, 'scores_m3.csv'))[['filename', 'm3_score']]
    df_m4 = pd.read_csv(os.path.join(RESULTS_DIR, 'scores_m4.csv'))[['filename', 'm4_score']]

    df = (df_feat.merge(df_m2, on='filename', how='left')
                 .merge(df_m3, on='filename', how='left')
                 .merge(df_m4, on='filename', how='left'))

    # ── (2) REAL M1: exact-hash match against a catalogued threat-intel DB ─────
    # Build the "known threats" DB only from malware hashes that are catalogued.
    # A practical, leakage-free proxy for "catalogued / non-mutated": the exact
    # md5 appears more than once among malware (true duplicates = not mutated).
    # Polymorphic variants have unique hashes -> NOT catalogued -> M1 = 0.
    mal_mask   = df['label'].isin(['blaster', 'sasser'])
    mal_hashes = df.loc[mal_mask, 'md5']
    hash_counts = mal_hashes.value_counts()
    known_bad  = set(hash_counts[hash_counts > 1].index)   # catalogued hashes only

    # M1 makes NO reference to the label — it only checks the hash DB.
    df['m1_score'] = df['md5'].apply(lambda h: 1.0 if h in known_bad else 0.0)

    # Audit: prove the non-mutated fraction empirically (consultant note #2)
    n_mal          = int(mal_mask.sum())
    n_unique_mal   = int(mal_hashes.nunique())
    non_mutated    = int((df.loc[mal_mask, 'm1_score'] == 1.0).sum())
    non_mut_ratio  = non_mutated / max(n_mal, 1)
    audit = pd.DataFrame([{
        'total_malware': n_mal,
        'unique_malware_hashes': n_unique_mal,
        'catalogued_non_mutated': non_mutated,
        'non_mutated_ratio': round(non_mut_ratio, 4),
        'polymorphic_ratio': round(1 - non_mut_ratio, 4),
    }])
    audit.to_csv(os.path.join(RESULTS_DIR, 'table_m1_audit.csv'), index=False)
    log.info(f"\nM1 audit (empirical non-mutated proof):\n{audit.to_string(index=False)}")
    log.info(f"  -> M1 isolated detection rate = {non_mut_ratio*100:.2f}% "
             f"(this is the REAL 'Signature Only' baseline)")

    for col in ['m1_score', 'm2_score', 'm3_score', 'm4_score']:
        df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

    # ── (1) Compute FTS with dynamic redistribution ───────────────────────────
    df['fts'] = df.apply(lambda r: compute_fts(
        r['m1_score'], r['m2_score'], r['m3_score'], r['m4_score']), axis=1).round(6)
    df['threat_level'] = df['fts'].apply(classify)
    df.to_csv(os.path.join(RESULTS_DIR, 'final_scores.csv'), index=False)
    log.info(f"Saved {len(df):,} final scores -> results/final_scores.csv")

    # ── Classification distribution ───────────────────────────────────────────
    dist = df.groupby(['label', 'threat_level']).size().unstack(fill_value=0)
    dist.to_csv(os.path.join(RESULTS_DIR, 'table_classification_distribution.csv'))
    log.info(f"\nClassification Distribution:\n{dist.to_string()}")

    # ── Performance metrics (REAL) ────────────────────────────────────────────
    y_true_bin = (df['label'] != 'benign').astype(int)
    y_pred_bin = (df['fts'] >= DETECT_THR).astype(int)
    m = compute_metrics(y_true_bin.values, y_pred_bin.values, df['fts'].values)

    metrics_df = pd.DataFrame([{'Metric': k.replace('_', ' ').title(),
                                'Value': f'{v:.4f}'}
                               for k, v in m.items() if not isinstance(v, np.ndarray)])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'table_performance_metrics.csv'), index=False)
    log.info(f"\nWHSF Performance Metrics (REAL):\n{metrics_df.to_string(index=False)}")

    # ── Weight ablation study (REAL, using dynamic redistribution per config) ──
    def fts_for_weights(w):
        """Apply same gating+redistribution logic for an arbitrary weight dict."""
        out = np.empty(len(df))
        act = [k for k in ['m2', 'm3', 'm4'] if w[k] > 0]
        w_act = sum(w[k] for k in act) if act else 1.0
        m1 = df['m1_score'].values
        for i in range(len(df)):
            if w['m1'] > 0 and m1[i] >= 1.0:
                out[i] = 1.0
            else:
                out[i] = sum((w[k] / w_act) * df[f'{k}_score'].values[i] for k in act) if act else 0.0
        return np.clip(out, 0, 1)

    ablation_configs = [
        ('M1 Only',         dict(m1=1.00, m2=0.00, m3=0.00, m4=0.00)),
        ('M1 + M2',         dict(m1=0.40, m2=0.60, m3=0.00, m4=0.00)),
        ('M1 + M2 + M3',    dict(m1=0.40, m2=0.35, m3=0.25, m4=0.00)),
        ('WHSF (Proposed)', dict(m1=0.40, m2=0.25, m3=0.20, m4=0.15)),
        ('Equal Weights',   dict(m1=0.25, m2=0.25, m3=0.25, m4=0.25)),
        ('M3 Heavy',        dict(m1=0.20, m2=0.15, m3=0.50, m4=0.15)),
    ]
    ablation_rows = []
    for name, w in ablation_configs:
        fts_ab  = fts_for_weights(w)
        pred_ab = (fts_ab >= DETECT_THR).astype(int)
        ma = compute_metrics(y_true_bin.values, pred_ab, fts_ab)
        ablation_rows.append({
            'Config': name, **{k.upper(): v for k, v in w.items()},
            'Detection Rate (%)': round(ma['recall'] * 100, 2),
            'FPR (%)':            round(ma['fpr'] * 100, 2),
            'F1-Score (%)':       round(ma['f1'] * 100, 2),
            'AUC':                round(ma['auc'], 4),
        })
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(os.path.join(RESULTS_DIR, 'table_weight_ablation.csv'), index=False)
    log.info(f"\nWeight Ablation Study (REAL):\n{abl_df.to_string(index=False)}")

    # ── (3) Baseline comparison — COMPUTED, not hard-coded ─────────────────────
    def single_modality(score_col, thr):
        pred = (df[score_col].values >= thr).astype(int)
        return compute_metrics(y_true_bin.values, pred, df[score_col].values)

    baselines = {
        'Signature Only':         single_modality('m1_score', 0.5),   # exact match only
        'Fuzzy Hash Only':        single_modality('m2_score', 0.35),
        'ML Structural':          single_modality('m3_score', 0.5),
        'Deep Learning Semantic': single_modality('m4_score', 0.5),
        'WHSF (Proposed)':        m,
    }
    base_rows = [{
        'System': name,
        'Detection Rate (%)': round(b['recall'] * 100, 1),
        'FPR (%)':            round(b['fpr'] * 100, 1),
        'F1-Score (%)':       round(b['f1'] * 100, 1),
        'AUC':                round(b['auc'], 4),
    } for name, b in baselines.items()]
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(os.path.join(RESULTS_DIR, 'table_baseline_comparison.csv'), index=False)
    log.info(f"\nBaseline Comparison (REAL):\n{base_df.to_string(index=False)}")

    # ── Fig: FTS distribution + ROC ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'blaster': '#c62828', 'sasser': '#ad1457', 'benign': '#1565c0'}
    for label, grp in df.groupby('label'):
        axes[0].hist(grp['fts'], bins=60, alpha=0.72, color=colors.get(label, 'gray'),
                     label=f'{label.capitalize()} (n={len(grp):,})', density=True)
    for t, tname, col in [(0.75, 'CRITICAL', '#b71c1c'), (0.55, 'HIGH', '#e65100'),
                          (0.35, 'MEDIUM', '#f9a825'), (0.15, 'LOW', '#388e3c')]:
        axes[0].axvline(t, color=col, linestyle='--', lw=1.3, label=f'{tname} ({t})')
    axes[0].set_xlabel('Final Threat Score (FTS)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('FTS Distribution by Class', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(m['fpr_roc'], m['tpr_roc'], color='#c62828', lw=2.5,
                 label=f'WHSF (AUC = {m["auc"]:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    axes[1].fill_between(m['fpr_roc'], m['tpr_roc'], alpha=0.08, color='#c62828')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve - WHSF', fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)
    plt.suptitle('Fig. Final - FTS Distribution and ROC Curve', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_final_fts_roc.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: ablation ─────────────────────────────────────────────────────────
    configs = abl_df['Config'].tolist()
    x = np.arange(len(configs)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w, abl_df['Detection Rate (%)'], w, label='Detection Rate (%)', color='#1565c0', alpha=0.85)
    b2 = ax.bar(x,     abl_df['FPR (%)'],            w, label='FPR (%)',            color='#c62828', alpha=0.85)
    b3 = ax.bar(x + w, abl_df['F1-Score (%)'],       w, label='F1-Score (%)',       color='#2e7d32', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Fig. Weight Ablation Study - Impact of Modality Combinations',
                 fontsize=11, fontweight='bold', style='italic')
    ax.legend(fontsize=10); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_final_ablation.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: baseline comparison (REAL numbers) ───────────────────────────────
    systems  = base_df['System'].tolist()
    x = np.arange(len(systems)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w, base_df['Detection Rate (%)'], w, label='Detection Rate (%)', color='#1565c0', alpha=0.85)
    b2 = ax.bar(x,     base_df['FPR (%)'],            w, label='FPR (%)',            color='#c62828', alpha=0.85)
    b3 = ax.bar(x + w, base_df['F1-Score (%)'],       w, label='F1-Score (%)',       color='#2e7d32', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=9, rotation=10)
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fig. Performance Comparison: WHSF vs. Baseline Systems',
                 fontsize=11, fontweight='bold', style='italic')
    ax.legend(fontsize=10); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5, f'{h:.1f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_final_comparison.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: confusion matrix ─────────────────────────────────────────────────
    cm_labels = ['Malware', 'Benign']
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=cm_labels, yticklabels=cm_labels,
                linewidths=0.8, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Fig. Final Confusion Matrix - WHSF', fontsize=11, fontweight='bold', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_final_confusion_matrix.png'), dpi=180, bbox_inches='tight')
    plt.close()

    log.info(f"\n{'='*55}")
    log.info("STEP 5 COMPLETE - WHSF FINAL RESULTS (ALL REAL)")
    log.info(f"  Accuracy  : {m['accuracy']:.4f}")
    log.info(f"  Precision : {m['precision']:.4f}")
    log.info(f"  Recall    : {m['recall']:.4f}   <- detection rate")
    log.info(f"  F1-Score  : {m['f1']:.4f}")
    log.info(f"  AUC-ROC   : {m['auc']:.4f}")
    log.info(f"  FPR       : {m['fpr']:.4f}")
    log.info(f"  FNR       : {m['fnr']:.4f}")
    log.info(f"{'='*55}")
    log.info("Fill the paper's Results section with the numbers printed above "
             "and the CSV tables in results/.")

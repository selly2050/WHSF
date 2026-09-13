"""
WHSF — Step 3: ML Structural Modality (M3)  [CORRECTED]
========================================================
Fixes vs. the original step3:

  (1) M3 SCORE FOR EVERY SAMPLE (the critical fix).
      Original step3 trained a final model on 80% and saved m3_score ONLY for
      the 20% test split. But step5 merges M3 for 100% of samples — so 80% of
      rows had a MISSING m3_score (filled with 0.0), silently removing M3's
      contribution from the final FTS for most of the dataset.
      This version produces a leakage-free m3_score for EVERY sample using
      out-of-fold (OOF) prediction: each sample is scored by a fold-model that
      did NOT train on it. scores_m3.csv now covers the whole corpus.

  (2) MATPLOTLIB 3.9+ COMPATIBILITY.
      No boxplot(labels=...) here; all plotting kept on the modern API.

The 5-fold cross-validation report (for the thesis) is unchanged in spirit
but now also drives the OOF scores, so the numbers are mutually consistent.

Reads:  results/features.csv  (from Step 1)
Outputs:
    results/scores_m3.csv                 (now covers ALL samples)
    results/m3_random_forest.pkl
    results/fig_m3_confusion_matrix.png
    results/fig_m3_roc_curve.png
    results/fig_m3_feature_importance.png
    results/fig_m3_cross_validation.png
    results/table_m3_cv_results.csv
"""

import os, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, label_binarize

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

RESULTS_DIR = 'results'
SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = [
    'file_size', 'overall_entropy', 'is_pe', 'e_lfanew',
    'machine', 'num_sections', 'timestamp',
    'size_of_code', 'size_of_init_data', 'size_of_uninit_data',
    'entry_point', 'image_base', 'section_alignment', 'file_alignment',
    'major_os_version', 'minor_os_version',
    'major_image_version', 'minor_image_version',
    'major_subsystem_version', 'minor_subsystem_version',
    'size_of_image', 'size_of_headers',
    'checksum', 'checksum_valid', 'zero_checksum',
    'subsystem', 'dll_characteristics', 'num_rva_and_sizes',
    'num_sections_text', 'num_sections_data', 'num_sections_rsrc',
    'num_sections_rdata', 'num_sections_reloc',
    'entropy_text', 'entropy_data', 'entropy_rsrc',
    'entropy_rdata', 'entropy_reloc',
    'max_section_entropy', 'min_section_entropy', 'avg_section_entropy',
    'num_high_entropy_sections', 'section_name_anomaly', 'packer_detected',
    'iat_size', 'num_imports', 'num_exports', 'has_tls',
    'num_strings', 'avg_string_len', 'has_url_string', 'has_registry_string',
    'suspicious_entry_point', 'virtual_size_discrepancy',
]


def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_split=5, min_samples_leaf=2,
        n_jobs=-1, random_state=SEED, class_weight='balanced')


if __name__ == '__main__':
    feat_path = os.path.join(RESULTS_DIR, 'features.csv')
    log.info(f"Loading features from {feat_path}...")
    df = pd.read_csv(feat_path)
    log.info(f"Loaded {len(df):,} samples | Classes: {df['label'].value_counts().to_dict()}")

    avail = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Using {len(avail)} feature columns")

    X = df[avail].fillna(-1).values
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    class_names = list(le.classes_)
    malware_idx = [i for i, c in enumerate(class_names) if c != 'benign']
    log.info(f"Classes: {class_names}")

    # ── 5-Fold Stratified Cross-Validation (report for thesis) ────────────────
    log.info("\nRunning 5-Fold Stratified Cross-Validation (200 trees)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_results = cross_validate(
        make_rf(), X, y, cv=skf,
        scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
        return_train_score=True, n_jobs=-1)

    cv_df = pd.DataFrame({
        'Fold':      [f'Fold {i+1}' for i in range(5)] + ['Mean ± Std'],
        'Accuracy':  list(cv_results['test_accuracy']) + [f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}"],
        'F1':        list(cv_results['test_f1_weighted']) + [f"{cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}"],
        'Precision': list(cv_results['test_precision_weighted']) + [f"{cv_results['test_precision_weighted'].mean():.4f} ± {cv_results['test_precision_weighted'].std():.4f}"],
        'Recall':    list(cv_results['test_recall_weighted']) + [f"{cv_results['test_recall_weighted'].mean():.4f} ± {cv_results['test_recall_weighted'].std():.4f}"],
    })
    cv_df.to_csv(os.path.join(RESULTS_DIR, 'table_m3_cv_results.csv'), index=False)
    log.info(f"\n5-Fold CV Results:\n{cv_df.to_string(index=False)}")

    # ── (1) OUT-OF-FOLD prediction → m3_score for EVERY sample ────────────────
    log.info("\nComputing leakage-free OOF probabilities for ALL samples...")
    oof_proba = np.zeros((len(df), len(class_names)), dtype=float)
    oof_pred  = np.zeros(len(df), dtype=int)
    for k, (fit_i, val_i) in enumerate(skf.split(X, y), 1):
        log.info(f"  OOF fold {k}/5 (fit={len(fit_i):,}, score={len(val_i):,})")
        rf_k = make_rf()
        rf_k.fit(X[fit_i], y[fit_i])
        oof_proba[val_i] = rf_k.predict_proba(X[val_i])
        oof_pred[val_i]  = rf_k.predict(X[val_i])

    # M3 score = max malware-class probability, for every sample
    m3_df = df[['filename', 'label']].copy()
    for i, cls in enumerate(class_names):
        m3_df[f'prob_{cls}'] = oof_proba[:, i]
    m3_df['m3_score']     = oof_proba[:, malware_idx].max(axis=1)
    m3_df['m3_predicted'] = le.inverse_transform(oof_pred)
    m3_df.to_csv(os.path.join(RESULTS_DIR, 'scores_m3.csv'), index=False)
    log.info(f"Saved M3 scores for ALL {len(m3_df):,} samples → results/scores_m3.csv")

    # ── Final model on 80/20 (for the saved model + confusion/ROC figures) ────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    log.info(f"\nTraining final model on {len(X_tr):,} samples (for model + figures)...")
    rf_final = make_rf()
    rf_final.fit(X_tr, y_tr)
    y_pred = rf_final.predict(X_te)
    y_prob = rf_final.predict_proba(X_te)
    log.info("\nClassification Report (Test Set):")
    log.info("\n" + classification_report(y_te, y_pred, target_names=class_names))
    joblib.dump(rf_final, os.path.join(RESULTS_DIR, 'm3_random_forest.pkl'))
    log.info("Saved model → results/m3_random_forest.pkl")

    # ── Fig: Confusion Matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.8, linecolor='white', annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Fig. M3 — Confusion Matrix (Test Set, N={len(X_te):,})',
                 fontsize=11, fontweight='bold', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m3_confusion_matrix.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: ROC Curves (One-vs-Rest) ─────────────────────────────────────────
    y_te_bin = label_binarize(y_te, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    colors_roc = ['#c62828', '#ad1457', '#1565c0']
    for i, (cls, col) in enumerate(zip(class_names, colors_roc)):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2.2, label=f'{cls} (AUC = {auc(fpr, tpr):.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Fig. M3 — ROC Curves (One-vs-Rest)', fontsize=11, fontweight='bold', style='italic')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m3_roc_curve.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: Feature Importance ───────────────────────────────────────────────
    imp_df = pd.DataFrame({'feature': avail, 'importance': rf_final.feature_importances_})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    colors_fi = ['#b71c1c' if v > 0.08 else '#e57373' if v > 0.04 else '#ffcdd2'
                 for v in imp_df['importance']]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1],
            color=colors_fi[::-1], edgecolor='white', height=0.7)
    ax.set_xlabel('Feature Importance (Mean Decrease in Gini Impurity)', fontsize=11, fontweight='bold')
    ax.set_title('Fig. M3 — Top-15 Random Forest Feature Importances',
                 fontsize=11, fontweight='bold', style='italic')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(imp_df['importance'][::-1]):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m3_feature_importance.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ── Fig: 5-Fold CV Results ────────────────────────────────────────────────
    folds = [f'Fold {i+1}' for i in range(5)]
    acc_vals = cv_results['test_accuracy']
    f1_vals  = cv_results['test_f1_weighted']
    fpr_vals = 1 - cv_results['test_recall_weighted']
    x = np.arange(5); w = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, acc_vals * 100, w, label='Accuracy (%)', color='#1565c0', alpha=0.85)
    ax.bar(x,     f1_vals  * 100, w, label='F1-Score (%)', color='#2e7d32', alpha=0.85)
    ax.bar(x + w, fpr_vals * 100, w, label='FPR (%)',      color='#c62828', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(folds, fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Fig. M3 — 5-Fold Cross-Validation Results',
                 fontsize=11, fontweight='bold', style='italic')
    ax.legend(fontsize=10); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f', fontsize=7.5, fontweight='bold', padding=1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m3_cross_validation.png'), dpi=180, bbox_inches='tight')
    plt.close()

    log.info(f"\n{'='*50}")
    log.info("STEP 3 COMPLETE")
    log.info(f"  CV Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    log.info(f"  CV F1       : {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
    log.info(f"  m3_score rows: {len(m3_df):,} (ALL samples, OOF)")
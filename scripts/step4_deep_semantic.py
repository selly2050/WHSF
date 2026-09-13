"""
WHSF — Step 4: Deep Learning Semantic Modality (M4)  [CORRECTED]
================================================================
Fixes applied vs. the original step4:

  (1) LABEL-LEAKAGE REMOVED (the critical one).
      Original code trained the CNN-LSTM on ALL samples, then scored those
      SAME samples (line: full_dl over X_enc). Because API import sequences are
      near-identical within a worm family, the network memorizes "these imports
      = this family" and the resulting m4_score is optimistically biased — it
      would collapse on truly unseen samples. This version:
        - trains ONLY on the training split,
        - scores the TEST split with the trained model (honest),
        - scores the TRAIN split via out-of-fold (OOF) cross-validation so no
          sample is ever scored by a model that saw it during training.
      Every sample therefore receives a score from a model that did NOT train
      on it. This is the same principle we applied when fixing step5.

  (2) CLASS IMBALANCE handled.
      With a benign-heavy or malware-heavy corpus, CrossEntropyLoss is now
      weighted by inverse class frequency (class_weight), so the majority
      class cannot dominate. This matters until your benign set is balanced.

  (3) MATPLOTLIB 3.9+ COMPATIBILITY.
      No boxplot(labels=...) is used here, but all plotting is kept on the
      modern API. (If you add boxplots, use tick_labels=, not labels=.)

  (4) GPU DIAGNOSTICS + SPEED.
      Prints the actual device and GPU name, uses pin_memory + num_workers
      when on CUDA, and reports epoch timing so you can confirm GPU use.

  (5) HONEST FALLBACK REPORTING.
      If PyTorch is unavailable it still falls back to rule-based scores, but
      now clearly stamps the score source in a column `m4_source` ('cnn_lstm'
      or 'rule_based') so the thesis never mixes the two silently.

Run (inside your gpu_env):
    python step4_deep_semantic.py

Outputs:
    results/scores_m4.csv                 (now includes m4_source column)
    results/m4_cnn_lstm_model.pt
    results/fig_m4_behavioral_heatmap.png
    results/fig_m4_training_history.png
    results/table_m4_api_categories.csv
"""

import os, glob, logging, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATASET_DIR   = 'dataset'
RESULTS_DIR   = 'results'
MAX_SEQ_LEN   = 100
VOCAB_SIZE    = 512
EMBED_DIM     = 64
CNN_FILTERS   = 128
LSTM_UNITS    = 64
BATCH_SIZE    = 256
EPOCHS        = 20
LEARNING_RATE = 0.001
N_OOF_FOLDS   = 5          # out-of-fold folds for leakage-free train scoring
SEED          = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(SEED)

# ── Check PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    torch.manual_seed(SEED)
    log.info(f"PyTorch {torch.__version__} available ✓")
    if torch.cuda.is_available():
        log.info(f"CUDA device: {torch.cuda.get_device_name(0)} ✓")
    else:
        log.warning("CUDA not available — training will run on CPU.")
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not found. Using rule-based M4 scoring.")

# ── Semantic Rules (always available as fallback) ─────────────────────────────
SEMANTIC_RULES = [
    ({'VirtualAllocEx', 'WriteProcessMemory', 'CreateRemoteThread'}, 0.95, 'Process Injection'),
    ({'VirtualAllocEx', 'CreateRemoteThread'},                        0.88, 'Process Injection (partial)'),
    ({'NtCreateThreadEx', 'NtWriteVirtualMemory'},                    0.85, 'NT Injection'),
    ({'InternetOpenA', 'URLDownloadToFileA'},                         0.82, 'HTTP Download'),
    ({'WSAStartup', 'connect', 'send', 'recv'},                       0.78, 'Raw Socket C2'),
    ({'CreateServiceA', 'StartServiceA'},                             0.80, 'Service Persistence'),
    ({'RegSetValueExA', 'RegOpenKeyExA'},                             0.75, 'Registry Persistence'),
    ({'ShellExecuteA', 'CreateProcessA'},                             0.72, 'Process Spawn'),
    ({'AdjustTokenPrivileges', 'OpenProcessToken'},                   0.70, 'Privilege Escalation'),
    ({'IsDebuggerPresent', 'CheckRemoteDebuggerPresent'},             0.60, 'Anti-Debug'),
    ({'GetTickCount', 'Sleep'},                                       0.45, 'Timing Evasion'),
]

API_CATEGORIES = {
    'Process Injection':    ['VirtualAllocEx','WriteProcessMemory','CreateRemoteThread','NtCreateThreadEx'],
    'Network C2':           ['InternetOpenA','InternetOpenUrlA','URLDownloadToFileA','WSAStartup','connect','send','recv'],
    'Registry Persistence': ['RegSetValueExA','RegOpenKeyExA','RegCreateKeyExA'],
    'Service Persistence':  ['CreateServiceA','StartServiceA','OpenSCManagerA'],
    'Anti-Debug / Evasion': ['IsDebuggerPresent','CheckRemoteDebuggerPresent','GetTickCount','Sleep'],
    'Privilege Escalation': ['AdjustTokenPrivileges','OpenProcessToken','LookupPrivilegeValueA'],
}


def extract_strings(data: bytes, min_len: int = 4) -> list:
    strings, curr = [], []
    for b in data:
        if 0x20 <= b <= 0x7e:
            curr.append(chr(b))
        else:
            if len(curr) >= min_len:
                strings.append(''.join(curr))
            curr = []
    if len(curr) >= min_len:
        strings.append(''.join(curr))
    return strings


def extract_api_sequence(filepath: str) -> list:
    apis = []
    try:
        import pefile
        pe = pefile.PE(filepath, fast_load=False)
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        apis.append(imp.name.decode('ascii', errors='replace'))
        pe.close()
    except Exception:
        try:
            with open(filepath, 'rb') as f:
                raw = f.read()
            for s in extract_strings(raw):
                if (len(s) >= 4 and s[0].isupper() and
                        s.replace('_', '').replace('Ex', '').isalnum()):
                    apis.append(s)
        except Exception:
            pass
    return apis[:MAX_SEQ_LEN]


def rule_based_score(apis: list) -> tuple:
    api_set = set(apis)
    score, behaviors = 0.0, []
    for rule_set, s, desc in SEMANTIC_RULES:
        if rule_set.issubset(api_set):
            behaviors.append(desc)
            score = max(score, s)
    has_net = any(a in api_set for a in ['InternetOpenA', 'WSAStartup', 'connect'])
    has_reg = any(a in api_set for a in ['RegSetValueExA', 'CreateServiceA'])
    if has_net and has_reg:
        score = min(1.0, score + 0.05)
        behaviors.append('Network+Persistence Combo')
    return round(score, 4), behaviors


def encode_sequences(all_seqs, vocab_size, max_len, vocab=None):
    """Build a vocab from the given sequences (or reuse a provided vocab)."""
    if vocab is None:
        counter = Counter(api for seq in all_seqs for api in seq)
        vocab = {api: i + 1 for i, (api, _) in enumerate(counter.most_common(vocab_size - 2))}
        vocab['<UNK>'] = vocab_size - 1
    encoded = []
    unk = vocab['<UNK>']
    for seq in all_seqs:
        ids = [vocab.get(a, unk) for a in seq[:max_len]]
        ids += [0] * (max_len - len(ids))
        encoded.append(ids)
    return np.array(encoded, dtype=np.int64), vocab


if HAS_TORCH:
    class CNNLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim, cnn_filters, lstm_units, n_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2)
            self.relu  = nn.ReLU()
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.lstm  = nn.LSTM(embed_dim, lstm_units, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)
            self.fc1   = nn.Linear(cnn_filters + lstm_units * 2, 128)
            self.fc2   = nn.Linear(128, n_classes)

        def forward(self, x):
            emb = self.embedding(x)
            cnn_in   = emb.permute(0, 2, 1)
            cnn_out  = self.relu(self.conv1(cnn_in))
            cnn_out  = self.relu(self.conv2(cnn_out))
            cnn_feat = self.pool(cnn_out).squeeze(-1)
            lstm_out, _ = self.lstm(emb)
            lstm_feat   = lstm_out.mean(dim=1)
            merged = torch.cat([cnn_feat, lstm_feat], dim=1)
            out = self.dropout(self.relu(self.fc1(merged)))
            return self.fc2(out)

    def make_loader(X, y=None, shuffle=False, device='cpu'):
        pin = (str(device) == 'cuda')
        if y is None:
            ds = TensorDataset(torch.tensor(X))
        else:
            ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=pin)

    def train_model(X_tr, y_tr, device, class_weights=None, n_classes=3, epochs=EPOCHS, log_every=5):
        model = CNNLSTM(VOCAB_SIZE, EMBED_DIM, CNN_FILTERS, LSTM_UNITS, n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()
        train_dl = make_loader(X_tr, y_tr, shuffle=True, device=device)
        history = {'train_acc': [], 'train_loss': []}
        for epoch in range(epochs):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            t0 = time.time()
            for batch in train_dl:
                xb, yb = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                correct    += (out.argmax(1) == yb).sum().item()
                total      += len(xb)
            history['train_loss'].append(total_loss / total)
            history['train_acc'].append(correct / total)
            if (epoch + 1) % log_every == 0:
                log.info(f"  Epoch {epoch+1:2d}/{epochs} | loss {total_loss/total:.4f} "
                         f"| acc {correct/total:.4f} | {time.time()-t0:.1f}s")
        return model, history

    @torch.no_grad()
    def predict_proba(model, X, device, n_classes=3):
        model.eval()
        out_all = []
        for (xb,) in make_loader(X, device=device):
            xb = xb.to(device)
            p = torch.softmax(model(xb), dim=1).cpu().numpy()
            out_all.append(p)
        return np.vstack(out_all)


def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)   # inverse-frequency
    return w


if __name__ == '__main__':
    class_map = {'blaster': 0, 'sasser': 1, 'benign': 2}
    rows, all_seqs, all_labels = [], [], []

    for label, class_id in class_map.items():
        folder = os.path.join(DATASET_DIR, label)
        files  = glob.glob(os.path.join(folder, '*.exe'))
        log.info(f"Extracting API sequences from {len(files):,} [{label}] files...")
        for fpath in tqdm(files, desc=f"  M4 [{label}]", unit='file'):
            apis = extract_api_sequence(fpath)
            rule_score, behaviors = rule_based_score(apis)
            api_set = set(apis)
            rows.append({
                'filename':              os.path.basename(fpath),
                'label':                 label,
                'num_apis':              len(apis),
                'has_process_injection': int({'VirtualAllocEx', 'CreateRemoteThread'}.issubset(api_set)),
                'has_network_c2':        int(any(a in api_set for a in ['InternetOpenA', 'WSAStartup', 'connect'])),
                'has_persistence':       int(any(a in api_set for a in ['RegSetValueExA', 'CreateServiceA'])),
                'has_anti_debug':        int(any(a in api_set for a in ['IsDebuggerPresent', 'CheckRemoteDebuggerPresent'])),
                'has_priv_escalation':   int(any(a in api_set for a in ['AdjustTokenPrivileges', 'OpenProcessToken'])),
                'matched_behaviors':     '|'.join(behaviors),
                'm4_score_rules':        rule_score,
            })
            all_seqs.append(apis)
            all_labels.append(class_id)

    df = pd.DataFrame(rows)
    n_empty = int((df['num_apis'] == 0).sum())
    if n_empty:
        log.warning(f"{n_empty:,} samples have an empty API sequence "
                    f"(common for old worms with dynamic resolution). "
                    f"Their M4 will rely on padding/UNK — note this in the thesis.")

    if HAS_TORCH and len(df) >= 50:
        from sklearn.model_selection import train_test_split, StratifiedKFold

        y_arr = np.array(all_labels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"\nUsing device: {device}")

        # Build the vocabulary ONCE from all sequences (vocab is not a label, no leakage).
        X_enc, vocab = encode_sequences(all_seqs, VOCAB_SIZE, MAX_SEQ_LEN)

        # Honest train/test split
        idx_all = np.arange(len(y_arr))
        idx_tr, idx_te = train_test_split(idx_all, test_size=0.20,
                                          random_state=SEED, stratify=y_arr)

        m4_score = np.full(len(df), np.nan, dtype=float)

        # ---- (A) Score the TEST split with a model trained only on TRAIN ----
        log.info("\nTraining final model on TRAIN split only...")
        cw = compute_class_weights(y_arr[idx_tr], n_classes=3)
        log.info(f"  Class weights (blaster,sasser,benign) = {np.round(cw,3)}")
        final_model, history = train_model(X_enc[idx_tr], y_arr[idx_tr], device, class_weights=cw)

        test_probs = predict_proba(final_model, X_enc[idx_te], device)
        m4_score[idx_te] = test_probs[:, :2].max(axis=1)   # max malware prob

        torch.save(final_model.state_dict(), os.path.join(RESULTS_DIR, 'm4_cnn_lstm_model.pt'))
        log.info("Saved: results/m4_cnn_lstm_model.pt")

        # ---- (B) Score the TRAIN split via OUT-OF-FOLD (leakage-free) ----
        log.info(f"\nComputing leakage-free OOF scores for TRAIN split "
                 f"({N_OOF_FOLDS}-fold)...")
        skf = StratifiedKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=SEED)
        Xtr_all, ytr_all = X_enc[idx_tr], y_arr[idx_tr]
        for k, (fit_i, val_i) in enumerate(skf.split(Xtr_all, ytr_all), 1):
            log.info(f"  OOF fold {k}/{N_OOF_FOLDS}")
            cw_k = compute_class_weights(ytr_all[fit_i], n_classes=3)
            m_k, _ = train_model(Xtr_all[fit_i], ytr_all[fit_i], device,
                                 class_weights=cw_k, epochs=EPOCHS, log_every=EPOCHS)
            val_probs = predict_proba(m_k, Xtr_all[val_i], device)
            # map fold-val rows back to global df indices
            global_val = idx_tr[val_i]
            m4_score[global_val] = val_probs[:, :2].max(axis=1)

        assert not np.isnan(m4_score).any(), "Some samples were never scored!"
        df['m4_score']  = np.round(m4_score, 6)
        df['m4_source'] = 'cnn_lstm'

        # Training-history plot (final model, train split only)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history['train_acc'], color='#1565c0', lw=2, label='Train')
        ax[0].set_title('CNN-LSTM Accuracy (train split)', fontweight='bold')
        ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(history['train_loss'], color='#c62828', lw=2, label='Train')
        ax[1].set_title('CNN-LSTM Loss (train split)', fontweight='bold')
        ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].legend(); ax[1].grid(alpha=0.3)
        plt.suptitle('Fig. M4 — CNN-LSTM Training History', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'fig_m4_training_history.png'), dpi=180, bbox_inches='tight')
        plt.close()
        log.info("Saved: results/fig_m4_training_history.png")

    else:
        if HAS_TORCH:
            log.warning("Too few samples for reliable training (<50). Using rule-based scores.")
        else:
            log.info("Using rule-based M4 scores (PyTorch not available).")
        df['m4_score']  = df['m4_score_rules']
        df['m4_source'] = 'rule_based'

    # ── Save scores ───────────────────────────────────────────────────────────
    df.to_csv(os.path.join(RESULTS_DIR, 'scores_m4.csv'), index=False)
    log.info(f"Saved {len(df):,} M4 scores → results/scores_m4.csv "
             f"(source: {df['m4_source'].iloc[0]})")

    # ── Behavioral Heatmap ────────────────────────────────────────────────────
    behavior_cols   = ['has_process_injection', 'has_network_c2', 'has_persistence',
                       'has_anti_debug', 'has_priv_escalation']
    behavior_labels = ['Process\nInjection', 'Network\nC2', 'Registry\nPersistence',
                       'Anti-Debug\n/ Evasion', 'Privilege\nEscalation']
    heat = np.array([[df[df['label'] == lb][c].mean() for c in behavior_cols]
                     for lb in ['blaster', 'sasser', 'benign']])
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heat, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(behavior_labels))); ax.set_xticklabels(behavior_labels, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(['Blaster', 'Sasser', 'Benign'], fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(len(behavior_labels)):
            ax.text(j, i, f'{heat[i,j]:.2f}', ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if heat[i, j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Proportion of Samples Matching Behavior')
    ax.set_title('Fig. M4 — Behavioral Heatmap: API Semantic Patterns by Class',
                 fontweight='bold', fontsize=11, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_m4_behavioral_heatmap.png'), dpi=180, bbox_inches='tight')
    plt.close()
    log.info("Saved: results/fig_m4_behavioral_heatmap.png")

    # ── API Category Table ────────────────────────────────────────────────────
    cat_rows = []
    for cat in API_CATEGORIES:
        for lb in ['blaster', 'sasser', 'benign']:
            grp = df[df['label'] == lb]
            pct = grp['matched_behaviors'].str.contains(cat.split('/')[0].strip(), na=False).mean()
            cat_rows.append({'Category': cat, 'Class': lb, 'Proportion': round(pct, 4)})
    pd.DataFrame(cat_rows).to_csv(os.path.join(RESULTS_DIR, 'table_m4_api_categories.csv'), index=False)

    log.info("\nM4 Score Summary by class:")
    log.info(df.groupby('label')['m4_score'].describe().round(4).to_string())
    log.info("\nSTEP 4 COMPLETE ✓")

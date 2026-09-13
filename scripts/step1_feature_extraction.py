"""
WHSF — Step 1: PE Feature Extraction (Binary .exe files)
=========================================================
Dataset structure expected:
    dataset/blaster/   *.exe   (50,000 Blaster worm samples)
    dataset/sasser/    *.exe   (50,000 Sasser worm samples)
    dataset/benign/    *.exe   (100,000 benign executables)

Extracts 54 PE structural features per file.
Output: results/features.csv

Install requirements:
    pip install pefile pandas tqdm
"""

import os, glob, math, hashlib, struct, logging
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ── Optional: use pefile for robust parsing ───────────────────────────────────
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False
    logging.warning("pefile not installed. Run: pip install pefile")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATASET_DIR = 'dataset'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def _atomic_write_csv(df, path):
    """Write a DataFrame to CSV atomically.
    Writes to a temp file in the SAME directory, flushes to disk, then
    os.replace() swaps it into place in one indivisible operation. If the
    process is killed mid-write, the original file is left intact.
    """
    import tempfile
    directory = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(suffix='.tmp', prefix='.features_', dir=directory)
    ok = False
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())     # force bytes to physical disk before swap
        os.replace(tmp, path)         # atomic on Windows and POSIX
        ok = True
    finally:
        # if the swap did not complete, remove the temp; original stays intact
        if not ok:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

# ── Shannon Entropy ───────────────────────────────────────────────────────────
def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    c = Counter(data)
    n = len(data)
    return round(-sum((v/n) * math.log2(v/n) for v in c.values()), 6)

# ── Known packer signatures (section name heuristics) ────────────────────────
PACKER_NAMES = {'upx0','upx1','upx2','.aspack','.adata','themida',
                'execrypt','.nsp0','.nsp1','pec2','.petite'}

def _read_bytes(filepath: str):
    """Read a file's bytes, tolerating Windows long paths and odd names.
    Returns bytes on success, or None if the file cannot be read."""
    candidates = [filepath]
    # Windows extended-length path lets us exceed the 260-char MAX_PATH limit
    if os.name == 'nt':
        try:
            ap = os.path.abspath(filepath)
            if not ap.startswith('\\\\?\\'):
                candidates.append('\\\\?\\' + ap)
        except Exception:
            pass
    for path in candidates:
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception:
            continue
    return None


# ── Extract all 54 features from one .exe file ────────────────────────────────
def extract_features(filepath: str) -> dict:
    feats = {
        'filename': os.path.basename(filepath),
        'file_size': 0, 'md5': '', 'sha256': '',
        'overall_entropy': 0.0, 'is_pe': 0,
        # Header fields
        'e_lfanew': 0, 'machine': 0, 'num_sections': 0,
        'timestamp': 0, 'size_of_code': 0, 'size_of_init_data': 0,
        'size_of_uninit_data': 0, 'entry_point': 0, 'image_base': 0,
        'section_alignment': 0, 'file_alignment': 0,
        'major_os_version': 0, 'minor_os_version': 0,
        'major_image_version': 0, 'minor_image_version': 0,
        'major_subsystem_version': 0, 'minor_subsystem_version': 0,
        'size_of_image': 0, 'size_of_headers': 0,
        'checksum': 0, 'checksum_valid': 0, 'zero_checksum': 0,
        'subsystem': 0, 'dll_characteristics': 0, 'num_rva_and_sizes': 0,
        # Section analysis
        'num_sections_text': 0, 'num_sections_data': 0,
        'num_sections_rsrc': 0, 'num_sections_rdata': 0,
        'num_sections_reloc': 0,
        'entropy_text': 0.0, 'entropy_data': 0.0,
        'entropy_rsrc': 0.0, 'entropy_rdata': 0.0, 'entropy_reloc': 0.0,
        'max_section_entropy': 0.0, 'min_section_entropy': 0.0,
        'avg_section_entropy': 0.0, 'num_high_entropy_sections': 0,
        'section_name_anomaly': 0, 'packer_detected': 0,
        # Import / Export
        'iat_size': 0, 'num_imports': 0,
        'num_exports': 0, 'has_tls': 0,
        # Strings
        'num_strings': 0, 'avg_string_len': 0.0,
        'has_url_string': 0, 'has_registry_string': 0,
        # Anomaly flags
        'suspicious_entry_point': 0,
        'virtual_size_discrepancy': 0,
    }

    raw = _read_bytes(filepath)
    if raw is None:
        # unreadable file (long path, permission, symlink): signal caller to SKIP
        return None

    feats['file_size']       = len(raw)
    feats['md5']             = hashlib.md5(raw).hexdigest()
    feats['sha256']          = hashlib.sha256(raw).hexdigest()
    feats['overall_entropy'] = entropy(raw)
    feats['is_pe']           = 1 if raw[:2] == b'MZ' else 0

    if not feats['is_pe']:
        return feats

    # ── pefile path (preferred) ───────────────────────────────────────────────
    if HAS_PEFILE:
        try:
            pe = pefile.PE(data=raw, fast_load=False)
            oh  = pe.OPTIONAL_HEADER
            fh  = pe.FILE_HEADER

            feats['e_lfanew']               = pe.DOS_HEADER.e_lfanew
            feats['machine']                = fh.Machine
            feats['num_sections']           = fh.NumberOfSections
            feats['timestamp']              = fh.TimeDateStamp
            feats['size_of_code']           = oh.SizeOfCode
            feats['size_of_init_data']      = oh.SizeOfInitializedData
            feats['size_of_uninit_data']    = oh.SizeOfUninitializedData
            feats['entry_point']            = oh.AddressOfEntryPoint
            feats['image_base']             = oh.ImageBase
            feats['section_alignment']      = oh.SectionAlignment
            feats['file_alignment']         = oh.FileAlignment
            feats['major_os_version']       = oh.MajorOperatingSystemVersion
            feats['minor_os_version']       = oh.MinorOperatingSystemVersion
            feats['major_image_version']    = oh.MajorImageVersion
            feats['minor_image_version']    = oh.MinorImageVersion
            feats['major_subsystem_version']= oh.MajorSubsystemVersion
            feats['minor_subsystem_version']= oh.MinorSubsystemVersion
            feats['size_of_image']          = oh.SizeOfImage
            feats['size_of_headers']        = oh.SizeOfHeaders
            feats['checksum']               = oh.CheckSum
            feats['checksum_valid']         = 1 if oh.CheckSum != 0 else 0
            feats['zero_checksum']          = 1 if oh.CheckSum == 0 else 0
            feats['subsystem']              = oh.Subsystem
            feats['dll_characteristics']    = oh.DllCharacteristics
            feats['num_rva_and_sizes']      = oh.NumberOfRvaAndSizes
            feats['suspicious_entry_point'] = 1 if oh.AddressOfEntryPoint == 0 else 0

            # Sections
            section_entropies = []
            sec_counts = {'.text':0,'.data':0,'.rsrc':0,'.rdata':0,'.reloc':0}
            packer_flag = 0
            vsize_disc  = 0

            for sec in pe.sections:
                name = sec.Name.rstrip(b'\x00').decode('ascii','replace').lower().strip()
                sec_data = sec.get_data()
                sec_ent  = entropy(sec_data)
                section_entropies.append(sec_ent)

                if name in sec_counts:
                    sec_counts[name] += 1
                    feats[f'entropy_{name[1:]}'] = sec_ent   # entropy_text, etc.
                if name in PACKER_NAMES:
                    packer_flag = 1

                # Virtual size vs raw size discrepancy
                if sec.Misc_VirtualSize > 0 and sec.SizeOfRawData > 0:
                    ratio = sec.Misc_VirtualSize / sec.SizeOfRawData
                    if ratio > 10 or ratio < 0.1:
                        vsize_disc = 1

            feats.update({
                'num_sections_text':  sec_counts['.text'],
                'num_sections_data':  sec_counts['.data'],
                'num_sections_rsrc':  sec_counts['.rsrc'],
                'num_sections_rdata': sec_counts['.rdata'],
                'num_sections_reloc': sec_counts['.reloc'],
                'max_section_entropy': max(section_entropies) if section_entropies else 0.0,
                'min_section_entropy': min(section_entropies) if section_entropies else 0.0,
                'avg_section_entropy': round(sum(section_entropies)/len(section_entropies), 6) if section_entropies else 0.0,
                'num_high_entropy_sections': sum(1 for e in section_entropies if e > 7.0),
                'packer_detected': packer_flag,
                'virtual_size_discrepancy': vsize_disc,
            })

            # Imports
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                total_imports = sum(len(e.imports) for e in pe.DIRECTORY_ENTRY_IMPORT)
                feats['num_imports'] = total_imports
                if hasattr(oh, 'DATA_DIRECTORY'):
                    iat_dir = oh.DATA_DIRECTORY[12]
                    feats['iat_size'] = iat_dir.Size

            # Exports
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                feats['num_exports'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)

            # TLS
            if hasattr(pe, 'DIRECTORY_ENTRY_TLS'):
                feats['has_tls'] = 1

            # Section name anomaly (non-standard names)
            standard = {'.text','.data','.rsrc','.rdata','.reloc','.bss',
                        '.idata','.edata','.tls','.debug','.pdata','.xdata'}
            anomaly = sum(
                1 for sec in pe.sections
                if sec.Name.rstrip(b'\x00').decode('ascii','replace').lower().strip()
                not in standard
            )
            feats['section_name_anomaly'] = anomaly

            pe.close()

        except Exception as e:
            log.debug(f"pefile error on {filepath}: {e}")

    # ── String analysis (always run) ──────────────────────────────────────────
    strings = []
    curr = []
    for b in raw:
        if 0x20 <= b <= 0x7e:
            curr.append(chr(b))
        else:
            if len(curr) >= 4:
                strings.append(''.join(curr))
            curr = []
    if len(curr) >= 4:
        strings.append(''.join(curr))

    feats['num_strings']         = len(strings)
    feats['avg_string_len']      = round(sum(len(s) for s in strings)/len(strings), 2) if strings else 0.0
    feats['has_url_string']      = 1 if any('http' in s.lower() for s in strings) else 0
    feats['has_registry_string'] = 1 if any('software\\' in s.lower() or 'hkey_' in s.lower() for s in strings) else 0

    return feats


# ── Main ──────────────────────────────────────────────────────────────────────

# ── Parallel worker (must be top-level so multiprocessing can pickle it) ──────
def _process_one(args):
    """Worker: extract features for one file and attach its label.
    Returns None if the file was unreadable (so it is skipped, not stored)."""
    fpath, label = args
    row = extract_features(fpath)
    if row is None:
        return ('SKIP', os.path.basename(fpath))
    row['label'] = label
    return ('OK', row)


if __name__ == '__main__':
    from multiprocessing import Pool, cpu_count

    if not HAS_PEFILE:
        log.warning("Installing pefile...")
        os.system('pip install pefile -q')
        import pefile
        HAS_PEFILE = True

    FEATURES_PATH = os.path.join(RESULTS_DIR, 'features.csv')

    # ── RESUMABLE: load any existing features and skip those filenames ─────────
    if os.path.exists(FEATURES_PATH) and os.path.getsize(FEATURES_PATH) > 0:
        # Read ONLY the filename column to keep memory low on resume.
        try:
            done_files = set(pd.read_csv(FEATURES_PATH, usecols=['filename'])['filename'].astype(str))
        except Exception:
            done_files = set()
        log.info(f"Found existing features.csv with {len(done_files):,} rows already extracted.")
    else:
        done_files = set()
        log.info("No existing features.csv — extracting from scratch.")

    class_map = {
        'blaster': os.path.join(DATASET_DIR, 'blaster'),
        'sasser':  os.path.join(DATASET_DIR, 'sasser'),
        'benign':  os.path.join(DATASET_DIR, 'benign'),
    }

    tasks = []
    for label, folder in class_map.items():
        files = glob.glob(os.path.join(folder, '*.exe'))
        log.info(f"[{label.upper()}] Found {len(files):,} .exe files in '{folder}'")
        for f in files:
            if os.path.basename(f) not in done_files:
                tasks.append((f, label))

    total_on_disk = sum(len(glob.glob(os.path.join(fld, '*.exe'))) for fld in class_map.values())
    if total_on_disk == 0:
        log.error("No .exe files found in dataset/blaster, dataset/sasser, dataset/benign. "
                  "Check the folder names and that files end with .exe.")
        raise SystemExit(1)

    log.info(f"Total .exe on disk : {total_on_disk:,}")
    log.info(f"Already extracted  : {len(done_files):,}")
    log.info(f"To extract (new)   : {len(tasks):,}")

    import csv as _csv

    # Stable column order taken from a template feature dict (no file needed).
    _template = extract_features.__wrapped__ if hasattr(extract_features, "__wrapped__") else None
    # Build the canonical column order by extracting the keys from a dummy dict.
    ID_COLS = ['filename', 'label', 'md5', 'sha256']
    # Get feature keys by calling the builder on a tiny in-memory non-PE blob path is hard;
    # instead reconstruct from the known dict via a throwaway extraction of an empty temp.
    import tempfile as _tf
    _fd, _tmpf = _tf.mkstemp(suffix='.exe'); os.close(_fd)
    try:
        _sample = extract_features(_tmpf) or {}
    finally:
        try: os.remove(_tmpf)
        except Exception: pass
    _feat_keys = [k for k in _sample.keys() if k not in ID_COLS]
    COLUMNS = ID_COLS + _feat_keys

    # Determine whether the CSV already has a header (resume) or must be created.
    file_exists = os.path.exists(FEATURES_PATH) and os.path.getsize(FEATURES_PATH) > 0

    skipped = []
    newly = 0

    if not tasks:
        log.info("Nothing new to extract. features.csv is already complete.")
    else:
        n_workers = max(1, cpu_count() - 1)
        log.info(f"Extracting {len(tasks):,} files using {n_workers} parallel workers...")

        FLUSH_EVERY = 2000
        buffer = []

        # Open the CSV in APPEND mode. We never hold all rows in memory — the OS
        # file is the source of truth, so memory stays flat regardless of size.
        write_header = not file_exists
        fout = open(FEATURES_PATH, 'a', newline='', encoding='utf-8')
        writer = _csv.DictWriter(fout, fieldnames=COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
            fout.flush()

        try:
            with Pool(processes=n_workers) as pool:
                for i, result in enumerate(tqdm(pool.imap_unordered(_process_one, tasks, chunksize=100),
                                                total=len(tasks), desc="  Extracting (parallel)", unit='file'), 1):
                    tag, payload = result
                    if tag == 'OK':
                        buffer.append(payload)
                        newly += 1
                    else:                       # 'SKIP'
                        skipped.append(payload)

                    if len(buffer) >= FLUSH_EVERY:
                        writer.writerows(buffer)
                        fout.flush(); os.fsync(fout.fileno())   # durable checkpoint
                        buffer.clear()                          # free memory
                        log.info(f"  checkpoint: {i:,}/{len(tasks):,} processed, {len(skipped):,} skipped")
            # flush any remaining rows
            if buffer:
                writer.writerows(buffer)
                fout.flush(); os.fsync(fout.fileno())
                buffer.clear()
        finally:
            fout.close()

    # ── Record skipped (unreadable) files for transparency in the thesis ──────
    if skipped:
        skip_path = os.path.join(RESULTS_DIR, 'skipped_files.txt')
        # append so reruns accumulate the full skip list
        with open(skip_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(skipped) + "\n")
        log.warning(f"{len(skipped):,} file(s) were unreadable and EXCLUDED "
                    f"(list saved to {skip_path}). They are NOT in features.csv.")

    # ── Final count (read only the filename column to stay memory-light) ──────
    try:
        total_rows = sum(1 for _ in open(FEATURES_PATH, encoding='utf-8')) - 1
    except Exception:
        total_rows = -1

    log.info(f"\n{'='*50}")
    log.info(f"STEP 1 COMPLETE (parallel + resumable, low-memory append)")
    log.info(f"  Rows in features.csv : {total_rows:,}")
    log.info(f"  Features             : {len(_feat_keys)}")
    log.info(f"  Newly extracted      : {newly:,}")
    log.info(f"  Skipped (unreadable) : {len(skipped):,}")
    log.info(f"  Output               : {FEATURES_PATH}")
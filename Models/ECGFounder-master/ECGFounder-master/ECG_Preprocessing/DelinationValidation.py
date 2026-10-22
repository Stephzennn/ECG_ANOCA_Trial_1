"""
qtdb_explore.py - first look at the PhysioNet QT Database (QTDB).

What it does
  1. Downloads QTDB once into a local folder (optional, --download).
  2. Builds an overview table of all records: lead names, sampling rate,
     duration, which annotation files exist, and how many beats the
     cardiologist annotated. Records with an MLII lead are flagged.
  3. For one record, turns the cardiologist's annotations (.q1c) into a
     beat-by-beat table of fiducial points (P/QRS/T onset, peak, offset).
  4. Plots ~10 s of that record (both leads) with the fiducial points marked.

Usage
  pip install wfdb matplotlib pandas
  python qtdb_explore.py --download                 # first run: get the data
  python qtdb_explore.py                            # overview + plot sel100
  python qtdb_explore.py --record sel102 --seconds 6

Outputs (in ./qtdb_outputs/)
  qtdb_overview.csv            one row per record
  <record>_q1c_fiducials.csv   one row per annotated beat (times in seconds)
  <record>_plot.png            signal + annotations

QTDB annotation files, for reference
  .q1c  cardiologist 1 waveform boundaries  <- the reference you validate against
  .q2c  cardiologist 2 (only some records)  <- for inter-observer comparison
  .pu0/.pu1/.pu  automatic ecgpuwave output (not ground truth)
  .atr/.man      beat-type annotations (not boundaries)
Symbols in .q1c:  '(' onset, ')' offset, 'p' P peak, 'N' R peak, 't' T peak, 'u' U peak.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wfdb

PEAK_TO_WAVE = {"p": "P", "N": "QRS", "t": "T", "u": "U"}


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading QTDB into {data_dir} ...")
    wfdb.dl_database("qtdb", dl_dir=data_dir)
    print("Done.")


def list_records(data_dir):
    records_file = os.path.join(data_dir, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file) as f:
            return [line.strip() for line in f if line.strip()]
    # fall back to any .hea file in the folder
    return sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea"))


def overview(data_dir):
    rows = []
    for rec in list_records(data_dir):
        path = os.path.join(data_dir, rec)
        try:
            hdr = wfdb.rdheader(path)
        except Exception as e:
            print(f"  skip {rec}: {e}")
            continue
        exts = [e for e in ("q1c", "q2c", "pu0", "pu1", "atr", "man")
                if os.path.exists(f"{path}.{e}")]
        n_beats = 0
        if "q1c" in exts:
            ann = wfdb.rdann(path, "q1c")
            n_beats = sum(s == "N" for s in ann.symbol)
        names = [n.strip() for n in hdr.sig_name]
        rows.append({
            "record": rec,
            "lead_0": names[0] if names else None,
            "lead_1": names[1] if len(names) > 1 else None,
            "has_MLII": any(n.upper() == "MLII" for n in names),
            "fs_hz": hdr.fs,
            "duration_min": round(hdr.sig_len / hdr.fs / 60, 1),
            "q1c_beats": n_beats,
            "annotation_files": ",".join(exts),
        })
    return pd.DataFrame(rows)


def q1c_to_fiducials(ann, fs):
    """Convert the flat q1c annotation stream into one row per beat.

    A wave is written as '(' peak ')'. The onset/offset belong to the peak
    they directly surround; a missing bracket is left blank.
    """
    syms, samps = ann.symbol, ann.sample
    waves = []  # (wave, onset, peak, offset) in samples
    for i, s in enumerate(syms):
        if s not in PEAK_TO_WAVE:
            continue
        on = samps[i - 1] if i > 0 and syms[i - 1] == "(" else None
        off = samps[i + 1] if i + 1 < len(syms) and syms[i + 1] == ")" else None
        waves.append((PEAK_TO_WAVE[s], on, samps[i], off))

    # group waves into beats, anchored on each QRS
    beats = []
    qrs_idx = [k for k, w in enumerate(waves) if w[0] == "QRS"]
    for j, k in enumerate(qrs_idx):
        prev_qrs = qrs_idx[j - 1] if j > 0 else -1
        next_qrs = qrs_idx[j + 1] if j + 1 < len(qrs_idx) else len(waves)
        beat = {"beat": j + 1}
        for name, idx_range in (("P", range(prev_qrs + 1, k)),
                                ("QRS", [k]),
                                ("T", range(k + 1, next_qrs))):
            match = [waves[m] for m in idx_range if waves[m][0] == name]
            # P: take the one closest before QRS; T: closest after QRS
            w = (match[-1] if name == "P" else match[0]) if match else None
            for part, val in zip(("on", "peak", "off"), w[1:] if w else (None,) * 3):
                beat[f"{name}_{part}_s"] = round(val / fs, 4) if val is not None else None
        beats.append(beat)

    df = pd.DataFrame(beats)
    if not df.empty:
        df["PR_ms"] = (df["QRS_on_s"] - df["P_on_s"]) * 1000
        df["QRS_ms"] = (df["QRS_off_s"] - df["QRS_on_s"]) * 1000
        df["QT_ms"] = (df["T_off_s"] - df["QRS_on_s"]) * 1000
        df[["PR_ms", "QRS_ms", "QT_ms"]] = df[["PR_ms", "QRS_ms", "QT_ms"]].round(1)
    return df


def plot_record(path, rec, ann, seconds, out_png):
    hdr = wfdb.rdheader(path)
    fs = hdr.fs
    # q1c annotations cover only selected beats, so centre the window on them
    start = max(0, int(ann.sample[0] - 1.0 * fs))
    stop = min(hdr.sig_len, start + int(seconds * fs))
    sig, fields = wfdb.rdsamp(path, sampfrom=start, sampto=stop)
    t = [(start + i) / fs for i in range(sig.shape[0])]

    styles = {"(": ("|", "green", "onset"), ")": ("|", "red", "offset"),
              "p": ("o", "orange", "P peak"), "N": ("^", "purple", "R peak"),
              "t": ("s", "blue", "T peak"), "u": ("D", "gray", "U peak")}

    n_leads = sig.shape[1]
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 3.2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]
    for ch, ax in enumerate(axes):
        ax.plot(t, sig[:, ch], lw=0.8, color="black")
        seen = set()
        for s, smp in zip(ann.symbol, ann.sample):
            if not (start <= smp < stop) or s not in styles:
                continue
            marker, color, label = styles[s]
            y = sig[smp - start, ch]
            ax.plot(smp / fs, y, marker, color=color, ms=8 if marker == "|" else 5,
                    label=label if label not in seen else None)
            seen.add(label)
        ax.set_ylabel(f"{fields['sig_name'][ch]} ({fields['units'][ch]})")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"QTDB {rec}: cardiologist annotations (.q1c), fs={fs} Hz")
    axes[0].legend(loc="upper right", ncol=6, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="qtdb_data")
    ap.add_argument("--out-dir", default="qtdb_outputs")
    ap.add_argument("--download", action="store_true", help="download QTDB first")
    ap.add_argument("--record", default="sel100")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    if args.download:
        download(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. overview of all records
    ov = overview(args.data_dir)
    ov_path = os.path.join(args.out_dir, "qtdb_overview.csv")
    ov.to_csv(ov_path, index=False)
    print(f"\n{len(ov)} records  |  {int(ov['has_MLII'].sum())} with MLII  |  "
          f"{int((ov['q1c_beats'] > 0).sum())} with cardiologist annotations (.q1c)")
    print(f"Lead combinations:\n{ov.groupby(['lead_0', 'lead_1']).size().sort_values(ascending=False).to_string()}")
    print(f"Saved {ov_path}")

    # 2. one record in detail
    path = os.path.join(args.data_dir, args.record)
    if not os.path.exists(f"{path}.q1c"):
        print(f"\n{args.record} has no .q1c file; pick a record with q1c_beats > 0.")
        return
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "q1c")
    fid = q1c_to_fiducials(ann, hdr.fs)
    fid_path = os.path.join(args.out_dir, f"{args.record}_q1c_fiducials.csv")
    fid.to_csv(fid_path, index=False)
    print(f"\n{args.record}: leads {hdr.sig_name}, {len(fid)} annotated beats")
    print(fid[["PR_ms", "QRS_ms", "QT_ms"]].describe().round(1).to_string())
    print(f"Saved {fid_path}")

    png = os.path.join(args.out_dir, f"{args.record}_plot.png")
    plot_record(path, args.record, ann, args.seconds, png)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()


import os, glob
import wfdb

def get_qtdb_files(data_dir="qtdb_data"):
    """Download QTDB if needed; return {record: {extension: path}}."""
    if not glob.glob(os.path.join(data_dir, "*.hea")):
        wfdb.dl_database("qtdb", dl_dir=data_dir)
    files = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.*"))):
        rec, ext = os.path.splitext(os.path.basename(path))
        if ext in (".hea", ".dat", ".q1c", ".q2c", ".pu", ".pu0", ".pu1", ".atr", ".man"):
            files.setdefault(rec, {})[ext[1:]] = path
    return files

files = get_qtdb_files()

#Here another part of the data

import os, glob
import wfdb

def get_ludb_files(data_dir="ludb_data"):
    """Download LUDB if needed; return {record: {extension: path}}."""
    pattern = os.path.join(data_dir, "**", "*.hea")
    if not glob.glob(pattern, recursive=True):
        wfdb.dl_database("ludb", dl_dir=data_dir)
    files = {}
    for hea in glob.glob(pattern, recursive=True):
        base = hea[:-4]
        rec = os.path.basename(base)
        files[rec] = {os.path.splitext(p)[1][1:]: p for p in glob.glob(base + ".*")}
    return dict(sorted(files.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))

filesLUDB = get_ludb_files()






rec = wfdb.rdrecord("qtdb_data/sel100")
rec.p_signal   # array, shape (225000, 2), in mV
rec.sig_name   # ['MLII', 'V5'] (or whatever the record has)
rec.fs         # 250

ann = wfdb.rdann("qtdb_data/sel100", "q1c")
ann.sample   # e.g. [ 2010, 2025, 2041, 2060, 2071, 2085, 2140, 2188, ...]
ann.symbol   # e.g. [  '(',  'p',  ')',  '(',  'N',  ')',  't',  ')', ...]

import pandas as pd
df = pd.DataFrame({"sample": ann.sample, "time_s": ann.sample / rec.fs, "symbol": ann.symbol}).head(20)

if "MLII" in rec.sig_name:
    lead = rec.sig_name.index("MLII")

#========================= Check Test Chek Test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

# Landmark: (key in waves, label, color, marker)
LANDMARKS = [
    ("ECG_P_Onsets",  "P on",   "#7F8C8D", "|"),
    ("ECG_P_Peaks",   "P",      "#7F8C8D", "o"),
    ("ECG_P_Offsets", "P off",  "#7F8C8D", "|"),
    ("ECG_R_Onsets",  "QRS on", "#2E86C1", "|"),
    ("ECG_R_Offsets", "QRS off","#2E86C1", "v"),
    ("ECG_T_Onsets",  "T on",   "#E67E22", ">"),
    ("ECG_T_Peaks",   "T",      "#B03A2E", "o"),
    ("ECG_T_Offsets", "T off",  "#27AE60", "<"),
]


def ann_df_to_waves(df):
    """QTDB annotation dataframe (sample, symbol) -> NeuroKit-style waves dict."""
    sym = df["symbol"].tolist()
    smp = df["sample"].to_numpy()
    peak_to_wave = {"p": "P", "N": "R", "t": "T"}
    waves = {f"ECG_{w}_{k}": [] for w in ("P", "R", "T")
             for k in ("Onsets", "Peaks", "Offsets")}
    for i, s in enumerate(sym):
        if s not in peak_to_wave:
            continue
        w = peak_to_wave[s]
        waves[f"ECG_{w}_Peaks"].append(smp[i])
        if i > 0 and sym[i - 1] == "(":                 # onset right before the peak
            waves[f"ECG_{w}_Onsets"].append(smp[i - 1])
        if i + 1 < len(sym) and sym[i + 1] == ")":      # offset right after the peak
            waves[f"ECG_{w}_Offsets"].append(smp[i + 1])
    return {k: np.array(v, dtype=float) for k, v in waves.items()}


def plot_delineation(signal, waves, rpeaks=None, start=0, dur=5, fs=250,
                     ax=None, title=""):
    """Overlay delineated points on a window of the signal."""
    N = int(dur * fs)
    start = int(min(max(start, 0), len(signal) - N))
    stop = start + N
    t = np.arange(start, stop) / fs

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    ax.plot(t, signal[start:stop], linewidth=0.9, color="#34495E", zorder=1)

    if rpeaks is not None:
        r = [int(x) for x in rpeaks if start <= x < stop]
        ax.plot(np.array(r)/fs, signal[r], "*", color="#8E44AD",
                markersize=13, label="R peak", zorder=3)

    for key, label, color, marker in LANDMARKS:
        if key not in waves:
            continue
        pts = [int(x) for x in waves[key]
               if np.isfinite(x) and start <= x < stop]
        if not pts:
            continue
        ax.plot(np.array(pts)/fs, signal[pts], marker, color=color,
                markersize=8, linestyle="none", label=label, zorder=3)
        for p in pts:                      # text label on each point
            ax.annotate(label, (p/fs, signal[p]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=color)

    ax.set_title(title or f"{start/fs:.1f}–{stop/fs:.1f} s")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=9, fontsize=8, loc="upper right")
    return ax


# --- usage ---
rec = wfdb.rdrecord("qtdb_data/sel100")
ann = wfdb.rdann("qtdb_data/sel100", "q1c")
df = pd.DataFrame({"sample": ann.sample, "time_s": ann.sample / rec.fs, "symbol": ann.symbol})

fs = rec.fs                      # 250 for QTDB, not 500
lead = 0                         # 0 or 1; check rec.sig_name for which is MLII
waves = ann_df_to_waves(df)
start = int(df["sample"].iloc[0] - fs)   # 1 s before the first annotated beat

plot_delineation(rec.p_signal[:, lead], waves, waves["ECG_R_Peaks"],
                 start=start, dur=5, fs=fs,
                 title=f"sel100 {rec.sig_name[lead]}: cardiologist (.q1c)")
plt.tight_layout(); plt.show()



pu = wfdb.rdann(f"qtdb_data/{record}", f"pu{lead}")
pu_df = pd.DataFrame({"sample": pu.sample, "symbol": pu.symbol})
plot_delineation(rec.p_signal[:, lead], ann_df_to_waves(pu_df), ...)

#============================ Check Test Check Test



files = get_ludb_files()
base = files["1"]["hea"][:-4]           # e.g. ludb_data/data/1

rec = wfdb.rdrecord(base)
ann = wfdb.rdann(base, "ii")            # lead II's own annotations
lead = [n.lower() for n in rec.sig_name].index("ii")
sig = rec.p_signal[:, lead]
ann = wfdb.rdann(base, "ii")
print(set(ann.symbol))           # expect {'(', ')', 'p', 'N', 't'}
print(ann.symbol[:12])           # expect a pattern like ( p ) ( N ) ( t ) ...

files = get_ludb_files()
base = files["1"]["hea"][:-4]

rec = wfdb.rdrecord(base)
ann = wfdb.rdann(base, "ii")
lead = [n.lower() for n in rec.sig_name].index("ii")
df = pd.DataFrame({"sample": ann.sample, "time_s": ann.sample / rec.fs, "symbol": ann.symbol})


print(rec.record_name, rec.fs, rec.p_signal.shape)   # expect '1', 500, (5000, 12)
print(df.head(12))                                     # samples should be 0–5000
print(waves["ECG_R_Peaks"][:5])                        # e.g. [  10.  660. 1340. ...]

base = files["1"]["hea"][:-4]
rec = wfdb.rdrecord(base)
ann = wfdb.rdann(base, "ii")
df = pd.DataFrame({"sample": ann.sample, "time_s": ann.sample / rec.fs, "symbol": ann.symbol})
waves = ann_df_to_waves(df)

lead = [n.lower() for n in rec.sig_name].index("ii")
plot_delineation(rec.p_signal[:, lead], waves, waves["ECG_R_Peaks"],
                 start=0, dur=10, fs=rec.fs, title="LUDB 1, lead II: cardiologist")
plt.tight_layout(); plt.show()




import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import neurokit2 as nk

# Landmark: (key in waves, label, marker)
LANDMARKS = [
    #("ECG_P_Onsets",  "P on",    "|"),
    #("ECG_P_Peaks",   "P",       "o"),
    #("ECG_P_Offsets", "P off",   "|"),
    #("ECG_R_Onsets",  "QRS on",  ">"),
    #("ECG_R_Peaks",   "R",       "*"),
    #("ECG_R_Offsets", "QRS off", "<"),
    #("ECG_T_Onsets",  "T on",    ">"),
    ("ECG_T_Peaks",   "T",       "o"),
    #("ECG_T_Offsets", "T off",   "<"),
]


def plot_overlay(signal, sources, start=0, dur=10, fs=500, ax=None, title="",
                 show_labels=True):
    """Plot one signal with several sets of delineated points.

    sources: {"name": (waves_dict, color)}. Same marker = same landmark;
    color = which source. Labels go above for the 1st source, below for the 2nd.
    """
    N = int(dur * fs)
    start = int(min(max(start, 0), len(signal) - N))
    stop = start + N
    t = np.arange(start, stop) / fs

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 5))
    ax.plot(t, signal[start:stop], linewidth=0.9, color="#34495E", zorder=1)

    handles = []
    for i, (name, (waves, color)) in enumerate(sources.items()):
        dy = 9 if i == 0 else -14
        for key, label, marker in LANDMARKS:
            pts = [int(x) for x in waves.get(key, [])
                   if np.isfinite(x) and start <= x < stop]
            if not pts:
                continue
            ax.plot(np.array(pts)/fs, signal[pts], marker, color=color,
                    markersize=11 if marker == "*" else 8, linestyle="none", zorder=3)
            for p in (pts if show_labels else []):
                ax.annotate(label, (p/fs, signal[p]), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=7, color=color)
        handles.append(mlines.Line2D([], [], color=color, marker="o",
                                     linestyle="none", label=name))

    ax.set_title(title or f"{start/fs:.1f}–{stop/fs:.1f} s")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    return ax


# --- LUDB record 1, lead II ---
base = files["1"]["hea"][:-4]
rec = wfdb.rdrecord(base)
ann = wfdb.rdann(base, "ii")
df = pd.DataFrame({"sample": ann.sample, "time_s": ann.sample / rec.fs, "symbol": ann.symbol})
waves_card = ann_df_to_waves(df)

fs = rec.fs
lead = [n.lower() for n in rec.sig_name].index("ii")
sig = rec.p_signal[:, lead]

# --- NeuroKit on the same lead ---
method = "dwt"
cleaned = nk.ecg_clean(sig, sampling_rate=fs)
_, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]
_, waves_nk = nk.ecg_delineate(cleaned, rpeaks=rpeaks, sampling_rate=fs, method=method)
waves_nk["ECG_R_Peaks"] = rpeaks          # delineate doesn't return R peaks

plot_overlay(sig,
             {"Cardiologist": (waves_card, "#C0392B"),
              f"NeuroKit ({method})": (waves_nk, "#2471A3")},
             start=0, dur=10, fs=fs, title="LUDB 1, lead II")
plt.tight_layout(); plt.show()




#------------------------ Check Check

"""
t_peak_eval.py - score NeuroKit T-peak delineation against LUDB cardiologist
T peaks, lead II.

For every record:
  1. Read lead II and the cardiologist lead-II annotations (".ii").
  2. Run NeuroKit (clean -> R peaks -> delineate) with the chosen method.
  3. Pair each cardiologist T peak with the nearest NeuroKit T peak within a
     tolerance window (one-to-one, closest pairs first).
  4. Error = NeuroKit time - cardiologist time, in ms
     (positive = NeuroKit is late, negative = early).

Only NeuroKit T peaks inside the annotated span are scored, because LUDB
cardiologists often skip the first and last beat of the 10 s strip.

Usage (notebook):
    from t_peak_eval import evaluate_t_peaks, summarize
    beats, records = evaluate_t_peaks(files, method="dwt")
    summarize(beats, records)

Returns
    beats   one row per cardiologist T peak (matched or missed) and per
            unmatched NeuroKit T peak (false positive)
    records one row per record: counts, status, error message if NeuroKit failed
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb


def ref_t_peaks(ann):
    """Cardiologist T-peak sample indices from a WFDB annotation."""
    return np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym == "t"])


def nk_t_peaks(sig, fs, method="dwt"):
    """NeuroKit T-peak sample indices for one lead."""
    cleaned = nk.ecg_clean(sig, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    _, waves = nk.ecg_delineate(cleaned, rpeaks=info["ECG_R_Peaks"],
                                sampling_rate=fs, method=method)
    t = np.asarray(waves.get("ECG_T_Peaks", []), dtype=float)
    return t[np.isfinite(t)].astype(int)


def match(ref, det, tol):
    """One-to-one pairing of ref and det points within tol samples,
    closest pairs first. Returns (pairs, unmatched_ref, unmatched_det)."""
    cand = sorted((abs(d - r), i, j)
                  for i, r in enumerate(ref) for j, d in enumerate(det)
                  if abs(d - r) <= tol)
    used_r, used_d, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_r or j in used_d:
            continue
        used_r.add(i); used_d.add(j); pairs.append((i, j))
    miss_r = [i for i in range(len(ref)) if i not in used_r]
    extra_d = [j for j in range(len(det)) if j not in used_d]
    return pairs, miss_r, extra_d


def evaluate_t_peaks(files, method="dwt", lead="ii", tol_ms=150, edge_ms=100):
    """Score NeuroKit T peaks against LUDB cardiologist T peaks on one lead.

    files   output of get_ludb_files()
    tol_ms  max distance for a NeuroKit point to count as the same T peak
    edge_ms NeuroKit points this far outside the first/last annotated T peak
            are ignored (unannotated edge beats, not false positives)
    """
    beat_rows, rec_rows = [], []
    for rec_id, paths in files.items():
        base = paths["hea"][:-4]
        row = {"record": rec_id, "method": method, "status": "ok", "error": ""}
        try:
            rec = wfdb.rdrecord(base)
            fs = rec.fs
            ch = [n.lower() for n in rec.sig_name].index(lead.lower())
            ref = ref_t_peaks(wfdb.rdann(base, lead))
            if len(ref) == 0:
                row["status"] = "no reference T peaks"
                rec_rows.append(row)
                continue
            det = nk_t_peaks(rec.p_signal[:, ch], fs, method)
        except Exception as e:                       # NeuroKit can fail on odd records
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
            rec_rows.append(row)
            continue

        edge = edge_ms * fs / 1000
        det = det[(det >= ref.min() - edge) & (det <= ref.max() + edge)]
        pairs, miss, extra = match(ref, det, tol_ms * fs / 1000)

        for i, j in pairs:
            beat_rows.append({"record": rec_id, "outcome": "matched",
                              "ref_s": ref[i] / fs, "det_s": det[j] / fs,
                              "error_ms": (det[j] - ref[i]) * 1000 / fs})
        for i in miss:
            beat_rows.append({"record": rec_id, "outcome": "missed",
                              "ref_s": ref[i] / fs, "det_s": np.nan, "error_ms": np.nan})
        for j in extra:
            beat_rows.append({"record": rec_id, "outcome": "extra",
                              "ref_s": np.nan, "det_s": det[j] / fs, "error_ms": np.nan})

        row.update(n_ref=len(ref), n_det=len(det), n_matched=len(pairs),
                   n_missed=len(miss), n_extra=len(extra))
        rec_rows.append(row)

    beats = pd.DataFrame(beat_rows, columns=["record", "outcome", "ref_s", "det_s", "error_ms"])
    beats["sq_error_ms2"] = beats["error_ms"] ** 2
    beats.insert(1, "method", method)
    return beats, pd.DataFrame(rec_rows)


def summarize(beats, records):
    """One-row summary: detection rates and error statistics in ms."""
    m = beats.loc[beats["outcome"] == "matched", "error_ms"]
    n_ref = (beats["outcome"] != "extra").sum()
    n_det = (beats["outcome"] != "missed").sum()
    out = pd.Series({
        "records_scored": (records["status"] == "ok").sum(),
        "records_failed": (records["status"] == "failed").sum(),
        "ref_T_peaks": n_ref,
        "matched": len(m),
        "sensitivity_%": 100 * len(m) / n_ref if n_ref else np.nan,
        "PPV_%": 100 * len(m) / n_det if n_det else np.nan,
        "mean_error_ms (bias)": m.mean(),
        "SD_ms": m.std(),
        "MAE_ms": m.abs().mean(),
        "RMSE_ms": np.sqrt((m ** 2).mean()),
        "median_abs_error_ms": m.abs().median(),
        "within_20ms_%": 100 * (m.abs() <= 20).mean(),
        "within_40ms_%": 100 * (m.abs() <= 40).mean(),
    }).round(1)
    return out

from t_peak_eval import evaluate_t_peaks, summarize

files = get_ludb_files()
beats, records = evaluate_t_peaks(files, method="peak")
summarize(beats, records)


#=========== On the same beats
beats_dwt,  rec_dwt  = evaluate_t_peaks(files, method="dwt")
beats_pk,   rec_pk   = evaluate_t_peaks(files, method="peak")
beats_prom, rec_prom = evaluate_t_peaks(files, method="prominence")

key = ["record", "ref_s"]
matched = lambda b, name: b[b.outcome == "matched"][key + ["error_ms"]].rename(columns={"error_ms": name})

m = (matched(beats_dwt, "dwt")
     .merge(matched(beats_pk, "peak"), on=key)
     .merge(matched(beats_prom, "prominence"), on=key))
print(len(m), "beats matched by all three")
print(m[["dwt", "peak", "prominence"]].abs().describe().round(1))

# beats where all three methods fail (missed or >40 ms): likely genuinely hard T waves
fail = lambda b: b[(b.outcome == "missed") | (b.error_ms.abs() > 40)][key]
all_fail = fail(beats_dwt).merge(fail(beats_pk), on=key).merge(fail(beats_prom), on=key)
print(len(all_fail), "beats fail with all three methods")
all_fail.groupby("record").size().sort_values(ascending=False).head(10)

summarize(beats_prom, rec_prom)




#Other Points 
"""
t_peak_eval.py - score NeuroKit delineation against LUDB cardiologist
annotations, lead II. Default point: T peak; any point in POINTS can be scored
with point="..." (e.g. "ECG_R_Offsets").

For every record:
  1. Read lead II and the cardiologist lead-II annotations (".ii").
  2. Run NeuroKit (clean -> R peaks -> delineate) with the chosen method.
  3. Pair each cardiologist T peak with the nearest NeuroKit T peak within a
     tolerance window (one-to-one, closest pairs first).
  4. Error = NeuroKit time - cardiologist time, in ms
     (positive = NeuroKit is late, negative = early).

Only NeuroKit T peaks inside the annotated span are scored, because LUDB
cardiologists often skip the first and last beat of the 10 s strip.

Usage (notebook):
    from t_peak_eval import evaluate_t_peaks, summarize
    beats, records = evaluate_t_peaks(files, method="dwt")
    summarize(beats, records)

Returns
    beats   one row per cardiologist T peak (matched or missed) and per
            unmatched NeuroKit T peak (false positive)
    records one row per record: counts, status, error message if NeuroKit failed
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb


# Points the cardiologist annotations contain (NeuroKit key names).
# Q and S peaks are NOT annotated in LUDB/QTDB, so they cannot be scored.
POINTS = ["ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets",
          "ECG_R_Onsets", "ECG_R_Peaks", "ECG_R_Offsets",
          "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]


def ref_points(ann, point="ECG_T_Peaks"):
    """Cardiologist sample indices for one point type, e.g. "ECG_R_Offsets".
    Same parsing as ann_df_to_waves: '(' peak ')' around each wave."""
    wave, part = point.split("_")[1], point.split("_")[2]   # e.g. "R", "Offsets"
    peak_sym = {"P": "p", "R": "N", "T": "t"}[wave]
    sym, smp = ann.symbol, ann.sample
    out = []
    for i, s in enumerate(sym):
        if s != peak_sym:
            continue
        if part == "Peaks":
            out.append(smp[i])
        elif part == "Onsets" and i > 0 and sym[i - 1] == "(":
            out.append(smp[i - 1])
        elif part == "Offsets" and i + 1 < len(sym) and sym[i + 1] == ")":
            out.append(smp[i + 1])
    return np.array(out, dtype=int)


def nk_points(sig, fs, method="dwt", point="ECG_T_Peaks", **delineate_kwargs):
    """NeuroKit sample indices for one point type on one lead.
    Returns None if this method does not output that point."""
    cleaned = nk.ecg_clean(sig, sampling_rate=fs)
    if point == "ECG_R_Peaks":
        # R peaks come from ecg_peaks; here `method` is the R-peak detector
        # (e.g. "neurokit", "pantompkins1985", "hamilton2002", "elgendi2010")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=method)
        return np.asarray(info["ECG_R_Peaks"], dtype=int)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    _, waves = nk.ecg_delineate(cleaned, rpeaks=info["ECG_R_Peaks"],
                                sampling_rate=fs, method=method,
                                **delineate_kwargs)
    if point not in waves:
        return None
    t = np.asarray(waves[point], dtype=float)
    return t[np.isfinite(t)].astype(int)


def match(ref, det, tol):
    """One-to-one pairing of ref and det points within tol samples,
    closest pairs first. Returns (pairs, unmatched_ref, unmatched_det)."""
    cand = sorted((abs(d - r), i, j)
                  for i, r in enumerate(ref) for j, d in enumerate(det)
                  if abs(d - r) <= tol)
    used_r, used_d, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_r or j in used_d:
            continue
        used_r.add(i); used_d.add(j); pairs.append((i, j))
    miss_r = [i for i in range(len(ref)) if i not in used_r]
    extra_d = [j for j in range(len(det)) if j not in used_d]
    return pairs, miss_r, extra_d


def evaluate_t_peaks(files, method="dwt", lead="ii", tol_ms=150, edge_ms=100,
                     point="ECG_T_Peaks", **delineate_kwargs):
    """Score one NeuroKit point type against LUDB cardiologist points on one lead.
    Default point is the T peak; any name in POINTS works, e.g.
    point="ECG_R_Offsets" for QRS offset (J point).

    files   output of get_ludb_files()
    tol_ms  max distance for a NeuroKit point to count as the same point
    edge_ms NeuroKit points this far outside the first/last annotated T peak
            are ignored (unannotated edge beats, not false positives)
    **delineate_kwargs  passed to nk.ecg_delineate (e.g. physiological limits
            for method="prominence")
    """
    beat_rows, rec_rows = [], []
    for rec_id, paths in files.items():
        base = paths["hea"][:-4]
        row = {"record": rec_id, "method": method, "point": point, "status": "ok", "error": ""}
        try:
            rec = wfdb.rdrecord(base)
            fs = rec.fs
            ch = [n.lower() for n in rec.sig_name].index(lead.lower())
            ref = ref_points(wfdb.rdann(base, lead), point)
            if len(ref) == 0:
                row["status"] = "no reference points"
                rec_rows.append(row)
                continue
            det = nk_points(rec.p_signal[:, ch], fs, method, point, **delineate_kwargs)
            if det is None:
                row["status"] = f"{method} does not return {point}"
                rec_rows.append(row)
                continue
        except Exception as e:                       # NeuroKit can fail on odd records
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
            rec_rows.append(row)
            continue

        edge = edge_ms * fs / 1000
        det = det[(det >= ref.min() - edge) & (det <= ref.max() + edge)]
        pairs, miss, extra = match(ref, det, tol_ms * fs / 1000)

        for i, j in pairs:
            beat_rows.append({"record": rec_id, "outcome": "matched",
                              "ref_s": ref[i] / fs, "det_s": det[j] / fs,
                              "error_ms": (det[j] - ref[i]) * 1000 / fs})
        for i in miss:
            beat_rows.append({"record": rec_id, "outcome": "missed",
                              "ref_s": ref[i] / fs, "det_s": np.nan, "error_ms": np.nan})
        for j in extra:
            beat_rows.append({"record": rec_id, "outcome": "extra",
                              "ref_s": np.nan, "det_s": det[j] / fs, "error_ms": np.nan})

        row.update(n_ref=len(ref), n_det=len(det), n_matched=len(pairs),
                   n_missed=len(miss), n_extra=len(extra))
        rec_rows.append(row)

    beats = pd.DataFrame(beat_rows, columns=["record", "outcome", "ref_s", "det_s", "error_ms"])
    beats["sq_error_ms2"] = beats["error_ms"] ** 2
    beats.insert(1, "method", method)
    beats.insert(2, "point", point)
    return beats, pd.DataFrame(rec_rows)


def compare_methods(files, methods=("peak", "dwt", "cwt", "prominence"), **kwargs):
    """Run evaluate_t_peaks for several methods.
    Returns (summary table with one column per method, {method: (beats, records)})."""
    results = {m: evaluate_t_peaks(files, method=m, **kwargs) for m in methods}
    table = pd.DataFrame({m: summarize(*results[m]) for m in methods})
    return table, results


def summarize(beats, records):
    """One-row summary: detection rates and error statistics in ms."""
    m = beats.loc[beats["outcome"] == "matched", "error_ms"]
    n_ref = (beats["outcome"] != "extra").sum()
    n_det = (beats["outcome"] != "missed").sum()
    out = pd.Series({
        "records_scored": (records["status"] == "ok").sum(),
        "records_failed": (records["status"] == "failed").sum(),
        "ref_points": n_ref,
        "matched": len(m),
        "sensitivity_%": 100 * len(m) / n_ref if n_ref else np.nan,
        "PPV_%": 100 * len(m) / n_det if n_det else np.nan,
        "mean_error_ms (bias)": m.mean(),
        "SD_ms": m.std(),
        "MAE_ms": m.abs().mean(),
        "RMSE_ms": np.sqrt((m ** 2).mean()),
        "median_abs_error_ms": m.abs().median(),
        "within_20ms_%": 100 * (m.abs() <= 20).mean(),
        "within_40ms_%": 100 * (m.abs() <= 40).mean(),
    }).round(1)
    return out


evaluate_point = evaluate_t_peaks   # same function, clearer name for non-T points

evaluate_point = evaluate_t_peaks   # same function, clearer name for non-T points


point = "ECG_R_Offsets"     # QRS offset = J point
methods = ["dwt", "cwt", "peak", "prominence"]

results = {m: evaluate_t_peaks(files, method=m, point=point) for m in methods}
pd.DataFrame({m: summarize(*results[m]) for m in methods})



b, r = results["prominence"]
b = b[b.outcome == "matched"].copy()
recs = sorted(b.record.unique())
train, test = recs[::2], recs[1::2]

offset = b[b.record.isin(train)].error_ms.median()     # estimated on train half
corrected = b[b.record.isin(test)].error_ms - offset   # applied to test half
print(f"offset {offset:.1f} ms")
print("test within 20 ms: before", round(100 * (b[b.record.isin(test)].error_ms.abs() <= 20).mean(), 1),
    "after", round(100 * (corrected.abs() <= 20).mean(), 1))
print("test SD after:", round(corrected.std(), 1))


r_methods = ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010",
             "engzeemod2012", "kalidas2017", "rodrigues2021", "martinez2004",
             "nabian2018", "christov2004", "zong2003", "gamboa2008",
             "manikandan2012", "promac"]

r_results = {m: evaluate_t_peaks(files, method=m, point="ECG_R_Peaks") for m in r_methods}
r_table = pd.DataFrame({m: summarize(*r_results[m]) for m in r_methods}).T
show = r_table[["records_failed", "sensitivity_%", "PPV_%", "mean_error_ms (bias)", "SD_ms", "median_abs_error_ms"]]

fig, ax = plt.subplots(figsize=(12, 0.4 * len(show) + 1))
ax.axis("off")
tbl = ax.table(cellText=show.round(1).values, rowLabels=show.index, colLabels=show.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
plt.savefig("r_peak_table.png", dpi=200, bbox_inches="tight")
show



t_off_methods = ["dwt", "cwt", "peak", "prominence"]

t_off_results = {m: evaluate_t_peaks(files, method=m, point="ECG_T_Offsets") for m in t_off_methods}
t_off_table = pd.DataFrame({m: summarize(*t_off_results[m]) for m in t_off_methods}).T
show = t_off_table[["records_failed", "sensitivity_%", "PPV_%", "mean_error_ms (bias)", "SD_ms", "median_abs_error_ms"]]

fig, ax = plt.subplots(figsize=(12, 0.4 * len(show) + 1))
ax.axis("off")
tbl = ax.table(cellText=show.round(1).values, rowLabels=show.index, colLabels=show.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
plt.savefig("t_offset_table.png", dpi=200, bbox_inches="tight")
show



t_on_methods = ["dwt", "cwt", "peak", "prominence"]

t_on_results = {m: evaluate_t_peaks(files, method=m, point="ECG_T_Onsets") for m in t_on_methods}
t_on_table = pd.DataFrame({m: summarize(*t_on_results[m]) for m in t_on_methods}).T
show = t_on_table[["records_failed", "sensitivity_%", "PPV_%", "mean_error_ms (bias)", "SD_ms", "median_abs_error_ms"]]

fig, ax = plt.subplots(figsize=(12, 0.4 * len(show) + 1))
ax.axis("off")
tbl = ax.table(cellText=show.round(1).values, rowLabels=show.index, colLabels=show.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
plt.savefig("t_onset_table.png", dpi=200, bbox_inches="tight")
show


qrs_on_methods = ["dwt", "cwt", "peak", "prominence"]

qrs_on_results = {m: evaluate_t_peaks(files, method=m, point="ECG_R_Onsets") for m in qrs_on_methods}
qrs_on_table = pd.DataFrame({m: summarize(*qrs_on_results[m]) for m in qrs_on_methods}).T
show = qrs_on_table[["records_failed", "sensitivity_%", "PPV_%", "mean_error_ms (bias)", "SD_ms", "median_abs_error_ms"]]

fig, ax = plt.subplots(figsize=(12, 0.4 * len(show) + 1))
ax.axis("off")
tbl = ax.table(cellText=show.round(1).values, rowLabels=show.index, colLabels=show.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
plt.savefig("qrs_onset_table.png", dpi=200, bbox_inches="tight")
show



best = {
    # point name:         (results,                                                          method,       CSE tolerance ms)
    "R peak":            (r_results["neurokit"],                                              "neurokit",   None),
    "QRS onset":         (qrs_on_results["prominence"],                                       "prominence", 6.5),
    "J point (QRS off)": (evaluate_t_peaks(files, method="prominence", point="ECG_R_Offsets"), "prominence", 11.6),
    "T onset":           (t_on_results["prominence"],                                         "prominence", None),
    "T peak":            (evaluate_t_peaks(files, method="prominence", point="ECG_T_Peaks"),   "prominence", None),
    "T offset":          (t_off_results["prominence"],                                        "prominence", 30.6),
}

rows = {}
for point, (res, method, tol) in best.items():
    s = summarize(*res)
    rows[point] = {
        "method": method,
        "sensitivity_%": s["sensitivity_%"],
        "PPV_%": s["PPV_%"],
        "bias_ms": s["mean_error_ms (bias)"],
        "SD_ms": s["SD_ms"],
        "median_abs_ms": s["median_abs_error_ms"],
        "CSE_tol_ms (2σ)": tol if tol else "none",
        "SD / tol": round(s["SD_ms"] / tol, 2) if tol else "n/a",
        "meets CSE": ("yes" if s["SD_ms"] <= tol else "no") if tol else "n/a",
    }
summary_all = pd.DataFrame(rows).T

fig, ax = plt.subplots(figsize=(15, 0.45 * len(summary_all) + 1.2))
ax.axis("off")
tbl = ax.table(cellText=summary_all.values, rowLabels=summary_all.index, colLabels=summary_all.columns,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
ax.set_title("LUDB lead II vs cardiologist annotations. CSE tolerances: CSE Working Party, Eur Heart J 1985;6:815-825",
             fontsize=9, loc="left")
plt.savefig("all_points_table.png", dpi=200, bbox_inches="tight")
summary_all
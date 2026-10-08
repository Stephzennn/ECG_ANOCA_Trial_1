import re, json, numpy as np, pandas as pd, torch, wfdb
from pathlib import Path
from net1d import Net1D
from util import filter_bandpass
import neurokit2 as nk

folder_path = Path("D:/ScienceDB_Data")
words = ["baseline", "data", "rest", "500hz"]
words_Disease = ["disease", "json"]
CKPT12 = "./checkpoint/12_lead_ECGFounder.pth"
CKPT1 = "./checkpoint/1_lead_ECGFounder.pth"          # use the actual downloaded file name
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]   # model order
tasks = [l.strip() for l in open("./tasks.txt")]
NAMES = {"normal_ecg": "NORMAL ECG", "abnormal_ecg": "ABNORMAL ECG", "nsr": "NORMAL SINUS RHYTHM",
         "af": "ATRIAL FIBRILLATION", "pvc": "PREMATURE VENTRICULAR COMPLEXES",
         "pac": "PREMATURE ATRIAL COMPLEXES", "sinus_arrhythmia": "WITH SINUS ARRHYTHMIA",
         "prolonged_qt": "PROLONGED QT", "1avb": "WITH 1ST DEGREE AV BLOCK", "short_pr": "WITH SHORT PR",
         "nonspecific_t": "NONSPECIFIC T WAVE ABNORMALITY",
         "nonspecific_st_t": "NONSPECIFIC ST AND T WAVE ABNORMALITY",
         "low_voltage": "LOW VOLTAGE QRS", "lad": "LEFT AXIS DEVIATION", "paced": "VENTRICULAR-PACED RHYTHM"}
KEY = {k: tasks.index(v) for k, v in NAMES.items()}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def getBeatData(csvFile, leadNumber=2, method="prominence", ecgIn=False, block_s=10):
    """Same inputs and output columns as before, but each row is now the
    median beat of a `block_s`-second block (default 10 s, like a standard
    clinical ECG) instead of a single raw beat."""
    fs = 500
 
    # ---- load: same as before (ecgIn used if given) ----
    if ecgIn is not False:
        ecg = np.asarray(ecgIn, dtype=float).squeeze()
    else:
        ecg_signal = pd.read_csv(csvFile, header=None, nrows=12, dtype="float32").to_numpy().T
        ecg = np.asarray(ecg_signal[:, leadNumber - 1], dtype=float)
 
    cleaned = np.asarray(nk.ecg_clean(ecg, sampling_rate=fs)).squeeze()
    _, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=fs,correct_artifacts=True)
    rpeaks = np.asarray(rpeak_info["ECG_R_Peaks"])
 
    # ---- 1. keep only normal beats: RR before AND after within 20% of median RR ----
    rr = np.diff(rpeaks)
    med_rr = np.median(rr)
    ok = np.zeros(len(rpeaks), dtype=bool)
    ok[1:-1] = (np.abs(rr[:-1] - med_rr) < 0.2 * med_rr) & (np.abs(rr[1:] - med_rr) < 0.2 * med_rr)
 
    rows = []
    for start in range(0, len(cleaned), block_s * fs):
        idx = np.where(ok & (rpeaks >= start) & (rpeaks < start + block_s * fs))[0]
        if len(idx) < 5:                                  # too few clean beats in this block
            continue
 
        # ---- 2. median beat: one RR long, R at 40% of the way in ----
        L = int(np.median(rpeaks[idx] - rpeaks[idx - 1]))
        pre = int(0.4 * L)
        beats = [cleaned[r - pre: r - pre + L] for r in rpeaks[idx]
                 if r - pre >= 0 and r - pre + L <= len(cleaned)]
        if len(beats) < 5:
            continue
        median_beat = np.median(np.vstack(beats), axis=0)
 
        # ---- 3. delineate: tile 5 copies so the delineator sees a rhythm, keep the middle ----
        sig = np.tile(median_beat, 5)
        tr = pre + L * np.arange(5)
        try:
            _, waves = nk.ecg_delineate(
                sig, rpeaks=tr, sampling_rate=fs, method=method
            )
        except (IndexError, ValueError):
            continue
 
        g = lambda k, i=2: waves[k][i]
        qrs_on, qrs_off = g("ECG_R_Onsets"), g("ECG_R_Offsets")
        t_on, t_peak, t_off = g("ECG_T_Onsets"), g("ECG_T_Peaks"), g("ECG_T_Offsets")
        s_peak, p_on, p_off = g("ECG_S_Peaks"), g("ECG_P_Onsets"), g("ECG_P_Offsets")
        qrs_on_next = g("ECG_R_Onsets", 3)
        r_peak = tr[2]
 
        # ---- 4. same feature calculation as before, on the middle median beat ----
        points = [qrs_on, qrs_off, t_on, t_peak, t_off, r_peak]
        if not np.all(np.isfinite(points)):
            continue
        qrs_on, qrs_off, t_on, t_peak, t_off, r_peak = map(int, points)
        if not (qrs_on <= r_peak <= qrs_off <= t_on < t_peak < t_off):
            continue
 
        if np.isfinite(qrs_on_next) and int(qrs_on_next) > t_off:
            x = [qrs_on, int(qrs_on_next)]
            y = [sig[qrs_on], sig[int(qrs_on_next)]]
        else:
            x = [qrs_on, qrs_on + 1]
            y = [sig[qrs_on], sig[qrs_on]]
        B = lambda i: np.interp(i, x, y)
 
        if np.isfinite(s_peak) and r_peak < int(s_peak) <= qrs_off:
            s_peak = int(s_peak)
            s_term = sig[s_peak] - B(s_peak)
        else:
            s_term = 0.0
 
        numerator = sig[t_peak] - B(t_peak)
        denominator = (sig[r_peak] - B(r_peak)) + s_term
        ratio = numerator / denominator if abs(denominator) > 1e-6 else np.nan
        pr_ms = ((qrs_on - int(p_on)) * 1000 / fs
                 if np.isfinite(p_on) and int(p_on) < qrs_on else np.nan)
        p_dur_ms = ((int(p_off) - int(p_on)) * 1000 / fs
                    if np.isfinite(p_on) and np.isfinite(p_off) and int(p_on) < int(p_off) <= qrs_on
                    else np.nan)
 
        qt_ms = (t_off - qrs_on) * 1000 / fs
        rr_s = L / fs                                   # median RR of this block, in seconds

        rows.append({
            "QRS_offset_to_T_onset_ms": (t_on - qrs_off) * 1000 / fs,
            "QRS_onset_to_QRS_offset_ms": (qrs_off - qrs_on) * 1000 / fs,
            "PR_interval_ms": pr_ms,
            "QT_interval_ms": qt_ms,
            "QTcF_ms": qt_ms / rr_s ** (1/3),           # Fridericia: QT / RR^(1/3)
            "T_wave_duration_ms": (t_off - t_on) * 1000 / fs,
            "T_peak_signed_amplitude": numerator,
            "QRS_balance": denominator,
            "T_over_QRS_balance": ratio,
            "P_duration_ms": p_dur_ms,
        })
 
    # ---- 5. heart rate / HRV from the whole recording, as before ----
    beat_data = pd.DataFrame(rows)
    #beat_data["HR_bpm"] = np.mean(nk.ecg_rate(rpeak_info, sampling_rate=fs))
    #tt = nk.hrv_time(rpeak_info, sampling_rate=fs)

    beat_data["HR_bpm"] = np.mean(nk.ecg_rate(rpeaks, sampling_rate=fs))
    tt = nk.hrv_time(rpeaks, sampling_rate=fs)
    beat_data["HRV_RMSSD"] = tt["HRV_RMSSD"].iloc[0]
    beat_data["HRV_MedianNN"] = tt["HRV_MedianNN"].iloc[0]
    return beat_data



def load_model(ckpt, n_in):
    m = Net1D(in_channels=n_in, base_filters=64, ratio=1, filter_list=[64,160,160,400,400,1024,1024],
              m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16, stride=2, groups_width=16,
              use_bn=False, use_do=False, n_classes=150, return_features=True)
    log = m.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"], strict=False)
    assert not log.missing_keys, log.missing_keys        # stop if weights didn't load
    return m.to(dev).eval()


def run(model, x):                                      # same preprocessing as ScienceDB_Dataset
    x = filter_bandpass(x, 500)
    x = torch.FloatTensor((x - x.mean()) / (x.std() + 1e-8))[None].to(dev)
    with torch.no_grad():
        logits, emb = model(x)
    return logits[0].cpu().numpy(), emb[0].cpu().numpy()


m12, m1 = load_model(CKPT12, 12), load_model(CKPT1, 1)
rows, emb12, emb1 = [], [], []

for file_path in folder_path.iterdir():
    if file_path.name[0:3] != "sub":
        continue
    pid = file_path.name[4:]

    disease_file = next((f for f in file_path.iterdir()
                         if all(w in f.name.lower() for w in words_Disease)), None)
    if disease_file is None:
        continue
    with open(disease_file) as f:
        if int(json.load(f)["ANOCA"]) != 0:
            continue

    for hea in file_path.iterdir():
        if hea.suffix != ".hea" or not all(w in hea.name.lower() for w in words):
            continue

        sig, meta = wfdb.rdsamp(str(hea.with_suffix("")))
        names = [n.upper() for n in meta["sig_name"]]
        x12 = sig[:, [names.index(l) for l in LEADS]].T          # 12 x 5000, chosen by lead name
        lead2 = x12[1]
        lead1 = x12[names.index("I")]

        k = int(re.findall(r"\d+", hea.stem)[-1])                # slice number from file name
        row = {"pid": pid, "slice_id": hea.stem, "start_s": k * 10}

        try:                                                     # our features, lead II
            bd = getBeatData(None, ecgIn=lead2, block_s=10)
            if len(bd):
                row.update(bd.iloc[0].to_dict())
        except Exception:
            pass

        l12, e12 = run(m12, x12)                                 # ECGFounder 12-lead
        l1, e1 = run(m1, lead2[None]) 
        lI, eI = run(m1, lead1[None])                           # ECGFounder single-lead (lead II)
        row.update({f"lI_{n}": lI[i] for n, i in KEY.items()})
        
        row.update({f"l12_{n}": l12[i] for n, i in KEY.items()})
        row.update({f"l1_{n}": l1[i] for n, i in KEY.items()})
        row.update({f"lI_{n}": lI[i] for n, i in KEY.items()})
        rows.append(row); emb12.append(e12); emb1.append(e1)
      
windows = pd.DataFrame(rows)

windows
windows.to_csv("windows.csv", index=False)
np.save("emb12.npy", np.array(emb12)); np.save("emb1.npy", np.array(emb1))
print(windows.shape, windows["pid"].nunique(), "participants")



"""
==========================================================================================
"""

import re, json, numpy as np, pandas as pd, torch, wfdb
from pathlib import Path
from net1d import Net1D
from util import filter_bandpass
#from getBeatData_median import getBeatData           # or wherever your median-beat getBeatData lives

ecg_path = "D:/ScienceDB_Data"                       # same ecg_path as ScienceDB_Dataset
csv_path = "./csv/ScienceDB_label.csv"
#words = ["rest"]                                      # must appear in filename_hr (adjust to your names)
CKPT12 = "./checkpoint/12_lead_ECGFounder.pth"
CKPT1 = "./checkpoint/1_lead_ECGFounder.pth"
BANDPASS = False                                      # False = same as your earlier ScienceDB run
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tasks = [l.strip() for l in open("./tasks.txt")]
NAMES = {"normal_ecg": "NORMAL ECG", "abnormal_ecg": "ABNORMAL ECG", "nsr": "NORMAL SINUS RHYTHM",
         "af": "ATRIAL FIBRILLATION", "pvc": "PREMATURE VENTRICULAR COMPLEXES",
         "pac": "PREMATURE ATRIAL COMPLEXES", "sinus_arrhythmia": "WITH SINUS ARRHYTHMIA",
         "prolonged_qt": "PROLONGED QT", "1avb": "WITH 1ST DEGREE AV BLOCK", "short_pr": "WITH SHORT PR",
         "nonspecific_t": "NONSPECIFIC T WAVE ABNORMALITY",
         "nonspecific_st_t": "NONSPECIFIC ST AND T WAVE ABNORMALITY",
         "low_voltage": "LOW VOLTAGE QRS", "lad": "LEFT AXIS DEVIATION", "paced": "VENTRICULAR-PACED RHYTHM"}
KEY = {k: tasks.index(v) for k, v in NAMES.items()}

NAMES.update({"marked_sinus_arrhythmia": "WITH MARKED SINUS ARRHYTHMIA",
              "psvc": "PREMATURE SUPRAVENTRICULAR COMPLEXES",
              "irregular": "WITH UNDETERMINED RHYTHM IRREGULARITY",
              "sinus_brady": "SINUS BRADYCARDIA", "sinus_tachy": "SINUS TACHYCARDIA"})
KEY = {k: tasks.index(v) for k, v in NAMES.items()}


NAMES.update({"rbbb": "RIGHT BUNDLE BRANCH BLOCK", "lbbb": "LEFT BUNDLE BRANCH BLOCK",
              "irbbb": "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",
              "ivcd": "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
              "qrs_widening": "WITH QRS WIDENING"})
KEY = {k: tasks.index(v) for k, v in NAMES.items()}



def load_model(ckpt, n_in):
    m = Net1D(in_channels=n_in, base_filters=64, ratio=1, filter_list=[64,160,160,400,400,1024,1024],
              m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16, stride=2, groups_width=16,
              use_bn=False, use_do=False, n_classes=150, return_features=True)
    log = m.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"], strict=False)
    assert not log.missing_keys, log.missing_keys
    return m.to(dev).eval()


def run(model, x):                                    # leads x 5000
    if BANDPASS:
        x = filter_bandpass(x, 500)
    x = torch.FloatTensor((x - x.mean()) / (x.std() + 1e-8))[None].to(dev)
    with torch.no_grad():
        logits, emb = model(x)
    return logits[0].cpu().numpy(), emb[0].cpu().numpy()


anoca_cache = {}
def anoca_negative(folder):                           # disease JSON in the slice's folder
    if folder not in anoca_cache:
        f = next((f for f in folder.glob("*.json") if "disease" in f.name.lower()), None)
        anoca_cache[folder] = f is not None and int(json.load(open(f))["ANOCA"]) == 0
    return anoca_cache[folder]


df = pd.read_csv(csv_path).dropna(subset=["filename_hr"]).drop_duplicates("filename_hr")
print(len(df), "slices, e.g.", df["filename_hr"].head(2).tolist())

m12, m1 = load_model(CKPT12, 12), load_model(CKPT1, 1)
rows, emb12, emb1 = [], [], []

for _, r in df.iterrows():
    rec = Path(ecg_path + r["filename_hr"])
    if not anoca_negative(rec.parents[1]):                        # sub_001/ holds the disease JSON
        continue
    sig, meta = wfdb.rdsamp(str(rec))
    x12 = sig.T                                                   # file lead order, as before
    lead2 = x12[[n.upper() for n in meta["sig_name"]].index("II")]
    names = [n.upper() for n in meta["sig_name"]]
    lead1 = x12[names.index("I")]
    start_s = int(rec.stem.split("_")[0])                         # "10_10seconds" -> 10
    row = {"pid": r["patient_id"], "slice_id": rec.stem, "start_s": start_s}

    try:
        bd = getBeatData(None, ecgIn=lead2, block_s=10)
        if len(bd):
            row.update(bd.iloc[0].to_dict())
    except Exception:
        pass
    try:
        bdI = getBeatData(None, ecgIn=lead1, block_s=10)
        if len(bdI):
            row.update({f"{k}_leadI": v for k, v in bdI.iloc[0].items()})
    except Exception:
        pass
    l12, e12 = run(m12, x12)                                 # ECGFounder 12-lead
    l1, e1 = run(m1, lead2[None]) 
    lI, eI = run(m1, lead1[None])                           # ECGFounder single-lead (lead II)
    row.update({f"lI_{n}": lI[i] for n, i in KEY.items()})
    
    row.update({f"l12_{n}": l12[i] for n, i in KEY.items()})
    row.update({f"l1_{n}": l1[i] for n, i in KEY.items()})
    row.update({f"lI_{n}": lI[i] for n, i in KEY.items()})
    rows.append(row); emb12.append(e12); emb1.append(e1)
    
windows = pd.DataFrame(rows)
windows.keys()

import matplotlib.pyplot as plt

def plot_windows(x, y, xlabel, ylabel, title):
    w = windows.dropna(subset=[x, y])
    codes = w["pid"].astype("category").cat.codes                 # one colour per participant
    plt.figure(figsize=(8, 5.5))
    plt.scatter(w[x], w[y], c=codes, cmap="nipy_spectral", s=10, alpha=.6)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}\nEach dot = one 10-second window, colour = participant "
              f"(n = {len(w)} windows, {w['pid'].nunique()} participants)")
    plt.show()



x = "T_over_QRS_balance"
w = windows.dropna(subset=[x])

# window level: outside 1.5×IQR of all windows
q1, q3 = w[x].quantile([.25, .75]); iqr = q3 - q1
out_win = w[(w[x] < q1 - 1.5 * iqr) | (w[x] > q3 + 1.5 * iqr)]
print(f"{len(out_win)} outlier windows")
print(out_win[["pid", "slice_id", x, "T_peak_signed_amplitude", "QRS_balance"]].sort_values(x))

# participant level: variance of T/QRS across that person's windows
v = w.groupby("pid")[x].agg(["median", "var", "count"])
q1, q3 = v["var"].quantile([.25, .75])
print(v[v["var"] > q3 + 1.5 * (q3 - q1)].sort_values("var", ascending=False))


w = windows[windows["pid"] == 77].sort_values("start_s")
t = np.arange(len(w)) * 10                                   # seconds from start of recording
x = "T_over_QRS_balance"

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, w[x], "o-", ms=4)
ax.set_xlabel("Time (s)"); ax.set_ylabel("T/QRS balance")
ax.set_title(f"Participant 77: T/QRS balance per 10-second window (n = {len(w)} windows)")
plt.tight_layout(); plt.show()





x = "T_over_QRS_balance"
v = windows.dropna(subset=[x]).groupby("pid")[x].agg(["median", "var"])

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(v["var"], v["median"], s=25)
for pid, r in v.iterrows():
    ax.annotate(str(pid), (r["var"], r["median"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Variance of T/QRS balance across 10-second windows")
ax.set_ylabel("Median T/QRS balance across 10-second windows")
ax.set_title(f"T/QRS balance per participant: variance vs median (n = {len(v)} participants)")
plt.tight_layout(); plt.show()


plot_windows("l12_normal_ecg", "l1_normal_ecg",
             "12-lead ECGFounder logit: normal ECG", "Single-lead (lead II) ECGFounder logit: normal ECG",
             "12-lead vs single-lead ECGFounder agreement")

plot_windows("QTcF_ms", "l1_prolonged_qt",
             "QTcF, lead I (ms)", "Single-lead ECGFounder logit: prolonged QT",
             "Measured QTcF vs ECGFounder prolonged-QT output")

# Lead II measurement, lead II model input
plot_windows("QT_interval_ms", "l1_prolonged_qt",
             "QT interval, lead II (ms)", "Single-lead ECGFounder logit (lead II): prolonged QT",
             "Measured QT vs ECGFounder prolonged-QT output, lead II")


plot_windows("QTcF_ms", "lI_prolonged_qt",
             "QTcF, measured on lead II (ms)", "Single-lead ECGFounder logit (lead I input): prolonged QT",
             "Measured QTcF vs ECGFounder prolonged-QT output, lead I model input")

plot_windows("T_over_QRS_balance", "l1_nonspecific_t",
             "T/QRS balance, lead II", "Single-lead ECGFounder logit: nonspecific T wave abnormality",
             "Measured T/QRS balance vs ECGFounder nonspecific T-wave output")

plot_windows("T_over_QRS_balance", "lI_nonspecific_t",
             "T/QRS balance, measured on lead II", "Single-lead ECGFounder logit (lead I input): nonspecific T wave abnormality",
             "Measured T/QRS balance vs ECGFounder nonspecific T-wave output, lead I model input")

plot_windows("PR_interval_ms", "l1_short_pr", "PR interval, lead II (ms)",
             "Single-lead ECGFounder logit: short PR", "Measured PR interval vs ECGFounder short-PR output")


plot_windows("QRS_balance", "l1_low_voltage", "Net QRS amplitude, lead II (R − |S|)",
             "Single-lead ECGFounder logit: low voltage QRS", "Measured QRS amplitude vs ECGFounder low-voltage output")


plot_windows("HRV_RMSSD", "l1_sinus_arrhythmia", "RMSSD within window (ms)",
             "Single-lead ECGFounder logit: sinus arrhythmia", "Beat-to-beat variability vs ECGFounder sinus-arrhythmia output")

windows.to_csv("windows.csv", index=False)
np.save("emb12.npy", np.array(emb12)); np.save("emb1.npy", np.array(emb1))
print(windows.shape, windows["pid"].nunique(), "participants")



def plot_windows(x, y, xlabel, ylabel, title, prob_label=None,logx=False):
    w = windows.dropna(subset=[x, y])
    codes = w["pid"].astype("category").cat.codes
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if logx: ax.set_xscale("log")
    ax.scatter(w[x], w[y], c=codes, cmap="nipy_spectral", s=10, alpha=.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nEach dot = one 10-second window, colour = participant "
                 f"(n = {len(w)} windows, {w['pid'].nunique()} participants)")
    if prob_label:                                               # probability axis on the right
        sec = ax.secondary_yaxis("right", functions=(lambda l: 1/(1+np.exp(-l)), lambda p: np.log(p/(1-p))))
        ticks = [t for t in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95]
                 if np.log(t/(1-t)) >= w[y].min() and np.log(t/(1-t)) <= w[y].max()]
        sec.set_yticks(ticks); sec.set_yticklabels([f"{t:.1%}".replace(".0%", "%") for t in ticks])
        sec.set_ylabel(prob_label)
    plt.tight_layout(); plt.show()

plot_windows("PR_interval_ms", "l1_short_pr",
             "PR interval, lead II (ms)", "Single-lead ECGFounder logit (lead II): short PR",
             "Measured PR interval vs ECGFounder short-PR output, lead II",
             prob_label="Model probability of short PR")


# HRV vs sinus arrhythmia: the main HRV agreement plot
plot_windows("HRV_RMSSD", "l1_sinus_arrhythmia", "RMSSD within window, lead II (ms, log scale)",
             "Single-lead ECGFounder logit (lead II): sinus arrhythmia",
             "Beat-to-beat variability vs ECGFounder sinus-arrhythmia output, lead II",
             prob_label="Model probability of sinus arrhythmia", logx=True)


# HRV vs ectopy/irregularity: the most irregular non-sinus label per window
windows["l1_irregular_any"] = windows[["l1_pvc", "l1_pac", "l1_psvc", "l1_af", "l1_irregular"]].max(axis=1)
plot_windows("HRV_RMSSD", "l1_irregular_any", "RMSSD within window, lead II (ms, log scale)",
             "Single-lead ECGFounder logit (lead II): highest of PVC / PAC / PSVC / AF / irregular",
             "Beat-to-beat variability vs ECGFounder ectopy/irregularity output, lead II",
             prob_label="Model probability (highest ectopy label)", logx=True)


windows["l1_wide_qrs"] = windows[["l1_rbbb", "l1_lbbb", "l1_irbbb", "l1_ivcd", "l1_qrs_widening"]].max(axis=1)
plot_windows("QRS_onset_to_QRS_offset_ms", "l1_wide_qrs", "QRS duration, lead II (ms)",
             "Single-lead ECGFounder logit (lead II): highest of RBBB / LBBB / IRBBB / IVCD / QRS widening",
             "Measured QRS duration vs ECGFounder wide-QRS output, lead II",
             prob_label="Model probability (highest wide-QRS label)")


set(windows['pid'])

for x in windows:
    print(x)

windows[windows['HRV_RMSSD'] >= 200]

rmssd_by_pid = windows.groupby("pid")["HRV_RMSSD"].median().sort_values(ascending=False)
rmssd_by_pid.describe()

p = windows.groupby("pid").agg(rmssd=("HRV_RMSSD", "median"), hr=("HR_bpm", "median")).dropna()

plt.figure(figsize=(8, 5.5))
plt.scatter(p["rmssd"], p["hr"], s=25, alpha=.8)
for pid, r in p.iterrows():
    plt.annotate(str(pid), (r["rmssd"], r["hr"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
plt.xlabel("RMSSD, median of 10-second windows (ms)")
plt.ylabel("Heart rate, median of 10-second windows (bpm)")
plt.title(f"Heart rate vs. heart rate variability, lead II (n = {len(p)})\n"
          "Each point = one participant, median over 36 ten-second windows (6-minute resting ECG)")
plt.tight_layout(); plt.show()


p = windows.groupby("pid")["T_over_QRS_balance"].agg(median="median", var="var").dropna()

plt.figure(figsize=(8, 5))
plt.scatter(p["var"], p["median"], s=25, alpha=.8)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.xlabel("Variance of T/QRS balance across 10-second windows")
plt.ylabel("Median T/QRS balance")
plt.title(f"T/QRS balance per participant, lead II (n = {len(p)})\n"
          "Each point = one participant, 10-second .hea windows (6-minute resting ECG)")
plt.show()


w = windows[windows["pid"].astype(int) == 64].sort_values("start_s").copy()
w["block30"] = np.arange(len(w)) // 3                    # 3 consecutive 10-s slices = 30 s
g = w.groupby("block30").agg(rmssd=("HRV_RMSSD", lambda s: np.sqrt(np.mean(s**2))),
                             rr=("HRV_MedianNN", "median"))

plt.figure(figsize=(8, 5))
plt.scatter(g["rmssd"], g["rr"], s=30)
plt.xlabel("Root mean square of successive differences (RMSSD) within window (ms)")
plt.ylabel("Median RR interval within window (ms)")
plt.title(f"Heart rate variability per 30-second window (n = {len(g)})\nParticipant 64, resting ECG, lead II")
plt.show()



plot_windows("QTcF_ms", "l1_prolonged_qt",
             "QTcF, lead II (ms)", "Single-lead ECGFounder logit (lead II): prolonged QT",
             "Measured QTcF vs ECGFounder prolonged-QT output, lead II",
             prob_label="Model probability of prolonged QT")


w = windows[windows["pid"].astype(int) == 64].sort_values("start_s").copy()
w["block30"] = np.arange(len(w)) // 3                    # 3 consecutive 10-s slices = 30 s
g = w.groupby("block30").agg(rmssd=("HRV_RMSSD", lambda s: np.sqrt(np.mean(s**2))),
                             rr=("HRV_MedianNN", "median"))

plt.figure(figsize=(8, 5))
plt.scatter(g["rmssd"], 60000 / g["rr"], s=30)
plt.xlabel("Root mean square of successive differences (RMSSD) within window (ms)")
plt.ylabel("Median heart rate within window (bpm)")
plt.title(f"Heart rate variability per 30-second window (n = {len(g)})\nParticipant 64, resting ECG, lead II")
plt.show()



def plot_feature_hea(col, label, units=""):
    p = windows.groupby("pid")[col].agg(median="median", var="var").dropna()
    plt.figure(figsize=(8, 5))
    plt.scatter(p["var"], p["median"], s=25, alpha=.8)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel(f"Variance of {label} across 10-second windows{f' ({units}²)' if units else ''}")
    plt.ylabel(f"Median {label}{f' ({units})' if units else ''}")
    plt.title(f"{label} per participant, lead II (n = {len(p)})\n"
              "Each point = one participant, 10-second windows (6-minute resting ECG)")
    plt.show()
plot_feature_hea("PR_interval_ms", "PR interval", "ms")


plot_windows("T_over_QRS_balance", "l1_nonspecific_t",
             "T/QRS balance, lead II", "ECGFounder logit (lead II): nonspecific T wave abnormality",
             "Measured T/QRS balance vs ECGFounder nonspecific T-wave output, lead II",
             prob_label="Model probability of nonspecific T wave abnormality")



from scipy.stats import spearmanr

x, y = "T_over_QRS_balance_leadI", "lI_nonspecific_t"
w = windows.dropna(subset=[x, y])

# 1. All windows (descriptive: windows from the same person aren't independent)
r_win, p_win = spearmanr(w[x], w[y])

# 2. One point per participant (medians, n = 34): independent, so the p-value is valid
m = w.groupby("pid")[[x, y]].median()
r_pid, p_pid = spearmanr(m[x], m[y])

print(f"Window level:      rho = {r_win:.2f}  (n = {len(w)} windows)")
print(f"Participant level: rho = {r_pid:.2f}, p = {p_pid:.3g}  (n = {len(m)} participants)")

x, y = "PR_interval_leadI", "lI_short_pr"
w = windows.dropna(subset=[x, y])
r_win, _ = spearmanr(w[x], w[y])
m = w.groupby("pid")[[x, y]].median()
r_pid, p_pid = spearmanr(m[x], m[y])
print(f"Window level: rho = {r_win:.2f} (n = {len(w)} windows)")
print(f"Participant level: rho = {r_pid:.2f}, p = {p_pid:.2g} (n = {len(m)} participants)")




x, y = "T_over_QRS_balance", "l1_nonspecific_st_t"
w = windows.dropna(subset=[x, y])
r_win, _ = spearmanr(w[x], w[y])
m = w.groupby("pid")[[x, y]].median()
r_pid, p_pid = spearmanr(m[x], m[y])
print(f"Window level: rho = {r_win:.2f} (n = {len(w)} windows)")
print(f"Participant level: rho = {r_pid:.2f}, p = {p_pid:.2g} (n = {len(m)} participants)")

plot_windows(x, y, "T/QRS balance (lead II)", "ECGFounder logit (lead II)",
             "T/QRS balance vs ECGFounder nonspecific ST and T wave abnormality",
             prob_label="Probability")
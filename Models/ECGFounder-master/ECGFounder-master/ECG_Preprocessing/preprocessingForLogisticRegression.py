#

import os
import ishneholterlib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

import wfdb
import pandas as pd
import neurokit2 as nk

import numpy as np
import pandas as pd
import neurokit2 as nk
#Open a subject's 500 sampling rate ECG file Rest 6 minutes total

# This is the csv path for subject 1's full csv
#csv_path = r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\data\ScienceDB_Data\sub_001\sub_001_baseline-data_rest-ecg-500hz.csv"

from pathlib import Path

# Here start iterating

# Here we are loading one patient
csv_path = Path(
    "C:/Users/Estif/Downloads/Langone/ANOCA/"
    "ECG_ANOCA_Trial_1/Models/ECGFounder-master/"
    "ECGFounder-master/data/ScienceDB_Data/sub_038/"
    "sub_038_baseline-data_rest-ecg-500hz.csv"
)

print(csv_path.exists())  # should print True


ecg_signal = pd.read_csv(
                        csv_path,
                        header=None,
                        nrows=12,
                        dtype="float32"
                    ).to_numpy().T


ecg_signal.shape




ecg = np.asarray(ecg_signal[:, 0], dtype=float)

#cleaned = ecg
cleaned = nk.ecg_clean(ecg, sampling_rate=500)

"""
This is a check on an individual patient


"""

import matplotlib.pyplot as plt
fs = 500
start = 50000            # starting sample (5000 / 500 = 10 s into the recording)
dur   = 5               # seconds to show

N = dur * fs
seg = slice(start, start + N)
t = np.arange(start, start + N) / fs     # x-axis in real recording time

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax[0].plot(t, ecg[seg], linewidth=0.8, color="#888888")
ax[0].set_title("Raw")
ax[1].plot(t, cleaned[seg], linewidth=0.8, color="#1F3A5F")
ax[1].set_title("Cleaned")
ax[1].set_xlabel("Time (s)")

for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()

_, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]

pre, post = int(0.25*fs), int(0.60*fs)
beats = np.array([cleaned[r-pre:r+post] for r in rpeaks
                  if r-pre >= 0 and r+post < len(cleaned)])
median_beat = np.median(beats, axis=0)
t_ms = (np.arange(-pre, post)) * 1000 / fs

plt.figure(figsize=(10, 5))
for b in beats[::10]:                      # a sample of individual beats
    plt.plot(t_ms, b, color="#BBBBBB", linewidth=0.4)
plt.plot(t_ms, median_beat, color="#B03A2E", linewidth=2.5, label=f"median of {len(beats)} beats")
plt.axvline(0, color="#1F3A5F", linestyle="--", linewidth=1)
plt.xlabel("Time from R peak (ms)"); plt.legend(); plt.grid(alpha=0.3)
plt.show()


# Delineate waves
signals, waves = nk.ecg_delineate(
    cleaned,
    rpeaks=rpeak_info["ECG_R_Peaks"],
    sampling_rate=500,
    #method="prominence"
    method = "dwt"
)


# where did DWT put the landmarks, relative to the median beat?
for k in ["ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]:
    offs = [(w - r) * 1000 / fs for w, r in zip(waves[k], rpeaks)
            if np.isfinite(w)]
    print(f"{k:20s} median {np.median(offs):6.0f} ms   IQR {np.percentile(offs,25):.0f}–{np.percentile(offs,75):.0f}")


#======================================== TO BE REMOVED


import numpy as np, matplotlib.pyplot as plt, neurokit2 as nk

LM = [("ECG_R_Offsets", "QRS off", "#2E86C1"),
      ("ECG_T_Onsets",  "T on",    "#E67E22"),
      ("ECG_T_Peaks",   "T",       "#B03A2E"),
      ("ECG_T_Offsets", "T off",   "#27AE60")]


def delineate_median_beat(median_beat, pre, fs=500, method="dwt", n_tile=12):
    """Delineate a single median beat by tiling it into a synthetic strip.
    Returns {key: ms from R peak}."""
    strip = np.tile(median_beat, n_tile)
    tiled_r = np.array([pre + i * len(median_beat) for i in range(n_tile)])
    try:
        _, w = nk.ecg_delineate(strip, rpeaks=tiled_r, sampling_rate=fs, method=method)
    except Exception:
        return {}
    out = {}
    for k, *_ in LM:
        offs = [(v - r) * 1000 / fs for v, r in zip(w.get(k, []), tiled_r)
                if np.isfinite(v)]
        out[k] = np.median(offs[1:-1]) if len(offs) > 2 else np.nan   # drop edge beats
    return out


def windowed_analysis(cleaned, rpeaks, fs=500, window_s=60, min_beats=10,
                      method="dwt"):
    win, pre = window_s * fs, int(0.60 * fs)
    rr_all = np.diff(rpeaks)
    results = []

    for w in range(int(np.ceil(len(cleaned) / win))):
        lo, hi = w * win, (w + 1) * win
        idx = [i for i, r in enumerate(rpeaks) if lo <= r < hi and i < len(rr_all)]
        if len(idx) < min_beats:
            continue

        rr = np.median(rr_all[idx])
        post = int(rr - pre)
        if post <= 0:
            continue

        beats = np.array([cleaned[rpeaks[i]-pre : rpeaks[i]+post] for i in idx
                          if rpeaks[i]-pre >= 0 and rpeaks[i]+post < len(cleaned)])
        if len(beats) < min_beats:
            continue

        mb = np.median(beats, axis=0)
        results.append({
            "window": w,
            "t_start_s": lo / fs,
            "median_beat": mb,
            "pre": pre,
            "n_beats": len(beats),
            "hr_bpm": 60 * fs / rr,
            "landmarks": delineate_median_beat(mb, pre, fs, method),
        })
    return results

res = windowed_analysis(cleaned, rpeaks, fs=fs, window_s=60, method="dwt")

hdr = f"{'win':>4} {'t(s)':>6} {'beats':>6} {'HR':>5}  " + \
      "".join(f"{lab:>9}" for _, lab, _ in LM)
print(hdr); print("-" * len(hdr))
for r in res:
    vals = "".join(f"{r['landmarks'].get(k, np.nan):9.0f}" for k, *_ in LM)
    print(f"{r['window']:>4} {r['t_start_s']:>6.0f} {r['n_beats']:>6} "
          f"{r['hr_bpm']:>5.0f}  {vals}")

# spread across windows — the medium-term variability
for k, lab, _ in LM:
    v = np.array([r["landmarks"].get(k, np.nan) for r in res], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) > 1:
        print(f"{lab:8s} median {np.median(v):6.0f} ms   "
              f"IQR {np.percentile(v,25):.0f}–{np.percentile(v,75):.0f}   SD {v.std():.1f}")

fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(res)))

for r, c in zip(res, cmap):
    t_ms = (np.arange(len(r["median_beat"])) - r["pre"]) * 1000 / fs
    ax.plot(t_ms, r["median_beat"], color=c, linewidth=1.2, alpha=0.85,
            label=f"{r['t_start_s']:.0f}s ({r['n_beats']} beats)")

# landmarks from every window, as short ticks at the top
y_top = ax.get_ylim()[1]
for k, lab, col in LM:
    xs = [r["landmarks"].get(k, np.nan) for r in res]
    xs = [x for x in xs if np.isfinite(x)]
    ax.plot(xs, [y_top * 0.95] * len(xs), "|", color=col, markersize=14,
            markeredgewidth=2)
    if xs:
        ax.text(np.median(xs), y_top * 1.02, lab, color=col, fontsize=9, ha="center")

ax.axvline(0, color="#34495E", linestyle=":", linewidth=1)
ax.set_xlabel("Time from R peak (ms)")
ax.set_title(f"30-s median beats ({len(res)} windows)")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=2, loc="lower right")
plt.tight_layout(); plt.show()

#========================================== TO BE REMOVED

"""

Here we will plot the delinated points on top of the previous plots

"""


import matplotlib.pyplot as plt
import numpy as np

fs = 500

# Landmark: (key in waves, label, color, marker)
LANDMARKS = [
    #("ECG_P_Onsets",  "P on",   "#7F8C8D", "|"),
    #("ECG_P_Peaks",   "P",      "#7F8C8D", "o"),
    #("ECG_P_Offsets", "P off",  "#7F8C8D", "|"),
    ("ECG_R_Onsets",  "QRS on", "#2E86C1", "|"),
    ("ECG_R_Offsets", "QRS off","#2E86C1", "v"),
    ("ECG_T_Onsets",  "T on",   "#E67E22", ">"),
    ("ECG_T_Peaks",   "T",      "#B03A2E", "o"),
    ("ECG_T_Offsets", "T off",  "#27AE60", "<"),
]


def plot_delineation(signal, waves, rpeaks=None, start=0, dur=5, fs=500,
                     ax=None, title=""):
    """Overlay delineated points on a window of the cleaned signal."""
    N = dur * fs
    start = int(min(max(start, 0), len(signal) - N))
    stop = start + N
    t = np.arange(start, stop) / fs

    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))
    ax.plot(t, signal[start:stop], linewidth=0.9, color="#34495E", zorder=1)

    if rpeaks is not None:
        r = [x for x in rpeaks if start <= x < stop]
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

_, rpeak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]

_, waves_dwt = nk.ecg_delineate(cleaned, rpeaks=rpeaks,
                                sampling_rate=fs, method="prominence")

plot_delineation(cleaned, waves_dwt, rpeaks, start=10000, dur=5, fs=fs,
                 title="DWT")
plt.tight_layout(); plt.show()
#==================

_, rpeak_info = nk.ecg_peaks(
    cleaned,
    sampling_rate=500
)

len(rpeak_info['ECG_R_Peaks'])

# Delineate waves
signals, waves = nk.ecg_delineate(
    cleaned,
    rpeaks=rpeak_info["ECG_R_Peaks"],
    sampling_rate=500,
    #method="prominence"
    method = "dwt"
)


#After this jumpt to the NEW part

# T-peak sample locations
t_peaks = np.asarray(waves["ECG_T_Peaks"], dtype=float)
t_peaks = t_peaks[np.isfinite(t_peaks)].astype(int)

print(t_peaks)
print("Number of T peaks:", len(t_peaks))


fs = 500
cleaned = np.asarray(cleaned).squeeze()

# R-peak locations
_, r_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
r_peaks = np.asarray(r_info["ECG_R_Peaks"], dtype=int)

# Valid T-peak locations
t_peaks = np.asarray(t_peaks)
t_peaks = t_peaks[np.isfinite(t_peaks)].astype(int)

rows = []

for t_peak in t_peaks:

    # Find the R peak immediately preceding this T peak
    preceding_r = r_peaks[r_peaks < t_peak]

    if len(preceding_r) == 0:
        continue

    r_peak = preceding_r[-1]

    # QRS window: 100 ms before to 120 ms after R peak
    qrs_start = max(0, r_peak - int(0.100 * fs))
    qrs_end = min(len(cleaned), r_peak + int(0.120 * fs) + 1)

    qrs_segment = cleaned[qrs_start:qrs_end]
    qrs_amplitude = np.ptp(qrs_segment)  # maximum - minimum

    if not np.isfinite(qrs_amplitude) or qrs_amplitude <= 0:
        continue

    t_peak_voltage = cleaned[t_peak]

    rows.append({
        "R_peak_sample": r_peak,
        "T_peak_sample": t_peak,
        "T_peak_time_seconds": t_peak / fs,
        "T_peak_voltage": t_peak_voltage,
        "QRS_peak_to_peak_amplitude": qrs_amplitude,
        "T_peak_QRS_ratio": t_peak_voltage / qrs_amplitude
    })

t_peak_data = pd.DataFrame(rows)

print(t_peak_data.head())
print(t_peak_data["T_peak_QRS_ratio"].describe())
print(
    "Percent negative:",
    100 * (t_peak_data["T_peak_QRS_ratio"] < 0).mean()
)

results = []

for ton, tpeak, toff in zip(
    waves["ECG_T_Onsets"],
    waves["ECG_T_Peaks"],
    waves["ECG_T_Offsets"]
):
    if not np.all(np.isfinite([ton, tpeak, toff])):
        continue

    ton, tpeak, toff = map(int, [ton, tpeak, toff])

    if not ton < tpeak < toff:
        continue

    baseline_at_peak = np.interp(
        tpeak,
        [ton, toff],
        [cleaned[ton], cleaned[toff]]
    )

    signed_amplitude = cleaned[tpeak] - baseline_at_peak

    results.append({
        "T_on": ton,
        "T_peak": tpeak,
        "T_off": toff,
        "T_peak_time_seconds": tpeak / 500,
        "T_peak_signed_amplitude": signed_amplitude
    })

t_wave_data = pd.DataFrame(results)

#============================== NEW NEW NEW 

import numpy as np
import pandas as pd

fs = 500
cleaned = np.asarray(cleaned).squeeze()

# From a given one lead, let us see what we can get from base neurokit

print(waves.keys())

"""

'ECG_P_Onsets', 'ECG_P_Peaks',
'ECG_P_Offsets', 'ECG_Q_Peaks',
'ECG_R_Onsets', 'ECG_R_Offsets',
'ECG_S_Peaks', 'ECG_T_Onsets',
'ECG_T_Peaks', 'ECG_T_Offsets'


# Here since this is the base deliniation, we should first plot these on 
the ecg so that we can see if the delination is accurate
"""

cleaned

rows = []

for qrs_on, qrs_off, t_on, t_peak, t_off, r_peak, s_peak in zip(
    waves["ECG_R_Onsets"], # check
    waves["ECG_R_Offsets"],
    waves["ECG_T_Onsets"],
    waves["ECG_T_Peaks"], #check
    waves["ECG_T_Offsets"], 
    waves["ECG_R_Peaks"], #check
    waves["ECG_S_Peaks"] #check
):
    points = [qrs_on, qrs_off, t_on, t_peak, t_off, r_peak, s_peak]
    qrs_on_next = np.append(qrs_on[1:], np.nan)
    if not np.all(np.isfinite(points)):
        continue

    qrs_on, qrs_off, t_on, t_peak, t_off = map(int, points)

    if not qrs_on < qrs_off <= t_on < t_peak < t_off:
        continue

    # QRS peak-to-peak amplitude
    qrs_segment = cleaned[qrs_on:qrs_off + 1]
    qrs_amplitude = np.ptp(qrs_segment)

    if not np.isfinite(qrs_amplitude) or qrs_amplitude <= 0:
        continue

    # Local baseline at the T peak
    baseline_at_tpeak = np.interp(
        t_peak,
        [t_on, t_off],
        [cleaned[t_on], cleaned[t_off]]
    )

    signed_t_amplitude = cleaned[t_peak] - baseline_at_tpeak #Here we have t peak , our numerator

    denominator = cleaned[r_peak] + cleaned[s_peak] - (2 * (baseline_at_tpeak))

    x = [r_on, r_on_next]
    y = [cleaned[r_on], cleaned[r_on_next]]
    B = lambda i: np.interp(i, x, y)

    numerator   = cleaned[t_peak] - B(t_peak)
    denominator = (cleaned[r_peak] - B(r_peak)) + (cleaned[s_peak] - B(s_peak))

    rows.append({
        "QRS_offset_to_T_onset_ms":
            (t_on - qrs_off) * 1000 / fs,

        "T_wave_duration_ms":
            (t_off - t_on) * 1000 / fs,

        "T_peak_signed_amplitude":
            signed_t_amplitude,

    })

beat_data = pd.DataFrame(rows)


beat_data


patient_features = beat_data.median().to_frame().T
patient_features["Valid_beats"] = len(beat_data)

patient_features


#===========================

"""
Here we will try to check the outliers 
if pid in ["064", "041"]:          # the patients you care about
    N = 3000
    w = [[x for x in waves[k] if np.isfinite(x) and x < N]
         for k in ["ECG_R_Offsets", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets"]]
    nk.events_plot(w, cleaned[:N])
    plt.title(f"sub_{pid}")
    plt.show()

"""






# ================ New Trial 

def getBeatData(csvFile, leadNumber = 2, method = "prominence"):
    ecg_signal = pd.read_csv(
        csvFile,
        header=None,
        nrows=12,
        dtype="float32"
    ).to_numpy().T
    ecg = np.asarray(ecg_signal[:, leadNumber -1], dtype=float)
    # In the aboe we are using V5
    cleaned = nk.ecg_clean(ecg, sampling_rate=500)
    _, rpeak_info = nk.ecg_peaks(
        cleaned,
        sampling_rate=500
    )

    # Delineate waves
    signals, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=rpeak_info["ECG_R_Peaks"],
        sampling_rate=500,
        method=method,
        #method="dwt"
    )
    fs = 500

    cleaned = np.asarray(cleaned).squeeze()
    #fs = 500
    rows = []
    rate = nk.ecg_rate(rpeak_info, sampling_rate=500, desired_length=len(cleaned))
    hr_bpm = np.median(rate)
    qrs_on_all = np.asarray(waves["ECG_R_Onsets"], dtype=float)
    qrs_on_next_all = np.append(qrs_on_all[1:], np.nan)

    for qrs_on, qrs_on_next, qrs_off, t_on, t_peak, t_off, r_peak, s_peak, p_on, p_off in zip(
        qrs_on_all,
        qrs_on_next_all,
        waves["ECG_R_Offsets"],
        waves["ECG_T_Onsets"],
        waves["ECG_T_Peaks"],
        waves["ECG_T_Offsets"],
        rpeak_info["ECG_R_Peaks"],
        waves["ECG_S_Peaks"],
        waves["ECG_P_Onsets"],
        waves["ECG_P_Offsets"]
    ):
        points = [qrs_on, qrs_off, t_on, t_peak, t_off, r_peak]
        if not np.all(np.isfinite(points)):
            continue
        qrs_on, qrs_off, t_on, t_peak, t_off, r_peak = map(int, points)

        if not (qrs_on <= r_peak <= qrs_off <= t_on < t_peak < t_off):
            continue

        # Baseline: line between this beat's and the next beat's QRS onset
        if np.isfinite(qrs_on_next) and int(qrs_on_next) > t_off:
            x = [qrs_on, int(qrs_on_next)]
            y = [cleaned[qrs_on], cleaned[int(qrs_on_next)]]
        else:
            x = [qrs_on, qrs_on + 1]                  # flat fallback
            y = [cleaned[qrs_on], cleaned[qrs_on]]
        B = lambda i: np.interp(i, x, y)

        # S: only if it falls inside the QRS, else treat as absent
        if np.isfinite(s_peak) and r_peak < int(s_peak) <= qrs_off:
            s_peak = int(s_peak)
            s_term = cleaned[s_peak] - B(s_peak)      # negative = depth below baseline
        else:
            s_term = 0.0

        numerator = cleaned[t_peak] - B(t_peak)
        denominator = (cleaned[r_peak] - B(r_peak)) + s_term

        ratio = numerator / denominator if abs(denominator) > 1e-6 else np.nan
        pr_ms = ((qrs_on - int(p_on)) * 1000 / fs
            if np.isfinite(p_on) and int(p_on) < qrs_on else np.nan)
        p_dur_ms = ((int(p_off) - int(p_on)) * 1000 / fs
            if np.isfinite(p_on) and np.isfinite(p_off) and int(p_on) < int(p_off) <= qrs_on
            else np.nan)
        rows.append({
            "QRS_offset_to_T_onset_ms": (t_on - qrs_off) * 1000 / fs,
            "QRS_onset_to_QRS_offset_ms": (qrs_off - qrs_on) * 1000 / fs,
            "PR_interval_ms": pr_ms,
            "QT_interval_ms": (t_off - qrs_on) * 1000 / fs,
            "T_wave_duration_ms": (t_off - t_on) * 1000 / fs,
            "T_peak_signed_amplitude": numerator,
            "QRS_balance": denominator,
            "T_over_QRS_balance": ratio,
            "P_duration_ms": p_dur_ms,
            
        })

    beat_data = pd.DataFrame(rows)
    beat_data["HR_bpm"] = np.mean(nk.ecg_rate(rpeak_info, sampling_rate=500))
    return beat_data

# Here we are loading one patient
csv_path = Path(
    "C:/Users/Estif/Downloads/Langone/ANOCA/"
    "ECG_ANOCA_Trial_1/Models/ECGFounder-master/"
    "ECGFounder-master/data/ScienceDB_Data/sub_010/"
    "sub_010_baseline-data_rest-ecg-500hz.csv"
)

#beat_data

checkBeatData = getBeatData(csv_path)


checkBeatData




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
            "R_to_T_peak_ms": (t_peak - r_peak) * 1000 / fs,
            "R_minus_T_amplitude": sig[r_peak] - sig[t_peak],
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


#===================================== NEW NEW 

# Here start iterating

import json

#folder_path = Path("C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data")

folder_path = Path("D:/ScienceDB_Data")

count = 0
all_Lead_Data = {}

fullDict = {} #file_path.name
for file_path in folder_path.iterdir():
    
    if file_path.name[0:3] == "sub":
        #print(file_path)
        patientID = int(file_path.name[4:])
        pid = file_path.name[4:]              # CHANGED
        print(pid)
        fullDict.setdefault(pid, {})          #  CHANGED: create the entry once
        words = ["baseline","data", "rest", "500hz" ]
        #words = ["data", "recover", "500hz" ]
        #words = ["data", "stress", "500hz" ] #"baseline",
        words_Disease = ['disease' , 'json']
        words_chief_complaint = ["complaint", "chief", "baseline" ] # sub_001_baseline-data_chief-complaint
        for csvFile in file_path.iterdir():
            contains_word = all(
                    word.lower() in (csvFile.name).lower()
                            for word in words
                        )
            contains_Disease = all(
                    words_Disease.lower() in (csvFile.name).lower()
                            for words_Disease in words_Disease
                        )
            contains_Chief_Complaint = all(
                                words_chief_complaint.lower() in (csvFile.name).lower()
                                        for words_chief_complaint in words_chief_complaint
                                    )
            if contains_Disease == True:
                with open(csvFile, "r") as file:
                    disease_data = json.load(file)
                #output150 = turnTo150Output(disease_data['ANOCA'], 0)
                fullDict[pid]["CMD"] = disease_data["CMD"]       # CHANGED
                fullDict[pid]["ANOCA"] = disease_data["ANOCA"]   # CHANGED
            if contains_Chief_Complaint == True:
                with open(csvFile, "r") as file:
                    disease_data = json.load(file)
                #output150 = turnTo150Output(disease_data['ANOCA'], 0)
                #fullDict[pid]["BMI"] = (disease_data["Demographic Information"])["BMI"]       # CHANGED
                #fullDict[pid]["Age"] = (disease_data["Demographic Information"])["Age"]   # CHANGED
                demo = disease_data["Demographic Information"][0]
                fullDict[pid]["BMI"] = demo["BMI"]
                fullDict[pid]["Age"] = demo["Age"] 
            if contains_word == True:
                print(csvFile)
                "======================================================="
                "The input is csvfile location"
                """
                ecg_signal = pd.read_csv(
                                        csvFile,
                                        header=None,
                                        nrows=12,
                                        dtype="float32"
                                    ).to_numpy().T
                ecg = np.asarray(ecg_signal[:, cc], dtype=float)
                # In the aboe we are using V5
                cleaned = nk.ecg_clean(ecg, sampling_rate=500)
                _, rpeak_info = nk.ecg_peaks(
                    cleaned,
                    sampling_rate=500
                )

                # Delineate waves
                signals, waves = nk.ecg_delineate(
                    cleaned,
                    rpeaks=rpeak_info["ECG_R_Peaks"],
                    sampling_rate=500,
                    #method="prominence",
                    method="dwt"
                )
                fs = 500

                cleaned = np.asarray(cleaned).squeeze()

                rows = []

                for qrs_on, qrs_off, t_on, t_peak, t_off in zip(
                    waves["ECG_R_Onsets"],
                    waves["ECG_R_Offsets"],
                    waves["ECG_T_Onsets"],
                    waves["ECG_T_Peaks"],
                    waves["ECG_T_Offsets"]
                    ):
                    points = [qrs_on, qrs_off, t_on, t_peak, t_off]

                    if not np.all(np.isfinite(points)):
                        continue

                    qrs_on, qrs_off, t_on, t_peak, t_off = map(int, points)

                    if not qrs_on < qrs_off <= t_on < t_peak < t_off:
                        continue

                    if not (40 <= (t_on - qrs_off) * 1000 / fs <= 250):   # ST segment
                        continue
                    if not (80 <= (t_off - t_on) * 1000 / fs <= 350):     # T-wave duration
                        continue

                    # QRS peak-to-peak amplitude
                    qrs_segment = cleaned[qrs_on:qrs_off + 1]
                    qrs_amplitude = np.ptp(qrs_segment)

                    if not np.isfinite(qrs_amplitude) or qrs_amplitude <= 0:
                        continue


                    
                    # Local baseline at the T peak
                    baseline_at_tpeak = np.interp(
                        t_peak,
                        [t_on, t_off],
                        [cleaned[t_on], cleaned[t_off]]
                    )

                    signed_t_amplitude = cleaned[t_peak] - baseline_at_tpeak

                    # T wave as a distribution over time
                    idx = np.arange(t_on, t_off + 1)
                    baseline_line = np.interp(
                        idx, [t_on, t_off], [cleaned[t_on], cleaned[t_off]]
                    )
                    w = np.abs(cleaned[idx] - baseline_line)   # height above baseline

                    total = w.sum()
                    if total <= 0:
                        continue

                    cdf = np.cumsum(w) / total
                    q25, q50, q75 = np.interp([0.25, 0.50, 0.75], cdf, idx)

                    iqr = q75 - q25
                    t_qskew = (q75 + q25 - 2 * q50) / iqr if iqr > 0 else np.nan

                    rows.append({
                        "QRS_offset_to_T_onset_ms":
                            (t_on - qrs_off) * 1000 / fs,

                        "T_wave_duration_ms":
                            (t_off - t_on) * 1000 / fs,

                        "T_peak_signed_amplitude":
                            signed_t_amplitude,

                        "T_Skew":t_qskew,

                    })

                beat_data = pd.DataFrame(rows)
                """

                "======================================================="
                "Here it can be replaced by a function return beat_data"
                beat_data = getBeatData(csvFile,method = "prominence")
                # Absolute first-order difference between successive beats
                #exclude = ["QRS_balance", "T_over_QRS_balance"]      # whichever you don't want
                #abs_diff_data = beat_data.drop(columns=exclude).diff().abs().iloc[1:] 
                # here add IQR beat_data.quantile(.75) - beat_data.quantile(.25)
                #Here print the IQR 
                
                print("Here is the IQR")
                iqr_features = (
                    beat_data.quantile(0.75) - beat_data.quantile(0.25)
                ).add_suffix("_IQR").to_frame().T

                #print(iqr_features)
                #abs_diff_data = beat_data.drop(columns=exclude).diff().abs().iloc[1:]
                # Median of the original features
                patient_features = beat_data.median().to_frame().T
                print(patient_features.keys())
                # Median of the absolute differences
                #diff_features = abs_diff_data.median().add_suffix("_absdiff").to_frame().T

                patient_features = pd.concat([patient_features,  iqr_features], axis=1) #diff_features,
                patient_features["Valid_beats"] = len(beat_data)
                #print(patient_features['QRS_offset_to_T_onset_ms'] >= 200 or patient_features['QRS_offset_to_T_onset_ms'] < 200)
                #if beat_data['QRS_offset_to_T_onset_ms'] > 200:
                print(patient_features)

                fullDict[pid].update(patient_features.iloc[0].to_dict())   # CHANGED: add, don't overwrite
    all_Lead_Data[5] = fullDict


all_Lead_Data.keys()

"""
Here we will plot the prominence algorithm based

"""
# Save as csv
prominence_df = pd.DataFrame.from_dict(all_Lead_Data[5], orient="index")   # CHANGED: one row per patient
print(prominence_df )

prominence_df.keys()

prominence_df.shape

#==================================

"""
And here we will plot the dwt algorithms based 
"""
dwt_df = pd.DataFrame.from_dict(all_Lead_Data[5], orient="index")   # CHANGED: one row per patient
print(dwt_df )

dwt_df.keys()

dwt_df.shape

#=====================================
#Place for plot

def compare_plot(prom, dwt, col, lo=None, hi=None):
    order = prom[col].sort_values().index
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4))
    if lo is not None:
        ax.axhspan(lo, hi, color="0.93", zorder=0)                       # normal range
    for df, name, color, dx in [(prom, "prominence", "#2a78d6", -0.15),
                                (dwt,  "dwt",        "#eb6834",  0.15)]:
        m, iqr = df.loc[order, col], df.loc[order, f"{col}_IQR"]
        ax.errorbar(x + dx, m, yerr=iqr / 2, fmt="o", ms=4, color=color,
                    elinewidth=1, capsize=0, label=name)
    ax.set(xlabel="Subject (sorted by prominence value)", ylabel=col, xticks=[])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

compare_plot(prominence_df, dwt_df, "QRS_onset_to_QRS_offset_ms", 80, 110)
compare_plot(prominence_df, dwt_df, "PR_interval_ms", 120, 200)
compare_plot(prominence_df, dwt_df, "QT_interval_ms", 350, 450)
compare_plot(prominence_df, dwt_df, "P_duration_ms", 80, 120)





#====================================
beat_data
# Save as csv
df = pd.DataFrame.from_dict(all_Lead_Data[5], orient="index")   # CHANGED: one row per patient
print(df)

df
df.keys()

fullDict["097"].keys()

combined_df = pd.concat(
    [
        pd.DataFrame(subject_data).assign(subject_id=subject_id)
        for subject_id, subject_data in fullDict.items()
    ],
    ignore_index=True
)


# Put subject_id first
combined_df = combined_df[
    ["subject_id"] +
    [column for column in combined_df.columns if column != "subject_id"]
]

combined_df['T_peak_signed_amplitude'] = round(combined_df['T_peak_signed_amplitude'] , 3)   

combined_df.columns

combined_df = df

#Correlation
columns = [
    'QRS_offset_to_T_onset_ms',
    'T_wave_duration_ms', #'T_peak_signed_amplitude', #'QRS_balance',
    'T_over_QRS_balance', 'QRS_offset_to_T_onset_ms_absdiff',
     'T_peak_signed_amplitude_absdiff',
    #'T_peak_signed_amplitude_IQR',
    'QRS_balance_IQR', #'QRS_offset_to_T_onset_ms_IQR',
    #'T_wave_duration_ms_IQR', 'T_wave_duration_ms_absdiff',
    'T_over_QRS_balance_IQR'

]

correlation_matrix = (
    combined_df[columns]
    .apply(pd.to_numeric, errors="coerce")
    .corr(method="spearman")
)

print(correlation_matrix.round(3))






import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Spearman Correlation Matrix")
plt.show()

#Regression

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

predictors = [
    "QRS_offset_to_T_onset_ms",
    #"T_wave_duration_ms",
    "T_peak_signed_amplitude",
    #'QRS_offset_to_T_onset_ms_absdiff',
    #'T_wave_duration_ms_absdiff',
   #'T_peak_signed_amplitude_absdiff',
    #'BMI',
    'Age',
    #'T_Skew',
    #'T_Skew_absdiff'
    'QRS_balance',
    #'T_over_QRS_balance',
    'HR_bpm'
]


model_data = combined_df[
    ["ANOCA"] + predictors
].copy()

# Keep only the columns used in the model, replace infs with NaN, then drop incomplete rows
model_data = model_data[predictors + ["ANOCA"]].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Patients used in model: {len(model_data)}")
# First inspect the current label format
print(model_data["ANOCA"].value_counts(dropna=False))

X = sm.add_constant(model_data[predictors])
y = model_data["ANOCA"]

model = sm.Logit(y, X).fit()

print(model.summary())


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

X = df[["QRS_offset_to_T_onset_ms", "T_peak_signed_amplitude", "QRS_balance", "Age", "HR_bpm"]]
y = df["CMD"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=50, random_state=0)
auc = cross_val_score(LogisticRegression(max_iter=1000, C=1e6), X, y, cv=cv, scoring="roc_auc")
print(auc.mean(), np.percentile(auc, [2.5, 97.5]))


import matplotlib.pyplot as plt

"""
"QRS_offset_to_T_onset_ms",
    "T_wave_duration_ms",
    "T_peak_signed_amplitude",
    'QRS_offset_to_T_onset_ms_absdiff',
    'T_wave_duration_ms_absdiff',
    'T_peak_signed_amplitude_absdiff',
    'BMI',
    'Age',
    #'T_Skew',
    #T_Skew_absdiff',
    'QRS_balance',
    'T_over_QRS_balance',
    'HR_bpm'
"""

features = [
    'QRS_offset_to_T_onset_ms',
    'T_wave_duration_ms', #'T_peak_signed_amplitude', #'QRS_balance',
    'T_over_QRS_balance', 'QRS_offset_to_T_onset_ms_absdiff',
    'T_peak_signed_amplitude_absdiff',
    #'T_peak_signed_amplitude_IQR',
    'QRS_balance_IQR', #'QRS_offset_to_T_onset_ms_IQR',
    #'T_wave_duration_ms_IQR', 'T_wave_duration_ms_absdiff',
    'T_over_QRS_balance_IQR'
]

#["QRS_offset_to_T_onset_ms", "T_peak_signed_amplitude", "QRS_balance", "HR_bpm"]
groups = [0, 1]
labels = ["Non-ANOCA", "ANOCA"]

fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.6))
rng = np.random.default_rng(0)

for ax, f in zip(axes, features):
    data = [df.loc[df["ANOCA"] == g, f].dropna() for g in groups]
    ax.boxplot(data, widths=0.5, showfliers=False,
               medianprops=dict(color="black", linewidth=2),
               boxprops=dict(color="0.4"), whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
    for i, d in enumerate(data, start=1):                     # individual patients
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, alpha=0.5, color="#2F6FDE")
    ax.set_xticks([1, 2], [f"{l}\n(n={len(d)})" for l, d in zip(labels, data)])
    ax.set_title(f, fontsize=9)
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
plt.show()



#Stepwise selection 

combined_df.keys()

candidates = [
    'QRS_offset_to_T_onset_ms',
    'T_wave_duration_ms', #'T_peak_signed_amplitude', #'QRS_balance',
    'T_over_QRS_balance', #'QRS_offset_to_T_onset_ms_absdiff',
    #'T_peak_signed_amplitude_absdiff',
    #'T_peak_signed_amplitude_IQR',
    'QRS_balance_IQR', #'QRS_offset_to_T_onset_ms_IQR',
    #'T_wave_duration_ms_IQR', 'T_wave_duration_ms_absdiff',
    'T_over_QRS_balance_IQR', "HR_bpm"
]


X =df[candidates]
print(X.nunique().sort_values().head())                         # 1 = constant column
print(np.linalg.matrix_rank(X.values), "of", X.shape[1])       # rank below the count = redundant columns
c = X.corr().abs(); np.fill_diagonal(c.values, 0)
print(c.stack().sort_values(ascending=False).head(5)) 


forced = []          # always kept; set to [] to let everything compete ["Age", "HR_bpm"] 
outcome = "ANOCA"

# Same rows for every model, so AICs are comparable
data = (combined_df[[outcome] + forced + candidates]
        .replace([np.inf, -np.inf], np.nan).dropna())
y = data[outcome]
print(f"Patients: {len(data)}")

def aic(cols):
    try:
        return sm.Logit(y, sm.add_constant(data[cols])).fit(disp=0).aic
    except (np.linalg.LinAlgError, Exception):
        return np.inf          # this combination can't be fit, so skip it
selected = []
current = aic(forced + selected)
while True:
    changed = False
    # try adding
    adds = {c: aic(forced + selected + [c]) for c in candidates if c not in selected}
    if adds:
        best = min(adds, key=adds.get)
        if adds[best] < current - 1e-6:
            selected.append(best); current = adds[best]; changed = True
            print(f"+ {best:40s} AIC={current:.2f}")
    # try removing
    if selected:
        drops = {c: aic(forced + [s for s in selected if s != c]) for c in selected}
        worst = min(drops, key=drops.get)
        if drops[worst] < current - 1e-6:
            selected.remove(worst); current = drops[worst]; changed = True
            print(f"- {worst:40s} AIC={current:.2f}")
    if not changed:
        break

final = sm.Logit(y, sm.add_constant(data[forced + selected])).fit(disp=0)
print(final.summary())
from sklearn.metrics import roc_auc_score

X_final = sm.add_constant(data[forced + selected])
print(f"In-sample AUC: {roc_auc_score(y, final.predict(X_final)):.3f}")



"""
Here we will try elasticNet as opposed to stepwise

"""

#================
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegressionCV

df_all = df.copy(); df = df[~(df["T_over_QRS_balance"] < -5)]     # removes patient 030; restore with df = df_all
feats = ["Age", "BMI", "HR_bpm", "HRV_RMSSD", "PR_interval_ms", "QTcF_ms",
         "QRS_offset_to_T_onset_ms", "T_peak_signed_amplitude", "QRS_balance", "T_over_QRS_balance"]

feats = ["Age", "BMI", "HR_bpm", "HRV_RMSSD", "PR_interval_ms", "QTcF_ms",
         "QRS_offset_to_T_onset_ms", "T_peak_signed_amplitude", "QRS_balance", "T_over_QRS_balance",
         "R_to_T_peak_ms"]


d = df_all.dropna(subset=feats + ["ANOCA"])
X, y = d[feats], d["ANOCA"]
print(len(d), "patients")

enet = make_pipeline(StandardScaler(), LogisticRegressionCV(
    penalty="elasticnet", solver="saga", l1_ratios=[.1, .5, .9], Cs=20,
    cv=5, scoring="roc_auc", max_iter=10000))

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
auc = cross_val_score(enet, X, y, cv=cv, scoring="roc_auc")
print(f"Elastic net CV AUC = {auc.mean():.2f} ({np.percentile(auc, 2.5):.2f}–{np.percentile(auc, 97.5):.2f})")

enet.fit(X, y)
print(pd.Series(enet[-1].coef_[0], index=feats).sort_values())

# inference model (pre-specified)
print(smf.logit("ANOCA ~ T_over_QRS_balance + Age + BMI", d).fit().summary())


from scipy.stats import chi2
lin = smf.logit("ANOCA ~ T_over_QRS_balance + Age + BMI", d).fit(disp=0)
spl = smf.logit("ANOCA ~ cr(T_over_QRS_balance, df=3) + Age + BMI", d).fit(disp=0)
spl = smf.logit("ANOCA ~ cr(T_over_QRS_balance, df=3, constraints='center') + Age + BMI", d).fit(disp=0)
print(spl.mle_retvals["converged"])
lr = 2 * (spl.llf - lin.llf)
print(f"LR = {lr:.2f}, p = {chi2.sf(lr, spl.df_model - lin.df_model):.3f}, AIC lin = {lin.aic:.1f}, spline = {spl.aic:.1f}")


grid = pd.DataFrame({"T_over_QRS_balance": np.linspace(d["T_over_QRS_balance"].min(), d["T_over_QRS_balance"].max(), 200),
                     "Age": d["Age"].mean(), "BMI": d["BMI"].mean()})
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(grid["T_over_QRS_balance"], spl.predict(grid), label="Spline")
ax.plot(grid["T_over_QRS_balance"], lin.predict(grid), "--", label="Linear")
ax.scatter(d["T_over_QRS_balance"], d["ANOCA"] + np.random.default_rng(0).normal(0, .02, len(d)), s=12, alpha=.4)
ax.set_xlabel("T/QRS balance (lead II)"); ax.set_ylabel("Probability of ANOCA")
ax.set_title("Probability of ANOCA vs T/QRS balance"); ax.legend(); plt.tight_layout(); plt.show()


from scipy.stats import fisher_exact
ct = pd.crosstab(d["T_over_QRS_balance"] < 0.1, d["ANOCA"])
print(ct); print(fisher_exact(ct))

print(smf.logit("ANOCA ~ T_peak_signed_amplitude + Age + BMI", d).fit().summary())
print(enet[-1].l1_ratio_, enet[-1].C_)   # which mix and penalty strength it chose


d = df.dropna(subset=["T_over_QRS_balance", "ANOCA"])
groups = [d.loc[d["ANOCA"] == g, "T_over_QRS_balance"] for g in (0, 1)]

fig, ax = plt.subplots(figsize=(5, 5.5))
ax.boxplot(groups, widths=.5, showfliers=False)               # dots below show every patient
for i, g in enumerate(groups, 1):
    ax.scatter(np.random.default_rng(0).normal(i, .06, len(g)), g, s=18, alpha=.6)
ax.set_xticks([1, 2], [f"ANOCA negative\n(n = {len(groups[0])})", f"ANOCA positive\n(n = {len(groups[1])})"])
ax.set_ylabel("T/QRS balance (lead II, median per patient)")
ax.set_title("T/QRS balance by ANOCA status")
plt.tight_layout(); plt.show()


from scipy.stats import mannwhitneyu
print(mannwhitneyu(*groups))                                    # rank-based, unaffected by the −15.5 value
print(d.loc[d["T_over_QRS_balance"] < -5, ["T_peak_signed_amplitude", "QRS_balance", "ANOCA"]])  # is QRS_balance ~0?
print(smf.logit("ANOCA ~ T_over_QRS_balance + Age + BMI", d[d["T_over_QRS_balance"] > -5]).fit().summary())  # sensitivity only

#================


print(d[["T_peak_signed_amplitude", "T_over_QRS_balance"]].corr(method="spearman"))

m = smf.logit("ANOCA ~ T_peak_signed_amplitude + T_over_QRS_balance + Age + BMI", d).fit()
print(m.summary())
print(m.wald_test("T_peak_signed_amplitude = 0, T_over_QRS_balance = 0", scalar=True))   # joint T-wave test


m = smf.logit("ANOCA ~ T_peak_signed_amplitude + Age + BMI", d).fit()
print(m.summary())
print("OR per 100 µV lower T amplitude:", np.exp(-100 * m.params["T_peak_signed_amplitude"]),
      np.exp(-100 * m.conf_int().loc["T_peak_signed_amplitude"])[::-1].values)


for c in ["T_peak_signed_amplitude", "T_over_QRS_balance"]:
    m = smf.logit(f"ANOCA ~ {c} + Age + BMI", d).fit(disp=0)
    sd = d[c].std()
    print(f"{c}: OR per 1 SD lower = {np.exp(-m.params[c] * sd):.2f} "
          f"({np.exp(-m.conf_int().loc[c, 1] * sd):.2f}–{np.exp(-m.conf_int().loc[c, 0] * sd):.2f}), SD = {sd:.2f}")

"""
At Rest

T_wave_duration_ms_absdiff          19
QRS_offset_to_T_onset_ms_absdiff    26
BMI                                 34
T_wave_duration_ms                  49
QRS_offset_to_T_onset_ms            59
dtype: int64
11 of 11
T_peak_signed_amplitude           T_wave_duration_ms                  0.611824
T_wave_duration_ms                T_peak_signed_amplitude             0.611824
                                  QRS_offset_to_T_onset_ms_absdiff    0.603931
QRS_offset_to_T_onset_ms_absdiff  T_wave_duration_ms                  0.603931
T_peak_signed_amplitude           T_wave_duration_ms_absdiff          0.516639
dtype: float64
Patients: 97
+ T_wave_duration_ms                       AIC=118.73
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       95
Method:                           MLE   Df Model:                            1
Date:                Thu, 24 Sep 2026   Pseudo R-squ.:                 0.08702
Time:                        14:03:27   Log-Likelihood:                -57.366
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                 0.0009435
======================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------
const                  4.3433      1.356      3.204      0.001       1.686       7.000
T_wave_duration_ms    -0.0280      0.010     -2.877      0.004      -0.047      -0.009
======================================================================================


At Stress

T_wave_duration_ms_absdiff          25
QRS_offset_to_T_onset_ms_absdiff    30
BMI                                 34
T_wave_duration_ms                  50
QRS_offset_to_T_onset_ms            65
dtype: int64
10 of 10
T_peak_signed_amplitude  T_wave_duration_ms                  0.623298
T_wave_duration_ms       T_peak_signed_amplitude             0.623298
T_peak_signed_amplitude  T_over_QRS_balance                  0.610210
T_over_QRS_balance       T_peak_signed_amplitude             0.610210
T_wave_duration_ms       QRS_offset_to_T_onset_ms_absdiff    0.567957
dtype: float64
Patients: 97
+ QRS_offset_to_T_onset_ms_absdiff         AIC=119.91
+ T_over_QRS_balance                       AIC=118.90
+ QRS_offset_to_T_onset_ms                 AIC=117.30
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       92
Method:                           MLE   Df Model:                            4
Date:                Thu, 24 Sep 2026   Pseudo R-squ.:                  0.1461
Time:                        13:44:21   Log-Likelihood:                -53.652
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                  0.001048
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
const                                0.1808      1.978      0.091      0.927      -3.696       4.057
HR_bpm                              -0.0282      0.020     -1.385      0.166      -0.068       0.012
QRS_offset_to_T_onset_ms_absdiff     0.0538      0.023      2.312      0.021       0.008       0.099
T_over_QRS_balance                  -1.2497      0.671     -1.863      0.063      -2.565       0.065
QRS_offset_to_T_onset_ms             0.0123      0.007      1.813      0.070      -0.001       0.026
====================================================================================================



At Recovery 
QRS_offset_to_T_onset_ms_absdiff    15
T_wave_duration_ms_absdiff          16
T_wave_duration_ms                  29
Age                                 34
QRS_offset_to_T_onset_ms            46
dtype: int64
10 of 10
T_wave_duration_ms_absdiff        QRS_offset_to_T_onset_ms_absdiff    0.823446
QRS_offset_to_T_onset_ms_absdiff  T_wave_duration_ms_absdiff          0.823446
T_wave_duration_ms_absdiff        T_peak_signed_amplitude             0.668699
T_peak_signed_amplitude           T_wave_duration_ms_absdiff          0.668699
                                  QRS_offset_to_T_onset_ms_absdiff    0.559159
dtype: float64
Patients: 97
+ T_wave_duration_ms_absdiff               AIC=119.19
+ T_over_QRS_balance                       AIC=116.08
+ QRS_offset_to_T_onset_ms                 AIC=113.55
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       92
Method:                           MLE   Df Model:                            4
Date:                Fri, 25 Sep 2026   Pseudo R-squ.:                  0.1760
Time:                        09:46:24   Log-Likelihood:                -51.774
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                 0.0001897
==============================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------
const                         -2.5545      2.668     -0.958      0.338      -7.783       2.674
HR_bpm                         0.0067      0.023      0.292      0.770      -0.039       0.052
T_wave_duration_ms_absdiff     0.0946      0.052      1.802      0.071      -0.008       0.197
T_over_QRS_balance            -2.3323      0.946     -2.466      0.014      -4.186      -0.479
QRS_offset_to_T_onset_ms       0.0235      0.011      2.072      0.038       0.001       0.046
==============================================================================================
"""


"""

Lead 1

Patients used in model: 97
ANOCA
1    63
0    34
Name: count, dtype: int64
Optimization terminated successfully.
         Current function value: 0.546465
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       86
Method:                           MLE   Df Model:                           10
Date:                Thu, 17 Sep 2026   Pseudo R-squ.:                  0.1564
Time:                        13:23:02   Log-Likelihood:                -53.007
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                   0.03272
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
const                                7.1968      4.745      1.517      0.129      -2.103      16.497
QRS_offset_to_T_onset_ms             0.0104      0.009      1.126      0.260      -0.008       0.029
T_wave_duration_ms                  -0.0726      0.030     -2.451      0.014      -0.131      -0.015
T_peak_signed_amplitude              0.0379      0.029      1.330      0.184      -0.018       0.094
QRS_offset_to_T_onset_ms_absdiff    -0.0374      0.042     -0.884      0.377      -0.120       0.045
T_wave_duration_ms_absdiff           0.0549      0.065      0.842      0.400      -0.073       0.183
T_peak_signed_amplitude_absdiff     -0.1444      0.092     -1.571      0.116      -0.325       0.036
BMI                                 -0.0105      0.029     -0.356      0.722      -0.068       0.047
Age                                  0.0358      0.102      0.353      0.724      -0.163       0.235
T_Skew                             -11.8533     16.405     -0.723      0.470     -44.006      20.299
T_Skew_absdiff                       9.9156     31.695      0.313      0.754     -52.206      72.038
====================================================================================================

Lead 1
Name: count, dtype: int64
Optimization terminated successfully.
         Current function value: 0.624013
         Iterations 5
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       93
Method:                           MLE   Df Model:                            3
Date:                Fri, 11 Sep 2026   Pseudo R-squ.:                 0.03667
Time:                        13:29:52   Log-Likelihood:                -60.529
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                    0.2029
============================================================================================
                               coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------------
const                        3.8728      2.890      1.340      0.180      -1.792       9.537
QRS_offset_to_T_onset_ms     0.0084      0.009      0.955      0.339      -0.009       0.025
T_wave_duration_ms          -0.0261      0.015     -1.735      0.083      -0.056       0.003
T_peak_signed_amplitude      0.0078      0.011      0.706      0.480      -0.014       0.029
============================================================================================



Ratio Model
Stress
Patients used in model: 97
ANOCA
1    63
0    34
Name: count, dtype: int64
Optimization terminated successfully.
         Current function value: 0.567706
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       91
Method:                           MLE   Df Model:                            5
Date:                Thu, 24 Sep 2026   Pseudo R-squ.:                  0.1236
Time:                        10:28:19   Log-Likelihood:                -55.068
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                  0.008317
============================================================================================
                               coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------------
const                        1.9656      3.254      0.604      0.546      -4.413       8.344
QRS_offset_to_T_onset_ms     0.0162      0.007      2.281      0.023       0.002       0.030
T_peak_signed_amplitude     -0.0182      0.007     -2.635      0.008      -0.032      -0.005
Age                         -0.0752      0.100     -0.754      0.451      -0.271       0.120
QRS_balance                  0.0061      0.003      2.103      0.035       0.000       0.012
HR_bpm                      -0.0206      0.021     -0.963      0.335      -0.063       0.021
============================================================================================



"""
#here we will try the difference in the three variables as opposed to just the 
#varaiables

fullDict['097']




"""
lead 1
ANOACA ON ABS DIFFERENCE 

ANOCA
1    63
0    34
Name: count, dtype: int64
Optimization terminated successfully.
         Current function value: 0.599287
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                  ANOCA   No. Observations:                   97
Model:                          Logit   Df Residuals:                       91
Method:                           MLE   Df Model:                            5
Date:                Thu, 17 Sep 2026   Pseudo R-squ.:                 0.07484
Time:                        02:26:51   Log-Likelihood:                -58.131
converged:                       True   LL-Null:                       -62.833
Covariance Type:            nonrobust   LLR p-value:                   0.09397
====================================================================================================
                                       coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------------
const                               -0.5619      1.485     -0.378      0.705      -3.473       2.349
QRS_offset_to_T_onset_ms             0.0111      0.010      1.131      0.258      -0.008       0.030
T_peak_signed_amplitude             -0.0124      0.012     -1.031      0.302      -0.036       0.011
QRS_offset_to_T_onset_ms_absdiff    -0.0468      0.030     -1.576      0.115      -0.105       0.011
T_wave_duration_ms_absdiff           0.0859      0.042      2.035      0.042       0.003       0.169
T_peak_signed_amplitude_absdiff     -0.0099      0.028     -0.350      0.727      -0.066       0.046
====================================================================================================


"""









"""



        
            relativePath = (recordName.relative_to("C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data"))
            relativePath =  relativePath.with_suffix("").as_posix()
            tempList = []
            tempList.append(patientID)
            tempList.append("/" + relativePath)
            for csvFile in file_path.iterdir():
                        contains_word = all(
                                word.lower() in (csvFile.name).lower()
                                        for word in words
                                    )
                        if contains_word == True:
                            with open(csvFile, "r") as file:
                                disease_data = json.load(file)
                            #output150 = turnTo150Output(disease_data['ANOCA'], 0)
                            output150 = turnTo150Output(disease_data['CMD'], 0)
                            #tempList.append(disease_data['ANOCA']) #CMD
                            tempList.append(disease_data['CMD'])
                            tempList.append(str(output150))
                            temListmeta.append(tempList)

            """

csv_path = Path(
    "C:/Users/Estif/Downloads/Langone/ANOCA/"
    "ECG_ANOCA_Trial_1/Models/ECGFounder-master/"
    "ECGFounder-master/data/ScienceDB_Data/sub_010/"
    "sub_010_baseline-data_rest-ecg-500hz.csv"
)




































"""

print("Maximum is",max(t_wave_data['T_peak_signed_amplitude']))


print("Minimum is",min(t_wave_data['T_peak_signed_amplitude']))


amp = t_wave_data["T_peak_signed_amplitude"]

print(amp.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print("Negative peaks:", (amp < 0).sum())
print("Positive peaks:", (amp > 0).sum())
print("Percent negative:", 100 * (amp < 0).mean())



signals, waves = nk.ecg_delineate(
    cleaned,
    sampling_rate=500,
    method="prominence"
)

cleaned = nk.ecg_clean(ecg, sampling_rate=500)



signals.shape ['ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
waves['ECG_T_Peaks']

['ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']

segment = cleaned[ton:toff + 1]

#Turn it into usable wave form db

# extract the T on and t off after delination



"""

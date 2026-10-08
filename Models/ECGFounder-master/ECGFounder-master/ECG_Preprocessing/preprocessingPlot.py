


"""
Here , we will use the philips dataset we got, to plot the same plots as the anocavs non anoca. 

First we must plot the philips datset and see the difference between the uncleaned vs cleaned. 

We can take samples like we did in the delinaiton step

"""


"""
Step 1  plot the raw philips dataset
"""
import os
import ishneholterlib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk


record = ishneholterlib.Holter("C:\\Users\\Estif\\Downloads\\Langone\\ANOCA\\ECG_ANOCA_Trial_1\\Models\\ECGFounder-master\\ECGFounder-master\\data\\SnippetECG_file\\ISHNE_VE_File.ecg")


record.load_data()

sampling_rate = record.sr


original_fs = sampling_rate   # Sampling rate in the ISHNE header
target_fs = 500  # Desired sampling rate

# Resample Signal to 500 from 256

def Plot(signals, sampling_rate, lead_names):
    duration = 100
    number_of_samples = min(
        signals.shape[0],
        sampling_rate * duration
    )
    time = np.arange(number_of_samples) / sampling_rate

    fig, axes = plt.subplots(
        signals.shape[1],
        1,
        figsize=(14, 2.5 * signals.shape[1]),
        sharex=True
    )

    axes = np.atleast_1d(axes)

    for index, axis in enumerate(axes):
        axis.plot(time, signals[:number_of_samples, index])
        axis.set_ylabel(lead_names[index])
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


def resampleData(lead_data, original_fs, target_fs):
    resampledData = nk.signal_resample(
        lead_data,  
        sampling_rate=original_fs,
        desired_sampling_rate=target_fs,
            method="poly"
        )
    return np.array(resampledData)



signals = np.column_stack(
    [resampleData(lead.data, original_fs, target_fs) for lead in record.lead]
)



sampling_rate = target_fs


print(sampling_rate)

lead_names = [str(lead) for lead in record.lead]
"""
signals = np.column_stack(
    [np.asarray(lead.data) for lead in record.lead]
)
"""
print("Sampling rate:", sampling_rate)
print("Lead names:", lead_names)
print("Signal shape:", signals.shape)


Plot(signals, sampling_rate, lead_names)


"""
Step 2 plot the cleaned philips dataset

"""

listOfNewLeads = []

fullInfoProcessed = []


for x in range(signals.shape[1]):
    print(x)
    firstlead = signals[:,x]
    processed_data, info = nk.bio_process(ecg=firstlead, sampling_rate=sampling_rate)
    cleanedSeries = processed_data["ECG_Clean"]
    print(cleanedSeries.shape)
    listOfNewLeads.append(cleanedSeries)
    fullInfoProcessed.append((processed_data, info))
    print(f"Finished processing lead number {x}")


len(listOfNewLeads)

f = np.array(listOfNewLeads)
print(f.shape)
f = f.T
print(f.shape)


Plot(f, sampling_rate, lead_names)


"""

Step 3, divide the above two into more two groups. So we would have the left half unitl 40 of the raw data vs the other half.

And for the cleaned data, we would have the left half unitl 40 seconds and then the remaining half.

"""

RawLeftHalf = signals[:sampling_rate*40, :]

RawLeftHalf_ecg = np.asarray(RawLeftHalf[:, 0], dtype=float)

Plot(RawLeftHalf, sampling_rate, lead_names)
RawLeftHalf.shape


RawRightHalf = signals[sampling_rate*40:, :]


RawRightHalf_ecg = np.asarray(RawRightHalf[:, 0], dtype=float)

Plot(RawRightHalf, sampling_rate, lead_names)

CleanedLeftHalf = f[:sampling_rate*40, :]

CleanedLeftHalf_ecg = np.asarray(CleanedLeftHalf[:, 0], dtype=float)

Plot(CleanedLeftHalf, sampling_rate, lead_names)

CleanedRightHalf = f[sampling_rate*40:, :]

CleanedRightHalf_ecg = np.asarray(CleanedRightHalf[:, 0], dtype=float)

Plot(CleanedRightHalf, sampling_rate, lead_names)


"""
Step 4, For the 4 groups, we will randomly choose 5 seconds (or something recognizable as an ecg)
then we will plot the ecgs themselves + delination and point recognition. 

The idea is to compare the raw vs clean data and it's associated deliniation. 
For this , we will just use prominence.


"""



"""
This is a check on an individual patient


"""

fs = 500
start = 5000            # starting sample (5000 / 500 = 10 s into the recording)
dur   = 2               # seconds to show

N = dur * fs
seg = slice(start, start + N)
t = np.arange(start, start + N) / fs     # x-axis in real recording time

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax[0].plot(t, RawLeftHalf_ecg[seg], linewidth=0.8, color="#888888")
ax[0].set_title("Raw")
ax[1].plot(t, CleanedLeftHalf_ecg[seg], linewidth=0.8, color="#1F3A5F")
ax[1].set_title("Cleaned")
ax[1].set_xlabel("Time (s)")

for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()

"""
Here first do it for the raw data
"""

_, rpeak_info = nk.ecg_peaks(RawLeftHalf_ecg, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]

pre, post = int(0.25*fs), int(0.60*fs)
beats = np.array([RawLeftHalf_ecg[r-pre:r+post] for r in rpeaks
                  if r-pre >= 0 and r+post < len(RawLeftHalf_ecg)])
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
    RawLeftHalf_ecg,
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

res = windowed_analysis(RawLeftHalf_ecg, rpeaks, fs=fs, window_s=60, method="dwt")



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

_, rpeak_info = nk.ecg_peaks(CleanedLeftHalf_ecg, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]

_, waves_dwt = nk.ecg_delineate(CleanedLeftHalf_ecg, rpeaks=rpeaks,
                                sampling_rate=fs, method="prominence")

plot_delineation(CleanedLeftHalf_ecg, waves_dwt, rpeaks, start=10000, dur=5, fs=fs,
                 title="prominence on raw signal Right side")
plt.tight_layout(); plt.show()


"""
Step 5 , for this , we can plot a per beat patterns without aggregation. 
So for the philips, we can do one point in a scatterplot for example to represent the qrs duration. For the whole thing. 

"""

import pandas as pd

def getBeatData(csvFile, leadNumber = 2, method = "prominence", ecgIn = False):
   
    #ecg = ecgIn

    
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
        #correct_artifacts=True,
        sampling_rate=500,
        
    )

    #cleaned = ecgIn

    # Delineate waves
    signals, waves = nk.ecg_delineate(
        cleaned,
        rpeaks=rpeak_info["ECG_R_Peaks"],
        sampling_rate=500,
        method=method,
        max_p_basepoint_interval=100,   # default 100 ms
        max_r_basepoint_interval=200,   # default 100 ms
        max_t_basepoint_interval=300
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
    tt = nk.hrv_time(rpeak_info, sampling_rate=500)
    #print("This is tt two values")
    #print(tt["HRV_RMSSD"].iloc[0] , tt["HRV_MedianNN"].iloc[0])
    beat_data["HRV_RMSSD"] = tt["HRV_RMSSD"].iloc[0]
    #print(beat_data["HRV_RMSSD"])
    beat_data["HRV_MedianNN"] = tt["HRV_MedianNN"].iloc[0]


    return beat_data



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






from pathlib import Path

csv_path = Path(
    "C:/Users/Estif/Downloads/Langone/ANOCA/"
    "ECG_ANOCA_Trial_1/Models/ECGFounder-master/"
    "ECGFounder-master/data/ScienceDB_Data/sub_010/"
    "sub_010_baseline-data_rest-ecg-500hz.csv"
)


# Ge the full data 


signals = np.column_stack(
    [resampleData(lead.data, original_fs, target_fs) for lead in record.lead]
)

signals.shape

signals = signals[20_000:, ]  

signals.shape

PhilipsBeatData = getBeatData(csv_path, ecgIn=signals[:, 0])


PhilipsBeatData.keys()

import numpy as np, matplotlib.pyplot as plt
d = PhilipsBeatData
beat = range(len(d))

def with_box(col, figsize=(11, 4)):
    fig, (ax, bx) = plt.subplots(1, 2, sharey=True, figsize=figsize,
                                 gridspec_kw={"width_ratios": [5, 1], "wspace": .05})
    bx.boxplot(d[col].dropna(), widths=.5)
    bx.set_xticks([])
    return ax

# 1. QT against heart rate
ax = with_box("QT_interval_ms", figsize=(9, 5))
hr = np.linspace(d.HR_bpm.min(), d.HR_bpm.max(), 100)
ax.scatter(d.HR_bpm, d.QT_interval_ms, s=8, alpha=.5)
ax.plot(hr, 440 * np.sqrt(60 / hr), "r--", label="QTc = 440 ms (Bazett)")
ax.set(xlabel="Heart rate (bpm)", ylabel="QT interval (ms)", title="QT interval against heart rate")
ax.legend()

# 2. QRS duration per beat
ax = with_box("QRS_onset_to_QRS_offset_ms")
ax.scatter(beat, d.QRS_onset_to_QRS_offset_ms, s=6, alpha=.5)
ax.axhline(120, color="r", ls="--", label="Upper normal limit (120 ms)")
ax.set(xlabel="Beat number", ylabel="QRS duration (ms)", title="QRS duration per beat")
ax.legend()

# 3. PR interval per beat
ax = with_box("PR_interval_ms")
ax.scatter(beat, d.PR_interval_ms, s=6, alpha=.5)
ax.axhspan(120, 200, color="green", alpha=.1, label="Normal range (120–200 ms)")
ax.set(xlabel="Beat number", ylabel="PR interval (ms)", title="PR interval per beat")
ax.legend()

plt.show()

"""
Now we will calculate the non specific t wave abnormality and plot the mean
over the variance. 

"""

PhilipsBeatData["T_over_QRS_balance"]

Non_Specific_T_Wave_Abnormality = []

fs = 500
window = 30 * fs                      # 15,000 samples

for start in range(0, signals.shape[0] - window + 1, window):
    segment = signals[start:start + window]
    try:
        TempPhilipsBeatData = getBeatData(csv_path, ecgIn=segment[:, 0])
    except (IndexError, ValueError) as e:
        print(f"Window at {start / fs:.0f} s skipped: {e}")
        continue
    if TempPhilipsBeatData.empty or "T_over_QRS_balance" not in TempPhilipsBeatData:
        print(f"No beats detected in window at {start / fs:.0f} s, skipped")
        continue
    print(TempPhilipsBeatData["T_over_QRS_balance"].median())
    print(TempPhilipsBeatData["HRV_RMSSD"])
    print(TempPhilipsBeatData["T_over_QRS_balance"].var())
    Non_Specific_T_Wave_Abnormality.append((TempPhilipsBeatData["T_over_QRS_balance"].median(),TempPhilipsBeatData["T_over_QRS_balance"].var() ))


Non_Specific_T_Wave_Abnormality = np.array(Non_Specific_T_Wave_Abnormality)

Non_Specific_T_Wave_Abnormality

import numpy as np, matplotlib.pyplot as plt

results = Non_Specific_T_Wave_Abnormality         # rows: [median, variance], one per window
plt.figure(figsize=(8, 5))
sc = plt.scatter(results[:, 1], results[:, 0], c=np.arange(len(results)) * 0.5, cmap="viridis", s=15)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.xlabel("Variance of T/QRS balance within window")
plt.ylabel("Median T/QRS balance within window")
plt.title("T/QRS balance per 30-second window, single participant")
plt.show()



hrv = []

fs = 500
window = 30 * fs                      # 15,000 samples

for start in range(0, signals.shape[0] - window + 1, window):
    segment = signals[start:start + window]
    #print(segment.shape)
    try:
        TempPhilipsBeatData = getBeatData(csv_path, ecgIn=segment[:, 0])
    except (IndexError, ValueError) as e:
        #print(f"Window at {start / fs:.0f} s skipped: {e}")
        continue
    if TempPhilipsBeatData.empty or "HRV_RMSSD" not in TempPhilipsBeatData:
        #print(f"No beats detected in window at {start / fs:.0f} s, skipped")
        continue
    print(TempPhilipsBeatData["HRV_MedianNN"].median(), "1") #"HRV_MedianNN", "HRV_RMSSD"
    print(TempPhilipsBeatData["HRV_RMSSD"].median(), "2")
    hrv.append((TempPhilipsBeatData["HRV_MedianNN"].median(),TempPhilipsBeatData["HRV_RMSSD"].median()))

hrv

("HRV_MedianNN", "HRV_RMSSD")

import numpy as np, matplotlib.pyplot as plt

hrv_results = np.array(hrv)        # rows: [HRV_MedianNN, HRV_RMSSD], one per window
plt.figure(figsize=(8, 5))
sc = plt.scatter(hrv_results[:, 1], 60000 / hrv_results[:, 0], c=np.arange(len(hrv_results)) * 0.5,
                 cmap="viridis", s=15)


#plt.xlabel("RMSSD within window (ms)")
plt.xlabel("Root mean square of successive differences (RMSSD) within window (ms)")
#plt.ylabel("Median RR interval within window (ms)")
plt.ylabel("Median heart rate within window (bpm)")
plt.title("Heart rate variability per 30-second window\n"
          "Single participant")
plt.show()






P_duration_ms = []


fs = 500
window = 30 * fs                      # 15,000 samples

for start in range(0, signals.shape[0] - window + 1, window):
    segment = signals[start:start + window]
    try:
        TempPhilipsBeatData = getBeatData(csv_path, ecgIn=segment[:, 0])
    except (IndexError, ValueError) as e:
        print(f"Window at {start / fs:.0f} s skipped: {e}")
        continue
    if TempPhilipsBeatData.empty or "T_over_QRS_balance" not in TempPhilipsBeatData:
        print(f"No beats detected in window at {start / fs:.0f} s, skipped")
        continue
    print(TempPhilipsBeatData["P_duration_ms"].median())
    print(TempPhilipsBeatData["P_duration_ms"].var())
    P_duration_ms.append((TempPhilipsBeatData["P_duration_ms"].median(),TempPhilipsBeatData["P_duration_ms"].var() ))


P_duration_ms 

import numpy as np, matplotlib.pyplot as plt

p_results = np.array(P_duration_ms)          # rows: [median P duration, variance], one per window
plt.figure(figsize=(8, 5))
sc = plt.scatter(p_results[:, 1], p_results[:, 0], c=np.arange(len(p_results)) * 0.5,
                 cmap="viridis", s=15)
plt.colorbar(sc, label="Window start (minutes into recording)")
plt.axhline(120, color="r", ls="--", lw=1, label="Upper normal limit (120 ms)")
plt.xlabel("Variance of P-wave duration within window (ms²)")
plt.ylabel("Median P-wave duration within window (ms)")
plt.title("P-wave duration per 30-second window\nSingle participant")
plt.legend()
plt.show()


QRS_onset_to_QRS_offset_ms = []


for start in range(0, signals.shape[0] - window + 1, window):
    segment = signals[start:start + window]
    try:
        TempPhilipsBeatData = getBeatData(csv_path, ecgIn=segment[:, 0])
    except (IndexError, ValueError) as e:
        print(f"Window at {start / fs:.0f} s skipped: {e}")
        continue
    if TempPhilipsBeatData.empty or "QRS_onset_to_QRS_offset_ms" not in TempPhilipsBeatData:
        print(f"No beats detected in window at {start / fs:.0f} s, skipped")
        continue
    print(TempPhilipsBeatData["QRS_onset_to_QRS_offset_ms"].median())
    print(TempPhilipsBeatData["QRS_onset_to_QRS_offset_ms"].var())
    QRS_onset_to_QRS_offset_ms.append((TempPhilipsBeatData["QRS_onset_to_QRS_offset_ms"].median(),TempPhilipsBeatData["QRS_onset_to_QRS_offset_ms"].var() ))

QRS_onset_to_QRS_offset_ms

import numpy as np, matplotlib.pyplot as plt

qrs_results = np.array(QRS_onset_to_QRS_offset_ms)        # rows: [median QRS duration, variance], one per window
plt.figure(figsize=(8, 5))
sc = plt.scatter(qrs_results[:, 1], qrs_results[:, 0], c=np.arange(len(qrs_results)) * 0.5,
                 cmap="viridis", s=15)
plt.colorbar(sc, label="Window start (minutes into recording)")
plt.axhline(120, color="r", ls="--", lw=1, label="Upper normal limit (120 ms)")
plt.xlabel("Variance of QRS duration within window (ms²)")
plt.ylabel("Median QRS duration within window (ms)")
plt.title("QRS duration per 30-second window\nSingle participant")
plt.legend()
plt.show()


"""
Here we will filter the normal people IE the controls and then plot the the same thing as the above 

"""



import json
import pandas as pd

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
        words = ["baseline","data", "rest", "500hz" ]
        #words = ["data", "recover", "500hz" ]
        #words = ["data", "stress", "500hz" ] #"baseline",
        words_Disease = ['disease' , 'json']
        words_chief_complaint = ["complaint", "chief", "baseline" ] # sub_001_baseline-data_chief-complaint
        disease_file = next((f for f in file_path.iterdir()
                             if all(w in f.name.lower() for w in words_Disease)), None)
        if disease_file is None:
            continue                                    # no disease file: skip patient
        with open(disease_file) as f:
            disease_data = json.load(f)
        if int(disease_data["ANOCA"]) != 0:
            continue   
        fullDict.setdefault(pid, {})          #  CHANGED: create the entry once
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
                "Here it can be replaced by a function return beat_data"
                beat_data = getBeatData(csvFile,method = "prominence")
                # Absolute first-order difference between successive beats
                exclude = ["QRS_balance", "T_over_QRS_balance", "HRV_MedianNN", "HRV_RMSSD"]      # whichever you don't want
                abs_diff_data = beat_data.drop(columns=exclude).diff().abs().iloc[1:] 
                # here add IQR beat_data.quantile(.75) - beat_data.quantile(.25)
                #Here print the IQR 
                
                print("Here is the IQR")
                iqr_features = (
                    beat_data.quantile(0.75) - beat_data.quantile(0.25)
                ).add_suffix("_IQR").to_frame().T

                var_features = (
                    beat_data.drop(columns=["HR_bpm", "HRV_RMSSD", "HRV_MedianNN"]).var()
                ).add_suffix("_var").to_frame().T

                print(iqr_features)
                abs_diff_data = beat_data.drop(columns=exclude).diff().abs().iloc[1:]
                # Median of the original features
                patient_features = beat_data.median().to_frame().T
                print(patient_features.keys())
                # Median of the absolute differences
                diff_features = abs_diff_data.median().add_suffix("_absdiff").to_frame().T

                patient_features = pd.concat([patient_features, diff_features, iqr_features, var_features], axis=1)
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
df = pd.DataFrame.from_dict(all_Lead_Data[5], orient="index")   # CHANGED: one row per patient
print(df )

df = df[df['CMD'] == 0]

df.keys() 

"""
Plot the same things , first the non specific t wave abnormality
per subject

"""
df.shape[0]
for x in range(df.shape[0]):
    print(df.iloc[x]["T_over_QRS_balance_var"])


df.keys()

import numpy as np, matplotlib.pyplot as plt, pandas as pd

df_all = pd.DataFrame.from_dict(fullDict, orient="index")    # one row per participant

def plot_feature(col, label, units=""):
    d = df_all[[col, f"{col}_var"]].dropna()                     # participants with values for this feature
    u = f" ({units})" if units else ""
    u2 = f" ({units}²)" if units else ""

    plt.figure(figsize=(8, 5))
    plt.scatter(d[f"{col}_var"], d[col], s=25, alpha=.8)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel(f"Variance of {label} across 10-second median beats{u2}")
    plt.ylabel(f"Median {label}{u}")
    plt.title(f"{label} per participant, lead II (n = {len(d)})\n"
              "Each point = one participant, 6-minute resting ECG")
    plt.show()

plot_feature("QTcF_ms", "QTcF", "ms")
plot_feature("QT_interval_ms", "QT interval", "ms")
plot_feature("T_over_QRS_balance", "T/QRS balance")
plot_feature("T_peak_signed_amplitude", "T-wave amplitude")

#New ones
plot_feature("PR_interval_ms", "PR interval", "ms")
plot_feature("QRS_offset_to_T_onset_ms", "ST segment duration", "ms")

#==========================

def plot_relation(x, y, xlabel, ylabel, title):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    plt.figure(figsize=(8, 5))
    plt.scatter(d["x"], d["y"], s=25, alpha=.8)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}, lead II (n = {len(d)})\n"
              "Each point = one participant, 6-minute resting ECG")
    return d

# 3. QT vs heart rate, with Fridericia curve
hr = 60000 / df_all["HRV_MedianNN"]
plot_relation(hr, df_all["QT_interval_ms"], "Heart rate (bpm)", "Median QT interval (ms)",
              "QT interval vs heart rate")
h = np.linspace(hr.min(), hr.max(), 100)
plt.plot(h, df_all["QTcF_ms"].median() * (60 / h) ** (1/3), "r--",
         label=f"Fridericia curve (QTcF = {df_all['QTcF_ms'].median():.0f} ms)")
plt.legend(); plt.show()

# 4. QTcF vs age
plot_relation(df_all["Age"], df_all["QTcF_ms"], "Age (years)", "Median QTcF (ms)",
              "QTcF vs age")
plt.show()






#==========================
def plot_hr_vs_rmssd():
    plt.figure(figsize=(8, 5))
    plt.scatter(df_all["HRV_RMSSD"], 60000 / df_all["HRV_MedianNN"], s=25, alpha=.8)
    plt.xlabel("Root mean square of successive differences (RMSSD) across the recording (ms)")
    plt.ylabel("Heart rate from median RR interval (bpm)")
    plt.title(f"Heart rate vs. heart rate variability, lead II (n = {len(df_all)})\n"
              "Each point = one participant, 6-minute resting ECG")
    plt.show()

plot_hr_vs_rmssd()

import numpy as np, matplotlib.pyplot as plt, pandas as pd

df_all = pd.DataFrame.from_dict(fullDict, orient="index")    # one row per participant

plt.figure(figsize=(8, 5))
plt.scatter(df_all["HRV_RMSSD"], 60000 / df_all["HRV_MedianNN"], s=25, alpha=.8)
plt.xlabel("Root mean square of successive differences (RMSSD) across the recording (ms)")
plt.ylabel("Heart rate from median RR interval (bpm)")
plt.title(f"Heart rate vs. heart rate variability (n = {len(df_all)} participants)\n"
          "Each point = one participant, 6-minute resting ECG, lead II")
plt.show()

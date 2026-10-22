import numpy as np
import ishneholterlib
import matplotlib.pyplot as plt
import neurokit2 as nk


trialFile = np.load("/gpfs/data/pmedlab/Cohorts/NYU/ECG/autoimmune_ecg/CA3299024.npy", allow_pickle=True)

type (trialFile)

print(trialFile.shape)

import pandas as pd 


x = pd.DataFrame(trialFile)


"""

Here we will cycle through all the data contained in the 
NYU FIle.

"""
from pathlib import Path
folder_path = Path("/gpfs/data/pmedlab/Cohorts/NYU/ECG/autoimmune_ecg/")


pattern = ".npy"

#print(text.endswith(pattern))


for file_path in folder_path.iterdir():

    print(file_path.name)



sampleFilePath = folder_path / "XD213310.npy"  


#sampleFilePath = folder_path / "CA1013018.npy"

sampleFile = np.load(sampleFilePath)

sampleFile.shape

ecg = np.asarray(ecg_signal[:, 0], dtype=float)



sampleFileCleaned = nk.ecg_clean(sampleFile[0, :], sampling_rate=500)



"""
A Pure Check
"""


import matplotlib.pyplot as plt
fs = 500
start = 0            # starting sample (5000 / 500 = 10 s into the recording)
dur   = 5               # seconds to show

N = dur * fs
seg = slice(start, start + N)
t = np.arange(start, start + N) / fs     # x-axis in real recording time

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax[0].plot(t, (sampleFile[0, :])[seg], linewidth=0.8, color="#888888")
ax[0].set_title("Raw")
ax[1].plot(t, sampleFileCleaned[seg], linewidth=0.8, color="#1F3A5F")
ax[1].set_title("Cleaned")
ax[1].set_xlabel("Time (s)")

for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()

_, rpeak_info = nk.ecg_peaks(sampleFileCleaned, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]

pre, post = int(0.25*fs), int(0.60*fs)
beats = np.array([sampleFileCleaned[r-pre:r+post] for r in rpeaks
                  if r-pre >= 0 and r+post < len(sampleFileCleaned)])
median_beat = np.median(beats, axis=0)
t_ms = (np.arange(-pre, post)) * 1000 / fs

plt.figure(figsize=(10, 5))
for b in beats[::10]:                      # a sample of individual beats
    plt.plot(t_ms, b, color="#BBBBBB", linewidth=0.4)
plt.plot(t_ms, median_beat, color="#B03A2E", linewidth=2.5, label=f"median of {len(beats)} beats")
plt.axvline(0, color="#1F3A5F", linestyle="--", linewidth=1)
plt.xlabel("Time from R peak (ms)"); plt.legend(); plt.grid(alpha=0.3)
plt.show()


#================================ Here it is just for plotting 

import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np

fs = 500

# Landmark: (key in waves, label, color, marker)
LANDMARKS = [
    #("ECG_P_Onsets",  "P on",   "#7F8C8D", "|"),
    #("ECG_P_Peaks",   "P",      "#7F8C8D", "o"),
    ("ECG_S_Peaks", "S Peak",  "#FD0F0F", ">"),
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

_, rpeak_info = nk.ecg_peaks(sampleFileCleaned, sampling_rate=fs)
rpeaks = rpeak_info["ECG_R_Peaks"]


signals, info = nk.ecg_process(sampleFileCleaned, sampling_rate=fs, method='neurokit')


#======== Check inversion

# Separate copy for detection; retain original polarity.
detect, flipped = nk.ecg_invert(sampleFileCleaned, sampling_rate=fs)

_, info = nk.ecg_peaks(detect, sampling_rate=fs)

_, waves = nk.ecg_delineate(
    detect,
    rpeaks=info["ECG_R_Peaks"],
    sampling_rate=fs,
    method="prominence",
    show=True,
    show_type="all",
)
#========



#nk.ecg_analyze(sampleFileCleaned, sampling_rate=fs, method='auto')

#
#_, waves_dwt = nk.ecg_delineate(sampleFileCleaned, rpeaks=rpeaks,
##                                sampling_rate=fs,method="prominence" ) # , method = "dwt"

#info.keys()

plot_delineation(sampleFileCleaned, info, rpeaks, start=0, dur=5, fs=fs,
                 title="Neurokit method Lead ??")
plt.tight_layout(); plt.show()

#==================

_, rpeak_info = nk.ecg_peaks(
    sampleFileCleaned,
    sampling_rate=500
)

len(rpeak_info['ECG_R_Peaks'])

# Delineate waves
signals, waves = nk.ecg_delineate(
    sampleFileCleaned,
    rpeaks=rpeak_info["ECG_R_Peaks"],
    sampling_rate=500,
    method="prominence"
    #method = "dwt"
)


#After this jumpt to the NEW part

# T-peak sample locations
t_peaks = np.asarray(waves["ECG_T_Peaks"], dtype=float)
t_peaks = t_peaks[np.isfinite(t_peaks)].astype(int)

print(t_peaks)
print("Number of T peaks:", len(t_peaks))


fs = 500
cleaned = np.asarray(sampleFileCleaned).squeeze()

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

#================================== Here it is just for plotting
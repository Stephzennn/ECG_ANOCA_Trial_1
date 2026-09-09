
import os
import ishneholterlib
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk



record = ishneholterlib.Holter("C:\\Users\\Estif\\Downloads\\Langone\\ANOCA\\ECG_ANOCA_Trial_1\\Models\\ECGFounder-master\\ECGFounder-master\\data\\SnippetECG_file\\ISHNE_VE_File.ecg")

print(record)

record.load_data()

sampling_rate = record.sr


original_fs = sampling_rate   # Sampling rate in the ISHNE header
target_fs = 100  # Desired sampling rate

# Resample Signal to 500 from 256

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


"""
for x in range(len(f)):
    print(f[x])
    #print(type(f[x]))
    print(f[x].shape)
    break


type(signals)

signals.shape

"""

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


Plot(f, sampling_rate, lead_names)

# Extract information  

"""
cleaned 6-lead waveform (which can be used in functional waveform analyses beyond the pre-specified feature-based analyses if useful) ###DONE
exact recording start date and time ###DONE
heart rate ###DONE
R-R intervals and short-term R-R variability ###DONE

recording-level signal-quality metric or usability flag
ST-segment elevation and depression by lead
T-wave amplitude and inversion by lead
QT interval
rhythm abnormalities
conduction abnormalities, including bundle-branch block


"""
# Preproccesing for Kardia 


#Save example files for 6 lead kardia export in pdf format. 


"""
THE BELOW DESCRIPTION IF TAKEN FROM THE GITHUB joelparkerhenderson/kardiamobile-6l-ecg-convert-pdf-to-edf-python

KardiaMobile 6-lead file format converter from Kardia ECG PDF to EDF.

This script reads the vector path data embedded in the PDF, converts
the path coordinates to voltage values using the known calibration
(10mm/mV, 25mm/s), and writes the result as a standard EDF file.

Metadata extracted from PDF:
  - Patient: Joel Henderson
  - DOB: 5/4/70, Sex: Male, Age: 55
  - Recorded: 2026-02-23 at 18:26:56
  - Heart Rate: 87 BPM
  - Duration: 30s
  - 6 leads: I, II, III, aVR, aVL, aVF
  - Sampling rate: 300 Hz
  - Kardia Determination: Normal Sinus Rhythm
"""

from datetime import datetime

import numpy as np
import pyedflib
import pymupdf




def extract_baselines(page):
    """Extract the baseline y-coordinates for each lead from grid lines."""
    paths = page.get_drawings()
    for path in paths:
        items = path.get("items", [])
        color = path.get("color")
        width = path.get("width", 0)
        # The baseline path has many horizontal lines and width ~0.4
        if (
            color == (0.0, 0.0, 0.0)
            and width is not None
            and 0.35 < width < 0.45
            and len(items) >= 6
        ):
            # Check if all items are horizontal lines spanning the page
            y_values = []
            for item in items:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 0.01 and abs(p2.x - p1.x) > 500:
                        y_values.append(p1.y)
            if len(y_values) >= 6:
                return y_values[:6]  # First 6 baselines = 6 leads
    return None



def extract_ecg_paths_from_page(page, baselines):
    """Extract ECG waveform points for each lead from a single page.

    Returns dict: lead_index -> list of (x, y) points sorted by x.
    """
    paths = page.get_drawings()
    leads = {i: [] for i in range(6)}

    for path in paths:
        color = path.get("color")
        width = path.get("width", 0)
        items = path.get("items", [])

        # ECG waveform paths: black, width ~0.4, many line segments
        if (
            color != (0.0, 0.0, 0.0)
            or width is None
            or not (0.35 < width < 0.45)
            or len(items) < 50
        ):
            continue

        # Extract points from line segments
        points = []
        for item in items:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if (
                    not points
                    or abs(points[-1][0] - p1.x) > 0.001
                    or abs(points[-1][1] - p1.y) > 0.001
                ):
                    points.append((p1.x, p1.y))
                points.append((p2.x, p2.y))

        if not points:
            continue

        # Determine which lead by y-center proximity to baselines
        y_center = np.mean([p[1] for p in points])
        min_dist = float("inf")
        best_lead = -1
        for li, bl in enumerate(baselines):
            dist = abs(y_center - bl)
            if dist < min_dist:
                min_dist = dist
                best_lead = li

        if best_lead >= 0 and min_dist < 50:
            leads[best_lead].extend(points)

    # Sort each lead's points by x-coordinate
    for li in leads:
        leads[li].sort(key=lambda p: p[0])

    return leads

def points_to_voltage(points, baseline_y, cal_pt_per_mv):
    """Convert (x, y) points to voltage values in millivolts.

    In the PDF, y increases downward, so voltage = (baseline - y) / scale.
    """
    voltages = [(baseline_y - y) / cal_pt_per_mv for _, y in points]
    return voltages



def Turn_PDF_to_EDF(pdf_path,
    edf_path,
    kardia_6l_defaults = False,
    baseline_page = 1):

    """
    Here we will have default values for Kardia exports
    """

    # Here we will have kardia default values, if they need to be changed , change them.

    KARDIA_6L_DEFAULTS = {
        "sample_rate_hz": 300,
        "paper_speed_mm_s": 25.0,
        "gain_mm_per_mv": 10.0,
        "frequency_low_hz": 0.5,
        "frequency_high_hz": 40.0,
        "input_range_mv_peak_to_peak": 10.0,
        "default_duration_seconds": 30,
        "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF"],
        "measured_leads": ["I", "II"],
        "derived_leads": ["III", "aVR", "aVL", "aVF"],
        "pdf_points_per_mm": 2.834645669,
        "pdf_points_per_mv": 28.346456693,
        "pdf_points_per_second": 70.866141732,
        "seconds_per_pdf_point": 0.014111111,
        "samples_per_mm": 12.0,
        "mm_per_sample": 0.083333333,
        "pdf_points_per_sample": 0.236220472,
        "input_min_mv": -5.0,
        "input_max_mv": 5.0,
        "default_samples_per_lead": 9000,
    }
    kardia_6l_defaults = KARDIA_6L_DEFAULTS




    doc = pymupdf.open(pdf_path)

    first_page_text = doc[0].get_text("text")

    first_page_lines = first_page_text.split("\n")

    dob = datetime.strptime(first_page_lines[(first_page_lines.index('Date of Birth') + 1)], "%m/%d/%y")
    
    Kardia_Determination = (first_page_lines[(first_page_lines.index('Kardia Determination:') + 1)]).replace(" ", "_")

    HeartRate = (first_page_lines[(first_page_lines.index('Heart Rate:') + 1)]).replace(" ", "_")

    Kardia_Determination_HR_Combined = Kardia_Determination + '_HR_' + HeartRate

    name = first_page_lines[(first_page_lines.index('EKG Recording Overview') - 1)]

    sex =  1 if (first_page_lines[(first_page_lines.index('Sex') + 1)] == 'Male') else 0 

    startDateTime =  first_page_lines[(first_page_lines.index('Recorded on:')) + 1: (first_page_lines.index('Recorded on:')) + 2 ]

    newStartDatetime0 = startDateTime[0].replace(",", "")

    newStartDatetime = newStartDatetime0.split(" ")

    hours = newStartDatetime[-1]

    # Normalize the narrow non-breaking space
    time_text = hours.replace("\u202f", " ").strip()
    parsed_time = datetime.strptime(time_text, "%I:%M:%S %p")

    hour = parsed_time.hour
    minute = parsed_time.minute
    second = parsed_time.second
    day = int(newStartDatetime[2])
    month_number = datetime.strptime(newStartDatetime[1], "%B").month
    year = datetime.strptime(newStartDatetime[-3], "%Y").year
    number_of_pages = doc.page_count
    ecg_page_indices = range(baseline_page, number_of_pages)  # Pages with ECG waveforms
    # Calibration: 1 mV = 28.346 PDF points (10mm at 2.8346 pt/mm)
    CAL_PT_PER_MV = kardia_6l_defaults["pdf_points_per_mv"]
    SAMPLE_RATE = kardia_6l_defaults['sample_rate_hz'] # Hz This part has to be 
    LEAD_NAMES = kardia_6l_defaults['lead_names'] 

    # Extract baselines from page 2 (index 1)
    baselines = extract_baselines(doc[baseline_page])
    if not baselines:
        raise RuntimeError("Could not find baseline grid lines in PDF")

    print(f"Baselines (PDF y-coordinates): {[f'{b:.3f}' for b in baselines]}")

    # Extract waveforms from all ECG pages (pages 2-5, indices 1-4)
    all_leads = {i: [] for i in range(6)}
    for pg_idx in ecg_page_indices:
        page = doc[pg_idx]
        page_leads = extract_ecg_paths_from_page(page, baselines)
        for li in range(6):
            all_leads[li].extend(page_leads[li])

    doc.close()

    # Convert to voltage arrays
    lead_voltages = {}
    for li in range(6):
        points = all_leads[li]
        # Remove duplicate x-coordinates (boundary points between segments)
        deduped = [points[0]]
        for i in range(1, len(points)):
            if abs(points[i][0] - deduped[-1][0]) > 0.01:
                deduped.append(points[i])
        voltages = points_to_voltage(deduped, baselines[li], CAL_PT_PER_MV)
        lead_voltages[li] = np.array(voltages, dtype=np.float64)
        print(
            f"Lead {LEAD_NAMES[li]:3s}: {len(voltages)} samples, "
            f"range [{min(voltages):.3f}, {max(voltages):.3f}] mV"
        )

    # Ensure all leads have the same length (trim to shortest)
    min_len = min(len(lead_voltages[li]) for li in range(6))
    for li in range(6):
        lead_voltages[li] = lead_voltages[li][:min_len]

    duration_sec = min_len / SAMPLE_RATE
    print(f"\nTotal samples per lead: {min_len}")
    print(f"Duration: {duration_sec:.2f} seconds")
    print(f"Sampling rate: {SAMPLE_RATE} Hz")

    # Write EDF file
    n_channels = 6
    """
    edf_writer = pyedflib.EdfWriter(
        edf_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
    )
    """
    edf_writer = pyedflib.EdfWriter(
        edf_path,
        n_channels,
        file_type=pyedflib.FILETYPE_EDF,
    )

    # Patient and recording info
    edf_writer.setPatientName(name) #Done /
    edf_writer.setSex(sex)  # 1 = male #Done /
    edf_writer.setBirthdate(datetime(dob.year, dob.month, dob.day).date()) #Done /
    edf_writer.setStartdatetime(datetime(year, month_number, day, hour, minute,second)) #Done /
    edf_writer.setEquipment("KardiaMobile_6L") # Keep hard coded
    edf_writer.setRecordingAdditional(Kardia_Determination_HR_Combined) #Done /


    # Configure channels
    for li in range(n_channels):
        edf_writer.setLabel(li, f"EKG {LEAD_NAMES[li]}")
        edf_writer.setPhysicalDimension(li, "mV")
        edf_writer.setSamplefrequency(li, SAMPLE_RATE)
        edf_writer.setTransducer(li, "KardiaMobile 6L electrode")
        edf_writer.setPrefilter(li, "Enhanced Filter, 50Hz mains")

        # Set physical min/max from actual data with margin
        phys_min = float(np.min(lead_voltages[li])) - 0.1
        phys_max = float(np.max(lead_voltages[li])) + 0.1
        edf_writer.setPhysicalMinimum(li, phys_min)
        edf_writer.setPhysicalMaximum(li, phys_max)
        edf_writer.setDigitalMinimum(li, -32768)
        edf_writer.setDigitalMaximum(li, 32767)

    # Write data
    edf_writer.writeSamples([lead_voltages[li] for li in range(n_channels)])
    edf_writer.close()

    print(f"\nEDF file written: {edf_path}")
    print(f"File size: {__import__('os').path.getsize(edf_path):,} bytes")

    return



pdf_path = r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\data\Kardia_Sample_Data\kardiamobile-6l-ecg.pdf"
edf_path = r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\data\Kardia_Sample_Data\kardiamobile-6-lead-3ecg.edf"

#Turn_PDF_to_EDF(pdf_path,edf_path)

#Sanity Check, read from edf file 

import wfdb
from wfdb.io.convert import read_edf

#edf_path = r"C:\path\to\Joel_Henderson.edf"

record = read_edf(edf_path)

print(record)

print("Record name:", record.record_name)
print("Sampling rate:", record.fs)
print("Channels:", record.n_sig)
print("Lead names:", record.sig_name)
print("Units:", record.units)
print("Samples:", record.sig_len)
print("Duration:", record.sig_len / record.fs)
print("Start date:", record.base_date)
print("Start time:", record.base_time)
print("Comments:", record.comments)
print("Signal shape:", record.p_signal.shape)


# After this , apply neuro kit

signals = record.p_signal

sampling_rate = record.fs

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


lead_names = [str(lead) for lead in record.sig_name]
Plot(f, sampling_rate,lead_names)



signals, info = nk.ecg_process(signals[:,1], sampling_rate=sampling_rate)


info.keys()
len(info['ECG_fixpeaks_rr'])

len(info['ECG_R_Peaks'])


#This part is the R R interval variablility


hrv_results = nk.hrv(
    signals,
    sampling_rate=sampling_rate,
    show=True,
)


fig = plt.gcf()
fig.set_dpi(150)
fig.set_size_inches(26, 20)

plt.tight_layout()
plt.show()





#nk.hrv(signals, sampling_rate=sampling_rate, show=True )

from pathlib import Path

import os
from pathlib import Path

mcr_path = Path(
    r"C:\Program Files\MATLAB\MATLAB Runtime\v910"
)

mcr_directories = [
    mcr_path / "runtime" / "win64",
    mcr_path / "bin" / "win64",
    mcr_path / "sys" / "os" / "win64",
    mcr_path / "extern" / "bin" / "win64",
]

for directory in mcr_directories:
    print(directory, directory.exists())

    if directory.exists():
        os.environ["PATH"] = (
            str(directory)
            + os.pathsep
            + os.environ.get("PATH", "")
        )

        # Keep returned handle alive in this Python process
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(directory))


runtime_dll = (
    mcr_path
    / "runtime"
    / "win64"
    / "mclmcrrt9_10.dll"
)

print(runtime_dll)
print(runtime_dll.exists())

from pathlib import Path
from pecg.ecg.wavedet_exe import Wavdet

wavedet_exe = (
    Path(Wavdet.__file__).parent
    / "peak_det_2023.exe"
)

print(wavedet_exe)
print(wavedet_exe.exists())




matlab_runtime_path = Path(
    r"C:\Program Files\MATLAB\MATLAB Runtime\v910"
)

print(matlab_runtime_path.exists())

import pecg 
import numpy as np
from pecg.ecg import Biomarkers as Bm
from pecg.ecg import FiducialPoints as Fp

# This is the location of matlab C:\Program Files\MATLAB\MATLAB Runtime

signals = np.asarray(signals, dtype=np.float64)

print(signals.shape)
print("Duration:", signals.shape[0] / sampling_rate, "seconds")

fp = Fp.FiducialPoints(signals, sampling_rate)

r_peaks = fp.jqrs()

r_peaks = np.asarray(r_peaks, dtype=float)

print(type(r_peaks))
print(r_peaks.shape)
print(r_peaks.ndim)

matlab_runtime_path = (
    r"C:\Program Files\MATLAB\MATLAB Runtime\v910"
)

fiducials = fp.wavedet(
    matlab_runtime_path,
    peaks=r_peaks,
)

bm = Bm.Biomarkers(signals, sampling_rate, fiducials)

ints, stat_i = bm.intervals()

waves, stat_w = bm.waves()


ints[0].keys()

stat_i[0].keys()

signals = record.p_signal

lead_index = 0
beat_index = 3

qrs_offset = fiducials[lead_index]["QRSoff"][beat_index]
qrs_onset = fiducials[lead_index]["QRSon"][beat_index]
p_offset = fiducials[lead_index]["Poff"][beat_index]

# Check before converting NaN to integer
if not np.all(np.isfinite([qrs_offset, qrs_onset, p_offset])):
    raise ValueError("One or more fiducials are missing")

qrs_offset = int(round(float(qrs_offset)))
qrs_onset = int(round(float(qrs_onset)))
p_offset = int(round(float(p_offset)))



# here try to extract st segment depression 
offset_40ms = round(0.040 * 300)  # 12 samples
offset_60ms = round(0.060 * 300)  # 18 samples
offset_80ms = round(0.080 * 300)  # 24 samples

st_sample = qrs_offset + round(0.060 * sampling_rate)

pr_baseline = np.median(
    signals[p_offset:qrs_onset]
)

st_deviation_mv = signals[st_sample] - pr_baseline

st_deviation_mv

for st_deviation_mvItem in st_deviation_mv:
    if st_deviation_mvItem <= -0.1:
        print("ST depression of at least 0.1 mV")
    elif st_deviation_mvItem >= 0.1:
        print("ST elevation of at least 0.1 mV")
    else:
        print("No substantial ST deviation")
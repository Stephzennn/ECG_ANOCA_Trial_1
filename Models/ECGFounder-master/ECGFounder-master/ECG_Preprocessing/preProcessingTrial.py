
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
target_fs = 500  # Desired sampling rate

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


"""
type(signals)

signals.shape


dd = signals.shape

print(dd)

dd1 = dd[1]

print(dd1)
for leads in range(signals.shape[1]):
    print(leads)

"""


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

"""
firstLead = signals[:, 0]




original_fs = sampling_rate   # Sampling rate in the ISHNE header
target_fs = 500

firstLead_resampled = nk.signal_resample(
    firstLead,
    sampling_rate=original_fs,
    desired_sampling_rate=target_fs,
    method="poly"
)

signals[:, 0] = firstLead_resampled

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

for x in range(len(f)):
    print(f[x])
    #print(type(f[x]))
    print(f[x].shape)
    break


type(signals)

signals.shape


def Plot(signals, sampling_rate):
    

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


Plot(f, sampling_rate)


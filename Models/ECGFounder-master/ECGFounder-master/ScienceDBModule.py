import pandas as pd
import wfdb
from wfdb.io.convert.csv import csv_to_wfdb

from dataset import LVEF_12lead_cls_Dataset
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import torch
import json

csv_filepath = './csv/ScienceDB_label.csv'
picklePath = './csv/ScienceDB_label.pkl'
#ecg_filepath = 'C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data/sub_001/sub_001_baseline-data_rest-ecg-500hz.csv'
ecg_filepath = 'C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data'


lead_names = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

fs = 500
duration = 10


# Here the ptxbl changes the wfdb file into an np array, check to see if you can transition from
#csv directly to nparray. if so, you might not need to change to wfdb.


#data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
#data = np.array([signal for signal, meta in data])


# Here we must create a label csv

folder_path = Path("C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data")

#Here we cut each patients ECG data into slices of 10 seconds.
for file_path in folder_path.iterdir():
    if file_path.name[0:3] == "sub":
        words = ["baseline","data", "rest", "500hz" ]
        target_dir  = file_path / "records500"
        target_dir.mkdir(parents=True, exist_ok=True)
        for csvFile in file_path.iterdir():
            contains_word = all(
                    word.lower() in (csvFile.name).lower()
                            for word in words
                        )
            if (contains_word) == True:
                ecg_signal = pd.read_csv(
                    csvFile,
                    header=None,
                    nrows=12,
                    dtype="float32"
                ).to_numpy().T
                count = 0
                nameCount = 0
                while True:
                    if count > ecg_signal.shape[0]:
                        break
                    ecg_signal_ten_seconds = ecg_signal[count:count + (fs * 10),:]
                    ecg_signal_ten_seconds = np.asarray(ecg_signal_ten_seconds, dtype=np.float64)
                    print(ecg_signal_ten_seconds.shape)
                    if ecg_signal_ten_seconds.shape[0] < 5000:
                        break
                    wfdb.wrsamp(
                        record_name= str(nameCount) + '_10seconds',
                        fs=500,  # Sampling frequency in Hz
                        units=["mV"] * 12,
                        sig_name= lead_names,
                        p_signal=ecg_signal_ten_seconds,
                        write_dir= target_dir
                    )
                    count += (fs * 10)
                    nameCount += 1
                print(target_dir)



def turnTo150Output(classs ):
    t = np.zeros([150]).astype(int)
    t = list(t)
    if classs == 1:
        t[0] = 1
        return t
    else:
        #t[2] = 1
        return t


temListmeta = []

for file_path in folder_path.iterdir():
    if file_path.name[0:3] == "sub":
        patientID = int(file_path.name[4:])
        words = ['disease' , 'json']
        for recordName in (file_path / "records500").iterdir():
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
                            output150 = turnTo150Output(disease_data['ANOCA'])
                            tempList.append(disease_data['ANOCA'])
                            tempList.append(str(output150))
                            temListmeta.append(tempList)


dfTemp = pd.DataFrame(temListmeta, columns=['patient_id', 'filename_hr','BinaryLabel', 'label',  ])


#output_path_pickle = r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\csv\ScienceDB_label.pkl"


# Save the label csv

output_path = r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\csv\ScienceDB_label.csv"

dfTemp.to_csv(output_path, index=False)


"""

UNUSED CODE UNUSED CODE UNUSED CODE UNUSED CODE UNUSED CODE UNUSED CODE UNUSED CODE  


# get positive vs negative label 
for file_path in folder_path.iterdir():
    if file_path.name[0:3] == "sub":
        words = ['disease' , 'json']
        for csvFile in file_path.iterdir():
            contains_word = all(
                    word.lower() in (csvFile.name).lower()
                            for word in words
                        )
            if contains_word == True:
                with open(csvFile, "r") as file:
                    disease_data = json.load(file)
                #print("For the subject",file_path.name[4:],"The anoca profile is",disease_data['ANOCA']   )
                output150 = turnTo150Output(disease_data['ANOCA'])
                print(output150)



    
turnTo150Output(1)
t = np.zeros([150])
t
df = pd.read_json()

with open("C:\\Users\\Estif\\Downloads\\Langone\\ANOCA\\ECG_ANOCA_Trial_1\\Models\\ECGFounder-master\\ECGFounder-master\\data\\ScienceDB_Data\\sub_097\\sub_097_disease.json", "r") as file:
    disease_data = json.load(file)
disease_data['ANOCA']




picklePath ecg_filepath

labels

import ast
import numpy as np

labels["label"] = labels["label"].apply(
    lambda s: np.asarray(ast.literal_eval(s), dtype=np.float64)
)

clean_string = (((labels['label'].iloc[0])[1:-1])).replace(", ", "")
pd.Series(list(clean_string)).astype(np.float64)


labels = self.labels_df.iloc[idx, -1]
labels = labels.astype(np.float32)

import ast
labels = pd.read_csv(csv_filepath)


labels = pd.read_pickle(picklePath)

labels

type(labels['label'].iloc[1])


xx = LVEF_12lead_cls_Dataset(ecg_filepath, labels)

check, tt = xx.__getitem__(5000)

type(tt)

tt

xx.__getitem__(1)





lead_names = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]

fs = 500
duration = 10

#Check check check
import time

start = time.time()


#Check first
#ecgSignaltest = pd.read_csv(ecg_filepath)

# Load only the portion being plotted

# One ecg csv data has 6 minutes (360 seconds) of data. so we would have 36 (10s) samples 
ecg_signal = pd.read_csv(
    ecg_filepath,
    header=None,
    nrows=12,
    #usecols=range(fs * duration),
    dtype="float32"
).to_numpy().T

print("ECG shape:", ecg_signal.shape)
print("Number of samples:", ecg_signal.shape[0])
print("Number of leads:", ecg_signal.shape[1])
print("Duration:", ecg_signal.shape[0] / fs, "seconds")


ecg_signal_ten_seconds = ecg_signal[:fs * 10,:]
ecg_signal_ten_seconds.shape

preview_record = wfdb.Record(
    p_signal=ecg_signal_ten_seconds,
    fs=fs,
    sig_name=lead_names,
    units=["mV"] * 12,
    record_name="science_db_ecg"
)

wfdb.plot_wfdb(
    record=preview_record,
    title="First 10 seconds",
    time_units="seconds")

print("Plotting time:", time.time() - start)

type(ecg_signal_ten_seconds)
import numpy as np
ecg_signal_ten_seconds = np.asarray(ecg_signal_ten_seconds, dtype=np.float64)

wfdb.wrsamp(
    record_name='Trial_Ten_Seconds',
    fs=500,  # Sampling frequency in Hz
    units=["mV"] * 12,
    sig_name= lead_names,
    p_signal=ecg_signal_ten_seconds,
    write_dir='C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data'
)



record = wfdb.rdrecord('C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000/00079_hr')

wfdb.plot_wfdb(record=record, title='ECG Record Plot')

record_metadata = wfdb.rdheader('C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000/00079_hr')

record_metadata2 = wfdb.rdheader('C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/Trial_Ten_Seconds')

record2 = wfdb.rdrecord('C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data/sub_001/records500/1_10seconds')

wfdb.plot_wfdb(record=record2, title='ECG Record Plot')

count = 0
nameCount = 0
while True:
    if count > ecg_signal.shape[0]:
        break
    ecg_signal_ten_seconds = ecg_signal[count:count + (fs * 10),:]
    ecg_signal_ten_seconds = np.asarray(ecg_signal_ten_seconds, dtype=np.float64)
    print(ecg_signal_ten_seconds.shape)
    if ecg_signal_ten_seconds.shape[0] < 5000:
        break
    wfdb.wrsamp(
        record_name= str(nameCount) + '_10seconds',
        fs=500,  # Sampling frequency in Hz
        units=["mV"] * 12,
        sig_name= lead_names,
        p_signal=ecg_signal_ten_seconds,
        write_dir='C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data/sub_001/records500'
    )
    count += (fs * 10)
    nameCount += 1


"""
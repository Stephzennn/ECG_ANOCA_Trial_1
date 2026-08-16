import numpy as np
import pandas as pd
import wfdb
import json
import torch
from scipy.interpolate import interp1d

class PTBXL_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecg_path, csv_path):

        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['filename_hr', 'label'])
        self.fs = 5000
        self.ecg_path = ecg_path


    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8) 
        
    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
    
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
    
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True) 
        x_new = np.linspace(0, t, num=int(fs_out), endpoint=True) 

        for i in range(ts.shape[0]):
            y_old = ts[i, :]
            f = interp1d(x_old, y_old, kind='linear')
            resampled_ts[i, :] = f(x_new)
    
        return resampled_ts
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        hash_file_name = row['filename_hr']
        label = row['label']
        label = json.loads(label)
        label = torch.tensor(label, dtype=torch.float)
        
        sample_rate = 500
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = self.z_score_normalization(data)
        signal = self.resample_unequal(data, sample_rate, self.fs)
        signal = torch.FloatTensor(signal)
        return signal, label
    



import os
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from shutil import copyfile
import pickle
import time
import wfdb
import ast
from scipy import signal
import json
from net1d import Net1D
from util import eval_with_dynamic_thresh
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from scipy.stats import bootstrap
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from scipy.interpolate import interp1d


#from sklearn.decomposition import PCA , TSNE

from sklearn.manifold import TSNE
device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

print(device)


# Create a dummy input tensor with shape (batch_size, channels, length)
x = torch.randn(64, 12, 5000)

# Move the input tensor to the same device as the model
x = x.to(device)


# Try with real dataset.

saved_dir = './res/eval'
csv_filepath = './csv/ptbxl_label.csv'
ecg_filepath = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'


all_embeddings = []
all_labels = []

from ptbxlModule import PTBXL_Dataset
tasks = []
batch_size = 512
with open(os.path.join('./tasks.txt'), 'r') as fin: 
    for line in fin:
        tasks.append(line.strip())

testset = PTBXL_Dataset(ecg_path=ecg_filepath, csv_path=csv_filepath)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers= 0) #(os.cpu_count())




#------------------

model = Net1D(
      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose= False, 
      use_bn=False,
      use_do=False,
      n_classes=150,
      return_features=True)

model.to(device)

checkpoint = torch.load('./checkpoint/12_lead_ECGFounder.pth', map_location=device)
state_dict = checkpoint['state_dict']

log = model.load_state_dict(state_dict, strict=False)

for name, param in model.named_parameters():
    param.requires_grad = False

model.to(device)

model.eval()


#_, deep_features = model(x)



#print(deep_features.shape)  # Should print the shape of the output features


# Here try to exract the deep features for every sample and the labels


prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
all_gt = []
all_pred_prob = []
all_thre_df = []
all_embeddings = []

with torch.no_grad():
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        logits, deep_features = model(input_x)
        pred = F.sigmoid(logits)
        all_pred_prob.append(pred.cpu().data.numpy())
        all_gt.append(input_y.cpu().data.numpy())
        all_embeddings.append(deep_features.cpu().data.numpy())
all_pred_prob = np.concatenate(all_pred_prob)
all_gt = np.concatenate(all_gt)
all_embeddings = np.concatenate(all_embeddings)
labels = np.concatenate(all_gt)
df_gt = pd.DataFrame(all_gt)

all_gt.shape

afib_gt = all_gt[:, 2]


afib_gt = afib_gt.reshape(afib_gt.shape[0], 1)

afib_pred_prob = all_pred_prob[:, 2]

df_afib_gt = pd.DataFrame(afib_gt)
df_afib_pred_prob = pd.DataFrame(afib_pred_prob)

combined_df = pd.concat([df_afib_gt, df_afib_pred_prob], axis=1)

combined_df.columns = ['afib_GroundTruth', 'afib_pred_prob']

combined_df.head()


#Plot kernel density plot 

ax = combined_df.loc[
    combined_df["afib_GroundTruth"] == 0,
    "afib_pred_prob"
].plot.kde(
    color="steelblue",
    linewidth=2.5,
    label="No AFib"
)

combined_df.loc[
    combined_df["afib_GroundTruth"] == 1,
    "afib_pred_prob"
].plot.kde(
    ax=ax,
    color="darkorange",
    linewidth=2.5,
    label="AFib"
)

plt.title("AFib Predicted Probability by Ground Truth")
plt.xlabel("Predicted AFib Probability")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()


out_path_auc = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"AFibKernel_Density_Plot.png")

#plt.savefig(out_path_auc)
plt.show()


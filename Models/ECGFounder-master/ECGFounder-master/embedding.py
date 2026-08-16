


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

type(df_gt) 
label_two = []
for x in range(df_gt.shape[0]):
    if df_gt.iloc[x][93] == 1:
        label_two.append(1)
    else:
        label_two.append(0)

type(label_two), len(label_two)
all_gt
type(all_embeddings), all_embeddings.shape, type(labels), labels.shape
label_two = np.array(label_two)
# For every output, extract PCA components of 50. and save it in a new file.



#pca = PCA(n_components=50, svd_solver='covariance_eigh')

# For every extracted per smaple PCA (50), extract 2 dimensional TSNE and save it in a new file. 

# For the extracted per sample TSNE (2), extract 2 dimensional UMAP and save it in a new file.

def run_tsne(embeddings, n_samples):
    #perplexity = args.tsne_perplexity or min(30, max(5, n_samples // 10))
    perplexity =  min(30, max(5, n_samples // 10))
        
    print(f"  t-SNE: perplexity={perplexity}, n_iter={1000}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=1000,
        random_state=42,
        verbose=0,
    )
    return tsne.fit_transform(embeddings)


# For the extracted per sample UMAP (2), plott the in a cluster and color it based on the label of the sample.


tsne_results = run_tsne(all_embeddings, all_embeddings.shape[0])

print(tsne_results.shape)

labels = np.random.randint(0, 2, size=all_embeddings.shape[0])



# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------



def save_individual_plot(z, labels, out_path):
    n_total = len(labels)
    n_pos   = int((labels == 1).sum())
    n_neg   = int((labels == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
               c="#4878d0", alpha=0.55, s=18, linewidths=0,
               label=f"SUPRAVENTRICULAR TACHYCARDIA-Negative  (n={n_neg})")
    ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
               c="#ed3bde", alpha=0.75, s=28, linewidths=0,
               label=f"SUPRAVENTRICULAR TACHYCARDIA-Positive  (n={n_pos})")

    mode_tag = "pretrained backbone"
    ax.set_title(
        f"{mode_tag} embedding (t-SNE) \n"
        f"N={n_total}",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.legend(fontsize=9, markerscale=1.8, framealpha=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")





out_path = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"embedding_tsne.png")

save_individual_plot(tsne_results, label_two, out_path)
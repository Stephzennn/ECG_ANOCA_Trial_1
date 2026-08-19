


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
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def createDataloader(task_Path, batch_Size, ecg_filepath, csv_path, num_workers = 0 ):
    from ptbxlModule import PTBXL_Dataset
    tasks = []
    batch_size = batch_Size
    #'./tasks.txt'
    with open(os.path.join(task_Path), 'r') as fin: 
        for line in fin:
            tasks.append(line.strip())
    #csv_filepath
    testset = PTBXL_Dataset(ecg_path=ecg_filepath, csv_path=csv_path)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers= num_workers) #(os.cpu_count())

    return testset, testloader


def loadWeightsToModel(weightPath,model,device):
    checkpoint = torch.load(weightPath, map_location=device)
    state_dict = checkpoint['state_dict']

    log = model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = False

    model.to(device)

    model.eval()
    return log
    

def generateOutput(testloader, model, device):
    prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
    all_gt = []
    all_pred_prob = []
    all_logits = []
    all_embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            logits, deep_features = model(input_x)
            all_logits.append(logits.detach().cpu().numpy())
            pred = F.sigmoid(logits)
            all_pred_prob.append(pred.cpu().data.numpy())
            all_gt.append(input_y.cpu().data.numpy())
            all_embeddings.append(deep_features.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_gt = np.concatenate(all_gt)
    all_embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_gt)
    df_gt = pd.DataFrame(all_gt)
    return all_gt, all_embeddings, df_gt, labels, all_pred_prob , all_logits



def extractResults(task: int, all_gt, all_pred_prob, taskPath ,df_gt):
    dd = "Place Holder"
    counter = 0
    with open(os.path.join(taskPath), 'r') as fin: 
            for line in fin:
                if counter == task:
                    dd = line.strip()
                    break
                counter += 1

    print(f"Extract Results for {dd}")
    afib_gt = all_gt[:, task]
    afib_gt = afib_gt.reshape(afib_gt.shape[0], 1)
    afib_pred_prob = all_pred_prob[:, task]

    afib_pred_prob = afib_pred_prob.reshape(afib_pred_prob.shape[0], 1)

    afib_gt.shape, afib_pred_prob.shape
    res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds = eval_with_dynamic_thresh(afib_gt, afib_pred_prob)

    label_two = []
    for x in range(df_gt.shape[0]):
        if df_gt.iloc[x][task] == 1:
            label_two.append(1)
        else:
            label_two.append(0)
    label_two = np.array(label_two)

    return res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds, label_two, afib_gt, afib_pred_prob, dd
    


# For every output, extract PCA components of 50. and save it in a new file.



#pca = PCA(n_components=50, svd_solver='covariance_eigh')

# For every extracted per smaple PCA (50), extract 2 dimensional TSNE and save it in a new file. 

# For the extracted per sample TSNE (2), extract 2 dimensional UMAP and save it in a new file.

def run_tsne(embeddings, n_samples):
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

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_individual_plot(z, labels, out_path, taskName):
    n_total = len(labels)
    n_pos   = int((labels == 1).sum())
    n_neg   = int((labels == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
               c="#4878d0", alpha=0.55, s=18, linewidths=0,
               label=f"{taskName}-Negative  (n={n_neg})")
    ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
               c="#ed3bde", alpha=0.75, s=28, linewidths=0,
               label=f"{taskName}-Positive  (n={n_pos})")

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

# Plot roc curve 

def plotSavePRAUC(groundTruth, prediction_Probability, outputPath ):
    prec, rec, _ = metrics.precision_recall_curve(groundTruth, prediction_Probability)
    auc_val = average_precision_score(groundTruth, prediction_Probability)
    plt.legend(loc = 'lower right')
    plt.plot(rec, prec, lw=1.8,
                    label = 'PR_AUC = %0.2f' % auc_val)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve ")
    plt.axhline(groundTruth.mean(), color="red", linestyle="--")
    plt.savefig(outputPath)

# ROC AUC
def plotSaveROCAUC(groundTruth, prediction_Probability, outputPath ):
    fpr, tpr, threshold = metrics.roc_curve(groundTruth,prediction_Probability)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(outputPath)
    plt.show()




if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    print(device)

    saved_dir = './res/eval'
    csv_filepath = './csv/ptbxl_label.csv'
    ecg_filepath = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    all_embeddings = []
    all_labels = []
    testset, testloader = createDataloader('./tasks.txt', 512, ecg_filepath,csv_filepath)
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


    log = loadWeightsToModel('./checkpoint/12_lead_ECGFounder.pth', model, device)
    all_gt, all_embeddings, df_gt, labels, all_pred_prob = generateOutput(testloader, model, device)
    TaskNumberID = 5
    res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds, label_two, afib_gt, afib_pred_prob, TaskName = extractResults(TaskNumberID, all_gt, all_pred_prob,'./tasks.txt', df_gt)
    
    tsne_results = run_tsne(all_embeddings, all_embeddings.shape[0])
    
    print(tsne_results.shape)
    out_path = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"Trialembedding_tsne.png")

    save_individual_plot(tsne_results, label_two, out_path, TaskName)
    out_path_prauc = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"TrialPRAuc_CurveNORMAL ECG.png")

    plotSavePRAUC( afib_gt, afib_pred_prob,out_path_prauc )
    out_path_auc = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"TrialAuc_CurveNORMAL ECG.png")
    plt.close()
    plotSaveROCAUC(afib_gt, afib_pred_prob,out_path_auc)
    plt.close()




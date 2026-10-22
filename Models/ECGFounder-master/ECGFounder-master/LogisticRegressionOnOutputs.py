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

from embedding import createDataloader, loadWeightsToModel, generateOutput, extractResults, save_individual_plot, plotSavePRAUC


if __name__ == "__main__":
    # Change the paths 
    saved_dir = './res/ScienceDBeval_Trial1'
    csv_filepath = './csv/ScienceDB_label.csv'
    #csv_filepath = './csv/ScienceDB_label_stress.csv'
    ecg_filepath = 'C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data'

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    all_embeddings = []
    all_labels = []
    testset, testloader = createDataloader('./tasks.txt', 512, ecg_filepath,csv_filepath,ScienceDB=False)
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
    all_gt, all_embeddings, df_gt, labels, all_pred_prob , all_logits = generateOutput(testloader, model, device)
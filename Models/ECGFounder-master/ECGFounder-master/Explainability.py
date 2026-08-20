
import os
import pandas as pd
from matplotlib import pyplot as plt
import os
from net1d import Net1D
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import Lime
from captum.attr import NeuronConductance
from embedding import createDataloader, loadWeightsToModel, generateOutput
import numpy as np

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

    all_gt, all_embeddings, df_gt, labels, all_pred_prob, all_logits = generateOutput(testloader, model, device)

    # Testset

    type(all_logits)
    len(all_logits)
    type(all_logits[0])

    

    ttr = np.empty(all_logits[0].shape)


    #ttr = np.vstack((ttr,all_logits[1] ))
    

    for x in range(len(all_logits)):
        ttr = np.vstack((ttr,all_logits[x] ))
        #print(all_logits[x].shape)

    print(ttr.shape)

   


    # The all_pred_prob refer to the output logits.
    #out_logits = model(test_input_tensor).detach().numpy() 
    #logits, deep_features = model(input_x)


    out_logits = ttr

    #out_logits = logits.detach().numpy()

    ig = IntegratedGradients(model)

    target = 5

    type(testset.data)

    print(testset.data.shape[0])
    # The testset takes in the number of samples.
    dd = testset.__getitem__(21798)

    print(dd[0].shape)
    type(dd)

    inputTrial = []

    #print()
    for x in range(testset.data.shape[0]):
            if x % 1000 == 0 :
                print(x)
            inputTrial.append(testset.__getitem__(x)[0] )

    len(inputTrial)
    print(inputTrial.shape)
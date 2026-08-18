


import os
import pandas as pd
from matplotlib import pyplot as plt
import os
from net1d import Net1D
import torch



from embedding import createDataloader, loadWeightsToModel, generateOutput, extractResults

#extractResults()
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
    TaskNumberID = 2
    res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1, optimal_thresholds, label_two, afib_gt, afib_pred_prob, TaskName = extractResults(TaskNumberID, all_gt, all_pred_prob,'./tasks.txt',df_gt)
    
    df_afib_gt = pd.DataFrame(afib_gt)
    df_afib_pred_prob = pd.DataFrame(afib_pred_prob)

    combined_df = pd.concat([df_afib_gt, df_afib_pred_prob], axis=1)

    combined_df.columns = ['afib_GroundTruth', 'afib_pred_prob']

    combined_df.head()
    
    ax = combined_df.loc[
    combined_df["afib_GroundTruth"] == 0,
    "afib_pred_prob"
    ].plot.kde(
        color="steelblue",
        linewidth=2.5,
        label=f"No {TaskName}"
    )

    combined_df.loc[
        combined_df["afib_GroundTruth"] == 1,
        "afib_pred_prob"
    ].plot.kde(
        ax=ax,
        color="darkorange",
        linewidth=2.5,
        label=f"{TaskName}"
    )

    plt.title(f"{TaskName} Predicted Probability by Ground Truth")
    plt.xlabel(f"Predicted {TaskName} Probability")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    out_path_auc = os.path.join(r"C:\Users\Estif\Downloads\Langone\ANOCA\ECG_ANOCA_Trial_1\Models\ECGFounder-master\ECGFounder-master\Images", f"{TaskName}Kernel_Density_Plot.png")

    plt.savefig(out_path_auc)
    plt.show()



"""
all_gt.shape

afib_gt = all_gt[:, 2]


afib_gt = afib_gt.reshape(afib_gt.shape[0], 1)

afib_pred_prob = all_pred_prob[:, 2]



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

plt.savefig(out_path_auc)
plt.show()

"""
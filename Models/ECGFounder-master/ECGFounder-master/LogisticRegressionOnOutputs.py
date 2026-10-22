import numpy as np
import pandas as pd
import torch
from pathlib import Path
from net1d import Net1D

from embedding import createDataloader, loadWeightsToModel, generateOutput, averageLogitsPerPerson, loadTaskNames


# Paths relative to this file, so it runs the same on Windows and on the cluster
base_path = Path(__file__).resolve().parent
taskPath = base_path / "tasks.txt"
checkpointPath = base_path / "checkpoint" / "12_lead_ECGFounder.pth"
csv_filepath = base_path / "csv" / "ScienceDB_label_all_phases.csv"
ecg_filepath = str(base_path / "data" / "ScienceDB_Data")

# The per person (and phase) 150 logits are saved here, so the model only runs once
personLogitsPath = base_path / "csv" / "ScienceDB_person_logits.csv"


# Run ECGFounder on every 10 second slice, then average the 150 logits per person and phase.
# If this was already done, just read the saved csv.
def getPersonLogits(rerun = False):
    if personLogitsPath.exists() and rerun == False:
        print(f"Reading saved person logits from {personLogitsPath}")
        return pd.read_csv(personLogitsPath, index_col = ["patient_id", "phase"])

    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    testset, testloader = createDataloader(taskPath, 512, ecg_filepath, csv_filepath, ScienceDB=True)
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
    log = loadWeightsToModel(checkpointPath, model, device)
    all_gt, all_embeddings, df_gt, labels, all_pred_prob , all_logits = generateOutput(testloader, model, device)

    personTable = averageLogitsPerPerson(testset, all_logits, taskPath)
    personTable.to_csv(personLogitsPath)
    print(f"Saved person logits to {personLogitsPath}")
    return personTable


if __name__ == "__main__":

    personTable = getPersonLogits()

    # quick checks: should be 97 people x 3 phases, 150 class columns + n_ECGs + labels
    print(personTable.shape)
    print(personTable.groupby("phase").size())
    print(personTable[["n_ECGs", "ANOCA", "CMD"]].head())

    # one phase at a time, e.g. rest
    restTable = personTable.xs("rest", level = "phase")



import os 

from ScienceDB_Dataset import ScienceDB_Dataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from net1d import Net1D
# Change the paths 
saved_dir = './res/ScienceDBeval'
csv_filepath = './csv/ScienceDB_label.csv'
ecg_filepath = 'C:/Users/Estif/Downloads/Langone/ANOCA/ECG_ANOCA_Trial_1/Models/ECGFounder-master/ECGFounder-master/data/ScienceDB_Data'


tasks = []
batch_size = 512
with open(os.path.join('./tasks.txt'), 'r') as fin: 
    for line in fin:
        tasks.append(line.strip())

#testset = PTBXL_Dataset(ecg_path=ecg_filepath, csv_path=csv_filepath)

testset = ScienceDB_Dataset(ecg_path=ecg_filepath, csv_path=csv_filepath)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers= 0) #(os.cpu_count())



device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

print(device)


### make model
model = Net1D(
    in_channels=12, 
    base_filters=64, #64
    ratio=1, 
    filter_list=[64, 160, 160, 400, 400, 1024, 1024], #[64, 160, 160, 400, 400, 1024, 1024]
    m_blocks_list=[2,2,2,3,3,4,4], 
    kernel_size=16, 
    stride=2, 
    groups_width=16,
    verbose=False, 
    use_bn=False,
    use_do=False,
    n_classes=150)

model.to(device)

checkpoint = torch.load('./checkpoint/12_lead_ECGFounder.pth', map_location=device)
state_dict = checkpoint['state_dict']

log = model.load_state_dict(state_dict, strict=False)

for name, param in model.named_parameters():
    param.requires_grad = False

model.to(device)


model.eval()
prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
all_gt = []
all_pred_prob = []
all_thre_df = []

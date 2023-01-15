import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from utils import ngsimDataset
from torch.utils.data import DataLoader
import time
import os

# ----1 parameters----
args={
    'use_cuda': False,
    # 'encoder_size':64,
    # 'decoder_size':128,
    # 'in_length':  16,
    # 'out_length':  25,
    'grid_size':  (13,3),
    # 'soc_conv_depth':  64,
    # 'conv_3x1_depth':  16,
    # 'dyn_embedding_size':  32,
    # 'input_embedding_size':  32,
    'num_lat_classes':  3,
    'num_lon_classes':  2,
    'use_maneuvers':  False, #s就是False m就是True
    'train_flag':  True,
    # 'pretrainEpochs': 1, #原来是5
    # 'trainEpochs':1, #原来是3
    'batch_size':1024 #原来是128 太小了
}

# ----2 load data----
trainset = ngsimDataset('data/TrainSet.mat')
validset = ngsimDataset('data/ValSet.mat')
trDataloader = DataLoader(trainset,batch_size=args['batch_size'],shuffle=True,num_workers=128,collate_fn=trainset.collate_fn)
valDataloader = DataLoader(validset,batch_size=args['batch_size'],shuffle=True,num_workers=128,collate_fn=validset.collate_fn)

# ----3 train----
#  model=
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
batch_size = args['batch_size']
criterion = nn.BCELoss() 

#--------------save model
file_time=time.strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.isdir('./model_results'):
    os.mkdir('./model_results')
model_path='model_results/lstm_s_'+file_time+'.ckpt' 
# model_path=os.path.join('model_results','cslstm_m',file_time,'.ckpt')#这样的结果是models/cslstm_m/2022-12-06-14-34-53/.ckpt
# model_path='model_results/cslstm_s_'+file_time+'.ckpt'
torch.save(model.state_dict(), model_path)
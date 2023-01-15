import math
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

import time
import os

from utils import ngsimDataset
from model import slstm,cslstm, vlstm
from loss import loss_v2


# ----1 parameters----
args={
    'use_cuda': True,
    'encoder_size':64,
    'decoder_size':128,
    'in_length':  16,
    'out_length':  25,
    'grid_size':  (13,3),
    'soc_conv_depth':  64,
    'conv_3x1_depth':  16,
    'dyn_embedding_size':  32,
    'input_embedding_size':  32, # 为什么
    'num_lat_classes':  3,
    'num_lon_classes':  2,
    'use_maneuvers':  False, #s就是False m就是True
    'train_flag':  True,
    'pretrainEpochs': 1, #原来是5
    'trainEpochs':1, #原来是3
    'batch_size':1024 #原来是128 太小了
}

# ----2 load data----
trainset = ngsimDataset('data/TrainSet.mat')
validset = ngsimDataset('data/ValSet.mat')
# collate_fn用自定义的！
trDataloader = DataLoader(trainset,batch_size=args['batch_size'],shuffle=True,num_workers=128,collate_fn=trainset.collate_fn)
valDataloader = DataLoader(validset,batch_size=args['batch_size'],shuffle=True,num_workers=128,collate_fn=validset.collate_fn)

# ----3 train----
model=slstm(args)
if args['use_cuda']:
    model = model.cuda() # model移到cuda上=to(device)
pretrainEpochs = args['pretrainEpochs'] 
trainEpochs =args['trainEpochs']
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
batch_size = args['batch_size']
criterion = nn.BCELoss() 

train_loss,val_loss,prev_val_loss = [],[],math.inf
avg_tr_loss = math.inf
avg_val_loss=math.inf
avg_val_lat_acc=0
avg_val_lon_acc=0
for epoch_num in range(pretrainEpochs+trainEpochs):
    if epoch_num == 0 and epoch_num < pretrainEpochs: #一开始用mse train,后面用nLL train，用到了BCELoss为什么？？？
        print('Pre-training with MSE loss') #train用RMSE没有acc
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss') #就是valid用NLL loss 才有acc



#--------------save model
file_time=time.strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.isdir('./model_results'):
    os.mkdir('./model_results')
model_path='model_results/slstm_s_'+file_time+'.ckpt' 
# model_path=os.path.join('model_results','cslstm_m',file_time,'.ckpt')#这样的结果是models/cslstm_m/2022-12-06-14-34-53/.ckpt
# model_path='model_results/cslstm_s_'+file_time+'.ckpt'
torch.save(model.state_dict(), model_path)
import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
# 打包dataloader里面的collate_fn用自定义的！
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
    if epoch_num == 0 and epoch_num < pretrainEpochs: #一开始用mse train,后面用nLL train，用到了BCELoss 为什么pretrain再train？？？
        print('Pre-training with RMSE loss') #train用RMSE没有acc
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss') #就是valid用NLL loss 才有acc
    
    # ----train----
    model.train_flag=True
    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        mask = mask.bool()  # hsy改10.24
        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
        
        # Forward pass
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = model(hist, nbrs, mask, lat_enc, lon_enc)
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)
        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99:
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0


# ----save model----
file_time=time.strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.isdir('./model_results'):
    os.mkdir('./model_results')
model_path='model_results/slstm_s_'+file_time+'.ckpt' 
# model_path=os.path.join('model_results','cslstm_m',file_time,'.ckpt')#这样的结果是models/cslstm_m/2022-12-06-14-34-53/.ckpt
# model_path='model_results/cslstm_s_'+file_time+'.ckpt'
torch.save(model.state_dict(), model_path)
import torch
import torch.nn as nn

## Custom activation for output layer (Graves, 2015) #为什么 没看懂
def outputActivation(x): #x是future_pred: [25,batch_size,5]
    muX = x[:,:,0:1] # [25,batch_size,第0] x坐标的均值
    muY = x[:,:,1:2] # [25,batch_size,第1]
    sigX = x[:,:,2:3] # [25,batch_size,第2] x坐标的方差
    sigY = x[:,:,3:4]# [25,batch_size,第3]
    rho = x[:,:,4:5]# [25,batch_size,第4] 但是不能改成[:,:,4]因为会自动降一维 xy的相关系数
    sigX = torch.exp(sigX)  
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho) #有点像二元正态分布，μx,μy,e^σx,e^σy,tanh(ρ)相关系数
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2) #按dim=2 串起来,除了dim=2其他维度都一样，=[25,batch_size,1+1+1+1+1=5]
    return out #[25,batch_size,5]

class sLSTM(nn.Module):    
    ## init
    def __init__(self,args):
        super(sLSTM,self).__init__()

        self.args=args
        self.use_cuda=args['use_cuda']
        self.use_maneuvers=args['use_maneuvers'] #true就是multi-modal的多条输出，false就是single-modal一条输出
        self.train_flag=args['train_flag'] #true: train; false: test; valid是true还是false？

        # 定义network layer的参数
        self.encoder_size = args['encoder_size'] # 64
        self.decoder_size = args['decoder_size'] # 128
        self.in_length = args['in_length'] # 16
        self.out_length = args['out_length'] # 25
        self.grid_size = args['grid_size'] # (13,3)
        self.soc_conv_depth = args['soc_conv_depth'] # 64
        self.conv_3x1_depth = args['conv_3x1_depth'] #？ 16        
        self.dyn_embedding_size = args['dyn_embedding_size'] # 32? 是什么
        self.input_embedding_size = args['input_embedding_size'] #32 
        self.num_lat_classes = args['num_lat_classes'] # 3
        self.num_lon_classes = args['num_lon_classes'] # 2
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth # 13-4=9 10//2=5  5*64=320为什么这么算

        # -----下面都是Encoder参数，包括加的social_pooling-----------
        # 定义network layer的weight
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)# 网络参数:(2,32) [输入,输出=input_embedding_size], Input embedding layer就是全连接层
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1) # 网络参数: (32,32,1) input_size=input_embedding_size, hidden_size=self.encoder_size, num_layers=1        
       
        # Vehicle dynamics embedding(这层是个什么东西，dynamics？？)
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size) #输出[encoder_size,dyn_embedding_size]

        # # Convolutional social pooling layer and social embedding layer 这个是cslstm比普通的lstm多了这里
        # self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3) #(64,64,3) in_channels：encoder_size,out_channels:soc_conv_depth,kernel_size 3*3 
        # self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))# （64，16，（3，1））in_channels,out_channels,kernel_size
        # self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0)) # 两个取1个大的，kernel_size=(2*1), padding=(1,0)为什么

        # FC social pooling layer (for comparison):上面三行改成这个就是slstm
        self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)
         #[soc_conv_depth64*13*3,5*16conv_3x1_depth就是soc_embedding_size] 这个怎么算的

        # -----下面都是Decoder参数-----------
        if self.use_maneuvers: #多了self.num_lat_classes + self.num_lon_classes
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size) #[soc_embedding_size+dyn_embedding_size+3+2,decoder_size]
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size) #网络参数：[80+32,128][inputsize=soc_embedding_size+dyn_embedding_size,hidden_size=decoder_size] num_layers默认1

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5) # [128,5]
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)#[112，3][soc_embedding_size+dyn_embedding_size,3]
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)#[112,2] [soc_embedding_size+dyn_embedding_size,2]

        # Activations:（neurons）
        self.leaky_relu = torch.nn.LeakyReLU(0.1) # negative_slope=0.1
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) # 在dim=1这一维上归一化，这一维的sum=1

    def decode(self,enc):
        '''
        enc: [batch_size,112]= [batch_size,soc_embeddingsize+input_embedding_size]
        enc.repeat(25,1,1): [25,batch_size,112]=[out_length,batch_size,soc_embeddingsize+input_embedding_size]
        h_dec:lstm的输入[25,batch_size,112]=[sequence_len=看历史几帧,batch_size,input_size=就是features] 
                取lstm输出的output:[25,batch_size,1*128] (sequence length,batch_size,num_directions*hidden_size)
        h_dec.permute(1,0,2): [batch_size,25,128]
        '''
        enc = enc.repeat(self.out_length, 1, 1) #enc是tensor
        h_dec, _ = self.dec_lstm(enc) # lstm输出: output, (h_n, c_n) h是hidden c是cell
        h_dec = h_dec.permute(1, 0, 2) #为什么重新排序了？？

        '''
        h_dec: [batch_size,25,128]
        self.op():[batch_size,25,5]
        .permute():[25,batch_size,5]
        outputActivation: 只是求了一些exp(),tanh()还是：[25,batch_size,5]
        '''
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred) #输出的莫非是带有二元正态分布的东西没懂 out = torch.cat([muX, muY, sigX, sigY, rho],dim=2) #按dim=2 串起来
        return fut_pred

    ## Forward Pass 重点看这里，标出每层输出的维度和上面的维度
    # def forward(self,hist,nbrs,masks,lat_enc,lon_enc):
    def forward(self,egoHist_batch, nbrHist_batch, nbrIdx_batch, lat_batch, lon_batch):
        #----encoder ego车的hist, nbrs车的hist和idx编码----
        ''''
        Forward pass hist:维度dim
        egoHist_batch:[31,batch,2] = [maxlen,batch_size,2]  2因为x,y; batchsize=len(samples) 见utils.py的collate_fn
        a=self.ip_emb(egoHist_batch): [31,batch_size,32] = [seq_len历史和当前的31帧,batch_size,每一帧的特征input_embedding_size] 
        b=self.leaky_relu(a): 不变
        _,(hist_enc,_)=self.enc_lstm(b,(h0,c0会默认给0的)): 返回output,(hidden_n,cell_n) 这里取hist_enc=h_n: [1,batch_size,32] = [num_layers*num_directions=1*1,batch_size,hidden_size]
        hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]): [batch_size,32]
        '''
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(egoHist_batch))) # lstm输出: (output, (h_n, c_n)) h是hidden c是cell,只要h_n作为hist_enc
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]))) #.view是改shape维数, hist_enc.shape:[1,batch_size,32]
        '''
        Forward pass nbrs
        nbrHis_batch:[31,nbr_batch_size很多个ego的周围车数总和,2]
        self.ip_emb(nbrHist_batch):[31,nbr_batch_size,32=input_embedding_size] 
        self.enc_lstm():返回(output,(hn,cn)),取nbrs_enc=hn [1,nbr_batch_size,32] = [num_layers*num_directions=1*1,nbr_batch_size,hidden_size]
        nbrs_enc.view(nbr_batch_size,32): [nbr_batch_size,32]
        '''
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrHist_batch)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        '''
        Masked scatter:
        soc_enc: [batch_size,3,13,64]
        .masked_scatter_: [batch_size,3,13,64]
        .permute: [batch_size,64,13,3]
        ''' 
        soc_enc = torch.zeros_like(nbrIdx_batch).float() #zeo_like创建和masks维数一样的 float64的全0 tensor
        soc_enc = soc_enc.masked_scatter_(nbrIdx_batch, nbrs_enc) # nbrIdx_batch里有车就复制，没车空就不复制，从nbrs_enc复制，由nbrIdx_batch判断要不要复制，mask (BoolTensor) – the boolean mask; source (Tensor) – the tensor to copy from
        soc_enc = soc_enc.permute(0,3,2,1)


        #----encoder与decoder之间，与普通lstm encoder-decoder不同的地方----
        '''
        Apply convolutional social pooling: # cslstm用这个
        son_enc: [batch_size,64,13,3]作为input[batch_size,win=channel,hin=sequence_length,embedding_size] #有点乱
        self.soc_conv():  [batch_size,64,11,1]=[batch_size,output,hout,wout] hout=(hin-hkernel+2padding)/stride+1=(13-3+0)/1+1=11,wout=(3-3)+1=1
        self.conv_3x1(): [batch_size,16,9,1]=[batch_size,output,hout,wout] hout=(hin-hkernel+2padding)/stride+1=(11-3+0)/1+1=9,wout=(1-1)+1=1
        self.maxpool(): [batch_size,16,(9+padding[0])//2=5,(1+padding[1])/2=1]=[batch_size,16,5,1]
        .veiw(-1,5):就是变成[batch_size*16*5*1/80,80] 对了都算对了！！！！
        '''
        # soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        # soc_enc = soc_enc.view(-1,self.soc_embedding_size)
        
        '''
        Apply fc soc pooling # slstm改上面为下面这个
        soc_enc:[batch_size,64,13,3]
        .view(-1,64*13*3): [batch_size,64*13*3]
        .soc_fc: [batch_size,80]= [batch_size,social_embedding_size=5*conv_3x1_depth]
        '''
        soc_enc = soc_enc.contiguous() #view只能用在contiguous的variable上，连续的变量
        soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        '''
        Concatenate encodings
        soc_enc:[batch_size,64,13,3]->[batch_size,80]=[batch_size,soc_embedding_size]
        hist_enc:[batch_size,32]
        enc: [batch_size,112]= [batch_size,soc_embeddingsize+input_embedding_size]
        '''
        enc = torch.cat((soc_enc,hist_enc),1) #在dim=1上连接，除了dim=1其它都维度要一样

        #----decoder----
        if self.use_maneuvers: 
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc)) #归一化softmax
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_batch, lon_batch), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_batch)
                        lon_enc_tmp = torch.zeros_like(lon_batch)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1) #几个tensor按dim=1连起来 没看懂
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            '''
            enc: [batch_size,112]= [batch_size,soc_embeddingsize+input_embedding_size]
            fut_pred=.decode:[25,batch_size,5] # 5是μx,μy,e^σx,e^σy,tanh(ρ)
            '''
            fut_pred = self.decode(enc) #没用多模态，没有lat和lon maneuver的label
            return fut_pred
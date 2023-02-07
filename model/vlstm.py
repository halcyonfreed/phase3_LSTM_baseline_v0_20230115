import torch
import torch.nn as nn
#vanilla lstm不含conv social pooling和social pooling 没改呢

# Custom activation for output layer (Graves, 2015) # decode输出output用
def outputActivation(x): #x是future_pred 三维tensor
    muX = x[:,:,0:1] 
    muY = x[:,:,1:2]
    sigX = x[:,:,2:3]
    sigY = x[:,:,3:4]
    rho = x[:,:,4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho) # 二元正态分布，μx,μy,σx,σy,ρ相关系数
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2) #按dim=2 串起来
    return out
class lstm(nn.Module):
    '''lstm的encoder+decoder架构'''
    def __init__(self,args):
        super(lstm,self).__init__()
        self.args=args
        self.use_cuda=args['use_cuda']
        self.use_maneuvers=args['use_maneuvers']
        self.train_flag=args['train_flag']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.num_lat_classes = args['num_lat_classes']
        self.num_lon_classes = args['num_lon_classes']
        # self.soc_conv_depth = args['soc_conv_depth'] #
        # self.conv_3x1_depth = args['conv_3x1_depth'] #？        
        # self.dyn_embedding_size = args['dyn_embedding_size'] #? 是什么
        self.input_embedding_size = args['input_embedding_size'] 
        # self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth #？(13-4+1)//2=5 为什么//2

        # 定义network layer的weight
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size) # [2,input_embedding_size]
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1) # 输出[input_embedding_size,encoder_size]：Encoder LSTM:input_size：input_embedding_size，hidden_size: self.encoder_size, num_layers:1        
       
        # Vehicle dynamics embedding(这层是个什么东西，dynamics？？)
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size) #输出[encoder_size,dyn_embedding_size]

        # Convolutional social pooling layer and social embedding layer #比普通的lstm，就是改了这里？？就好了？？
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3) #in_channels：encoder_size,out_channels:soc_conv_depth,kernel_size 3*3 
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))#in_channels,out_channels,kernel_size
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0)) #kernel_size=(2*1), padding=(1,0)为什么
        
        # FC social pooling layer (for comparison):可以把上面的conv social pooling改成这个
        self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth) #[soc_conv_depth*13*3，5*conv_3x1_depth] 这个怎么算的

        # Decoder LSTM 
        if self.use_maneuvers:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size) #[soc_embedding_size+dyn_embedding_size+3+2,decoder_size]
        else:
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size) #[soc_embedding_size+dyn_embedding_size,decoder_size]

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size,5) #为什么是5啊
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)#[soc_embedding_size+dyn_embedding_size,3]
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)#[soc_embedding_size+dyn_embedding_size,2]

        # Activations:（neurons）
        self.leaky_relu = torch.nn.LeakyReLU(0.1) #negative_slope=0.1
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) #在dim=1这一维上归一化，这一维的sum=1

    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1) #enc是tensor
        h_dec, _ = self.dec_lstm(enc) # lstm输出: (output, (h_n, c_n)) h是hidden c是cell
        h_dec = h_dec.permute(1, 0, 2) #为什么重新排序了？？

        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = self.outputActivation(fut_pred) #输出的莫非是带有二元正态分布的东西没懂 out = torch.cat([muX, muY, sigX, sigY, rho],dim=2) #按dim=2 串起来
        return fut_pred

    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):

        #----------------encoder部分：自车的hist，nbrs的，mask（就是grid）
        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist))) # lstm输出: (output, (h_n, c_n)) h是hidden c是cell,只要h_n作为hist_enc
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2]))) #.view是改shape维数
        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        ## Masked scatter: mask是和grid相关的
        soc_enc = torch.zeros_like(masks).float() #zeo_like创建和masks维数一样的 float64的全0 tensor
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc) #从nbrs_enc复制，由Mask判断要不要复制，mask (BoolTensor) – the boolean mask; source (Tensor) – the tensor to copy from
        soc_enc = soc_enc.permute(0,3,2,1)


        #-----------------encoder与decoder之间，算普通lstm encoder的延伸
        # ## Apply convolutional social pooling:
        # soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        # soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        ## Apply fc soc pooling 全连接的对比实验，上面可以换成下面这个
        soc_enc = soc_enc.contiguous() #view只能用在contiguous的variable上，连续的变量
        soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        soc_enc = self.leaky_relu(self.soc_fc(soc_enc))

        #------------------Concatenate encodings:
        enc = torch.cat((soc_enc,hist_enc),1)


        #----------------decoder
        if self.use_maneuvers: #多模态，一个模态一个结果！！！啊啊啊啊啊感觉有些复杂
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc)) #归一化softmax
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1) #几个tensor按dim=1连起来 没看懂
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc) #没用多模态，没有lat和lon maneuver的label
            return fut_pred
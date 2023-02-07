'''
@author: halcyonfreed
@date: 20230115-
@comment: 
        define my own dataset: 定义自己的dataset

'''
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import scipy.io as scp #读mat文件

args={
    't_h':30, # length of track history
    't_f':50,# length of predicted trajectory
    # d_s可改,先改成2
    'd_s':1,# down sampling rate of all sequences 采样的步长，原来是2,减少计算量，改成1?  https://blog.csdn.net/hxxjxw/article/details/106175155
    'enc_size':64,  # size of encoder LSTM
    'grid_size':(13,3) # size of social context grid 3根车道 前后6车，一共约13辆车的车长
}

class ngsimDataset(Dataset):
    def __init__(self,mat_file,args=args):
        # 见Readme.md——数据说明.md看datatype,shape,含义
        self.D=scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = args['t_h']
        self.t_f = args['t_f'] 
        self.d_s =args['d_s']
        self.enc_size = args['enc_size']
        self.grid_size = args['grid_size']
    
    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx): # idx 某辆车某一时刻
        # Get features of ego vehicles
        dsId= self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,8:] #3*13格,[1,39], 每格一辆车vehID
        nbrsHist=[] #neighbours
        egoHist = self.getHistory(vehId,t,vehId,dsId) 
        egoFut = self.getFuture(vehId,t,dsId)

        # Get track histories of all neighbours in the grid
        for i in grid: # i: vehId
            nbrsHist.append(self.getHistory(i.astype(int), t,vehId,dsId))
        
        # Convert longitudinal/lateral class to one-hot vector
        ''' 要转成独热码one-hot vector
            因为标签隐含了距离信息会有问题
            比如标签1 2 3 隐含1、2比1、3之间更近
        '''
        #  self.D[idx,7]=1.0或者2.0, 默认是float所以要int()，np.eye默认是float所以astype(int)
        egoLonEnc=np.eye(2)[int(self.D[idx,7])-1].astype(int) # normal [1,0] brake [0,1]
        egoLatEnc=np.eye(3)[int(self.D[idx,6])-1].astype(int) # keep[1,0,0] left[0,1,0] right[0,0,1]
        # egoLonEnc = np.zeros([2]) # 刹车和非刹车 初始化[0,0]
        # egoLonEnc[int(self.D[idx, 7] - 1)] = 1 # normal是[1,0] brake是
        # egoLatEnc = np.zeros([3]) #保持 左右变道 所以是3 [1,0,0] [0,1,0] [0,0,1]
        # egoLatEnc[int(self.D[idx, 6] - 1)] = 1
        return egoHist,egoFut,nbrsHist,egoLatEnc,egoLonEnc

    def getHistory(self,nbrId,t,egoId,dsId): 
        '''
        nbrId: nbrs周围车id(也可以是自车)
        t:时刻
        egoId: ego自车vehid,只是用来求相对于自车为原点的相对坐标Δx,Δy
        dsId:datasetID
        '''
        if nbrId == 0: # 应该没车id=0
            return np.empty([0,2]) #就是空[],array([], shape=(0, 2), dtype=float64)
        else:
            if self.T.shape[1]<=nbrId-1:
                return np.empty([0,2])
            egoTrack = self.T[dsId-1][egoId-1].transpose() #原:一种特征一行，转成:一种特征一列，frameID,x,y是列
            nbrTrack = self.T[dsId-1][nbrId-1].transpose() 
            egoPos = egoTrack[np.where(egoTrack[:,0]==t)][0,1:3] # 取第0列frameID=t的所有行:, 再取第0行第1,2列（不到3列）x,y

            if nbrTrack.size==0 or np.argwhere(nbrTrack[:, 0] == t).size==0: #没车/这一时刻没车
                 return np.empty([0,2])
            else:
                start=np.maximum(0, np.argwhere(nbrTrack[:, 0] == t).item() - self.t_h)
                end=np.argwhere(nbrTrack[:, 0] == t).item() + 1 # +1因为[1:3]是1,2没3
                egoHist=nbrTrack[start:end:self.d_s,1:3]-egoPos # 相对自车当前时刻的相对坐标！！！

            if len(egoHist)<self.t_h//self.d_s +1: #历史太短了扔掉
                return np.empty([0,2])
            
            return egoHist
    
    def getFuture(self, vehId, t,dsId): 
        '''
        vehId: 当前这个idx在traj里的vehID号
        t: 第几帧
        dsId: 在哪个dataset里一共6个
        '''
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        egoPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        start = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s # 未来不含当前这一帧
        end = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)

        fut = vehTrack[start:end:self.d_s,1:3]-egoPos
        return fut

    def collate_fn(self,samples):
        '''
        1. pytorch官方的dataloader里的这个换成自定义的,why? 
        用来打包成batch的
        2. samples比如1024=batch_size辆车一起看,维度就是1024*5: 5是 (egoHist,egoFut,nbrsHist,egoLatEnc,egoLonEnc) 5个变量都要打包
        但这里乱七八糟的还是看不懂,为毛要这么搞啊，傻逼吧
        '''
        # 1 初始化nbrsHist维度
        '''Initialize neighbors and neighbors length batches:'''
        nbr_batch_size = 0
        for _,_,nbrs,_,_ in samples: 
            nbr_batch_size += sum([len(nbrs[i])!=0 for i in range(len(nbrs))]) # 第i辆ego车的周围车辆数len(nbrs[i]，很多个ego的周围车数总和
        maxlen = self.t_h//self.d_s + 1  # 31
        nbrHist_batch = torch.zeros(maxlen,nbr_batch_size,2)  # dim: [31,nbr_batch_size,2]

        ''' Initialize social mask batch: social mask就是周围车的grid有关的'''
        pos = [0, 0]  # dim:[1,2]存(Δx,Δy)
        nbrIdx_batch = torch.zeros(len(samples), self.grid_size[1],self.grid_size[0],self.enc_size) #dim: [batch_size,3,13,64]
        nbrIdx_batch = nbrIdx_batch.byte()
        
        # 2 初始化egoHist,egoFut,egoLatEnc,egoLonEnc 维度
        '''Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches
            创建形状自定义torch.zeros()存信息'''
        egoHist_batch = torch.zeros(maxlen,len(samples),2) # [31,batch_size,2]
        egoFut_batch = torch.zeros(self.t_f//self.d_s,len(samples),2) # [50//1,batch_size,2]
        output_batch = torch.zeros(self.t_f//self.d_s,len(samples),2) # [50//1,batch_size,2]
        lat_batch = torch.zeros(len(samples),3) # [batch_size,3]
        lon_batch = torch.zeros(len(samples), 2) # [batch_size,2] 

        # 3 打包(hist, fut, nbrs, lat_enc, lon_enc)
        count = 0
        for sampleId,(hist, fut, nbrs, lat_enc, lon_enc) in enumerate(samples):
            # Set up history, future, lateral maneuver and longitudinal maneuver batches
            egoHist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0]) # 0存x
            egoHist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1]) # 1存y
            egoFut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            egoFut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            output_batch[0:len(fut),sampleId,:] = 1 # 可改 默认ouput的x,y都设成1
            lat_batch[sampleId,:] = torch.from_numpy(lat_enc)
            lon_batch[sampleId, :] = torch.from_numpy(lon_enc)

            # Set up neighbor, neighbor sequence length, and nbrPos（mask） batches:
            for id,nbr in enumerate(nbrs):
                if len(nbr)!=0:
                    nbrHist_batch[0:len(nbr),count,0] = torch.from_numpy(nbr[:, 0]) # count存周围车,0存x
                    nbrHist_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1]) # 1存y
                    # 第30个格子的新编号是(4,2)
                    pos[0] = id % self.grid_size[0] # 30%13=4 
                    pos[1] = id // self.grid_size[0] # 30//13=2
                    nbrIdx_batch[sampleId,pos[1],pos[0],:] = torch.ones(self.enc_size).byte() # byte从64float到8uint整型
                    count+=1
        return egoHist_batch, nbrHist_batch, nbrIdx_batch, lat_batch, lon_batch, egoFut_batch, output_batch
        
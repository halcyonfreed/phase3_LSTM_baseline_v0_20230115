import torch
import math

## train用的NLL：Batchwise NLL loss, uses mask for variable output lengths(输出长度不同，用 mask)
def maskedNLL(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask) # acc是什么？？mask就是输出，设和输出的tensor形状大小一样的zero tensor
    muX = y_pred[:,:,0] # μx 横向坐标x的期望（均值）
    muY = y_pred[:,:,1] # 径向坐标y的期望（均值）
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3] # σy
    rho = y_pred[:,:,4] # ρ
    x = y_gt[:,:, 0] # 横向坐标x真值
    y = y_gt[:,:, 1]
    # If we represent likelihood in feet^(-1): #无论feet还是m为单位，公式全是错的，垃圾玩意儿，浪费我时间
    # out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - torch.log(sigX*sigY*ohr) + 1.8379
    # If we represent likelihood in m^(-1): #统一用m
    # out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160

    # correct hsy改12.8
    # avoid_infinitive = 1e-6
    # ohr = torch.from_numpy(1/np.maximum(1 - rho * rho, avoid_infinitive)) #avoid infinite values, ohr=1/((ρ^2)^0.5)
    ohr = torch.pow((1-torch.pow(rho,2)),-1)  #1/(1-ρ2)
    out =0.093( 0.5*ohr * (
        torch.pow(x-muX,2)/ torch.pow(sigX,2) + torch.pow(y-muY,2)/ torch.pow(sigY,2) -
        2 * rho *(x-muX)* (y-muY) / (sigX * sigY)
            ) + torch.log(sigX * sigY) - 0.5*torch.log(torch,pow(ohr,-1)) + math.log(math.pi*2)) 

    acc[:,:,0] = out #??
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal


## valid用的NLL：## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
def maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes = 2,use_maneuvers = False, avg_along_time = False):
    if use_maneuvers:
        acc = torch.zeros(op_mask.shape[0],op_mask.shape[1],num_lon_classes*num_lat_classes).cuda()
        count = 0
        for k in range(num_lon_classes):
            for l in range(num_lat_classes):
                wts = lat_pred[:,l]*lon_pred[:,k] # wts什么东西？？waiting second?
                wts = wts.repeat(len(fut_pred[0]),1)
                
                y_pred = fut_pred[k*num_lat_classes + l]
                y_gt = fut
                muX = y_pred[:, :, 0]
                muY = y_pred[:, :, 1]
                sigX = y_pred[:, :, 2]
                sigY = y_pred[:, :, 3]
                rho = y_pred[:, :, 4]
                x = y_gt[:, :, 0]
                y = y_gt[:, :, 1]

                ohr = torch.pow((1-torch.pow(rho,2)),-1) 
                out =0.093( 0.5*ohr * (
                    torch.pow(x-muX,2)/ torch.pow(sigX,2) + torch.pow(y-muY,2)/ torch.pow(sigY,2) -
                    2 * rho *(x-muX)* (y-muY) / (sigX * sigY)
                        ) + torch.log(sigX * sigY) - 0.5*torch.log(torch,pow(ohr,-1)) + math.log(math.pi*2))


                #行为的准确度
                acc[:, :, count] = out+ torch.log(wts) #改了这里12.8 hsy
                count+=1  

        acc = -logsumexp(acc, dim = 2)
        acc = acc * op_mask[:,:,0] #op_mask就是output



        if avg_along_time: #有毛病，为什么要along_time？？？？？？
            lossVal = torch.sum(acc) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc,dim=1)
            counts = torch.sum(op_mask[:,:,0],dim=1)
            return lossVal,counts
    else:
        acc = torch.zeros(op_mask.shape[0], op_mask.shape[1], 1).cuda()
        y_pred = fut_pred
        y_gt = fut
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        sigX = y_pred[:, :, 2]
        sigY = y_pred[:, :, 3]
        rho = y_pred[:, :, 4]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]

        # correct hsy改12.8
        ohr = torch.pow((1-torch.pow(rho,2)),-1) 
        out =0.093*( 0.5*ohr * (
                torch.pow(x-muX,2)/ torch.pow(sigX,2) + torch.pow(y-muY,2)/ torch.pow(sigY,2) -
                2 * rho *(x-muX)* (y-muY) / (sigX * sigY)
                    ) + torch.log(sigX * sigY) - 0.5*torch.log(torch,pow(ohr,-1)) + math.log(math.pi*2))
             #单位是m^2


        acc[:, :, 0] = out
        acc = acc * op_mask[:, :, 0:1]
        if avg_along_time:
            lossVal = torch.sum(acc[:, :, 0]) / torch.sum(op_mask[:, :, 0])
            return lossVal
        else:
            lossVal = torch.sum(acc[:,:,0], dim=1)
            counts = torch.sum(op_mask[:, :, 0], dim=1)
            return lossVal,counts

## train用的RMSE： Batchwise MSE loss, uses mask for variable output lengths
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = 0.3048*torch.pow((torch.pow(x-muX, 2) + torch.pow(y-muY, 2)),0.5)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask) #单位是m
    return lossVal

## test/valid用的MSE loss for complete sequence, outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def maskedMSETest(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.3048*torch.pow((torch.pow(x-muX, 2) + torch.pow(y-muY, 2)),0.5)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    acc = acc * mask
    lossVal = torch.sum(acc[:,:,0],dim=1)
    counts = torch.sum(mask[:,:,0],dim=1)
    return lossVal, counts


## Helper function for log sum exp calculation:
def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    # s, _ = torch.max(inputs, dim=dim, keepdim=True) #s是取max值
    # outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log() #没看懂？应该是max+InΣe^(x-max) 为什么
    outputs =inputs.sum(dim=dim, keepdim=True)
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
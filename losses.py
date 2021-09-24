import torch
import torch.nn.functional as F
import numpy as np
from numpy import *
import math

def computeErrorStdMuOverPatientDimMean(predicitons, gts, hPixelSize=3.24, goodBScansInGtOrder=None):

    device = predicitons.device
    B,N, W = predicitons.shape # where N is numSurface
    absError = torch.abs(predicitons-gts)

    if goodBScansInGtOrder is None:
        P = B // slicesPerPatient
        absErrorPatient = torch.zeros((P,N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient:(p + 1) * slicesPerPatient, ], dim=(0,2))*hPixelSize
    else:
        P = len(goodBScansInGtOrder)
        absErrorPatient = torch.zeros((P, N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient+goodBScansInGtOrder[p][0]:p * slicesPerPatient+goodBScansInGtOrder[p][1], ], dim=(0,2))*hPixelSize

    stdSurface, muSurface = torch.std_mean(absErrorPatient, dim=0)
    std, mu = torch.std_mean(absErrorPatient)
    return stdSurface, muSurface, std,mu

class NCC:


    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [17] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win])

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class NCC_oct_metric:


    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true):
        #print(y_true.shape)
        losses = 0
        for i in range(y_true.shape[3]-1):
            
            I = y_true[:,:,:,i].unsqueeze(1)
            J = y_true[:,:,:,i+1].unsqueeze(1)
            # get dimension of volume
            # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
            ndims = len(list(I.size())) - 2
            assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

            # set window size
            win = [9] * ndims if self.win is None else self.win

            # compute filters
            sum_filt = torch.ones([1, 1, *win]).to("cuda")

            pad_no = math.floor(win[0]/2)

            if ndims == 1:
                stride = (1)
                padding = (pad_no)
            elif ndims == 2:
                stride = (1,1)
                padding = (pad_no, pad_no)
            else:
                stride = (1,1,1)
                padding = (pad_no, pad_no, pad_no)

            # get convolution function
            conv_fn = getattr(F, 'conv%dd' % ndims)

            # compute CC squares
            I2 = I * I
            J2 = J * J
            IJ = I * J

            I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + 1e-5)
            #print(torch.mean(cc))
            losses += -torch.mean(cc)
            #print(cc.shape)
        #print(losses.shape)
        
        return losses / y_true.shape[3] - 1
        
class NCC_oct:


    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true):
        #print(y_true.shape)
        losses = 0
        for i in range(y_true.shape[3]-1):
            
            I = y_true[:,:,:,i].unsqueeze(1)
            J = y_true[:,:,:,i+1].unsqueeze(1)
            # get dimension of volume
            # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
            ndims = len(list(I.size())) - 2
            assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

            # set window size
            win = [33] * ndims if self.win is None else self.win

            # compute filters
            sum_filt = torch.ones([1, 1, *win]).to("cuda")

            pad_no = math.floor(win[0]/2)

            if ndims == 1:
                stride = (1)
                padding = (pad_no)
            elif ndims == 2:
                stride = (1,1)
                padding = (pad_no, pad_no)
            else:
                stride = (1,1,1)
                padding = (pad_no, pad_no, pad_no)

            # get convolution function
            conv_fn = getattr(F, 'conv%dd' % ndims)

            # compute CC squares
            I2 = I * I
            J2 = J * J
            IJ = I * J

            I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            win_size = np.prod(win)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + 1e-5)
            losses += -torch.mean(cc)

        
        return losses / y_true.shape[4] - 1

 
class Grad_bscan:
    def loss(self, y_true):
        I = y_true[:,:,:,:-1]
        J = y_true[:,:,:,1:]
        loss = torch.mean(torch.pow((I-J), 2))
        return loss
 
class MSE_bscan:
    def loss(self, y_true):
        losses = 0
        for i in range(y_true.shape[3]-1):
            
            I = y_true[:,:,:,i].unsqueeze(1)
            J = y_true[:,:,:,i+1].unsqueeze(1)

            cc = torch.pow(I - J, 2)
            
            losses += torch.mean(cc)
        return losses / y_true.shape[3]
        
class Dice_intraBscan:
    def loss(self, y_true):
        I = y_true[:,:,:,:-1]
        J = y_true[:,:,:,1:]
        return (I-J).pow(2).mean()
        
class MSE:


    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:


    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class GeneralizedDiceLoss():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):

        B,K,H,W,D = inputx.shape
        device = inputx.device

        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()
        for k in range(0,K):
            targetProb[:,k,:,:,:] = torch.where(k ==target, torch.ones_like(target), targetProb[:,k,:,:,:])

        W = torch.zeros(K, device=device, dtype=torch.float64).float()
        for k in range(0,K):
            W[k] = torch.tensor(1.0).float()/((k == target).sum()**2)

        sumDims = (0,2,3,4)
        GDL = 1.0-2.0*((inputx*targetProb).sum(dim=sumDims)*W).sum()/((inputx+targetProb).sum(dim=sumDims)*W).sum()
        return GDL
 
class MultiLayerCrossEntropyLoss():
    def __init__(self, weight=None):
        self.m_weight = weight  # B,N,H,W, where N is nLayer, instead of num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):

        B,N,H,W, D = inputx.shape
        assert (B,H,W,D) == target.shape
        device = inputx.device

        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()  # size: B,N,H,W
        for k in range(0, N): # N layers
            targetProb[:, k, :, :, :] = torch.where(k == target, torch.ones_like(target).float(), targetProb[:, k, :, :, :])

        e = 1e-6
        inputx = inputx+e
        inputx = torch.where(inputx>=1, (1-e)*torch.ones_like(inputx).float(), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log()+(1-targetProb)*(1-inputx).log())).mean()
        else:
            loss = -(targetProb * inputx.log()+(1-targetProb)*(1-inputx).log()).mean()
        return loss
 
class MultiSurfaceCrossEntropyLoss():
    def __init__(self,  weight=None):
        self.m_weight = weight   # B,N,H,W, where N is the num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):

        B,N,H,W, D = inputx.shape
        device = inputx.device

        targetIndex = (target +0.5).long().unsqueeze(dim=-3) # size: B,N,1,W, D

        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device)  # size: B,N,H,W
        targetProb.scatter_(2, targetIndex, torch.ones_like(targetIndex))

        e = 1e-6
        inputx = inputx + e
        inputx = torch.where(inputx >= 1, (1 - e) * torch.ones_like(inputx), inputx)
        
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log())).mean()
        else:
            loss = -(targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log()).mean()
            
        return loss
        
class Grad_2d:

    def __init__(self, penalty='l2', loss_mult=None, weight = [1,1,1]):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.weight = weight
        
    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
        
        d = torch.mean(dx, (0,2,3)) * 0.04 + torch.mean(dy, (0,2,3))
        d = torch.mean(d * self.weight)
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return d

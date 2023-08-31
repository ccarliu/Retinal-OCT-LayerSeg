import torch
import torch.nn.functional as F
import numpy as np
from numpy import *
import math

def computeErrorStdMuOverPatientDimMean(predicitons, gts, hPixelSize=3.24, goodBScansInGtOrder=None):
    '''
    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension


    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :param goodBScansInGtOrder:
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             muPatient: (NumPatient) dimension, mean for each patient
             stdPatient: (NumPatient) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
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
    # size of stdSurface, muSurface: [N]
    std, mu = torch.std_mean(absErrorPatient)
    return stdSurface, muSurface, std,mu

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

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
    """
    Local (over window) normalized cross correlation loss.
    """

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
        
        return losses / 39
        
class NCC_oct:
    """
    Local (over window) normalized cross correlation loss.
    """

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
        
        return losses / 20

 
class Grad_bscan:
    def loss(self, y_true):
        #print(y_true.shape)
        I = y_true[:,:,:,:-1]
        J = y_true[:,:,:,1:]
        loss = torch.mean(torch.pow((I-J), 2))
            #print(cc.shape)
        return loss
 
class MSE_bscan:
    def loss(self, y_true):
        #print(y_true.shape)
        losses = 0
        for i in range(y_true.shape[3]-1):
            
            I = y_true[:,:,:,i].unsqueeze(1)
            J = y_true[:,:,:,i+1].unsqueeze(1)

            cc = torch.pow(I - J, 2)
            
            losses += torch.mean(cc)
            #print(cc.shape)
        return losses / 20 
        
class Dice_intraBscan:
    def loss(self, y_true):
        # print(y_true.shape)
        # print(y_true[:,:, 44, 20])
        I = y_true[:,:,:,:-1]
        J = y_true[:,:,:,1:]
        #print(I.shape, J.shape)
        return (I-J).pow(2).mean()
        
class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class GeneralizedDiceLoss_lesion():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target, mask = None):
        '''
         target: for N surfaces, surface 0 in its exact location marks as 1, surface N-1 in its exact location marks as N.
                region pixel between surface i and surface i+1 marks as i.

        :param inputx: float32 tensor, size of (B,K,H,W), where K is the number of classes. inputx is softmax probability along dimension K.
        :param target: long tensor, size of (B,H,W), where each element has a long value of [0,K) indicate the belonging class
        :return: a float scalar of mean dice over all classes and over batchSize.

        '''
        B,K,H,W,D = inputx.shape
        #assert (B,H,W) == target.shape
        #assert K == target.max()+1
        device = inputx.device

        # convert target of size(B,H,W) into (B,K,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()
        for k in range(0,K):
            targetProb[:,k,:,:,:] = torch.where(k ==target, torch.ones_like(target), targetProb[:,k,:,:,:])

        # compute weight of each class
        W = torch.zeros(K, device=device, dtype=torch.float64).float()
        for k in range(0,K):
            W[k] = torch.tensor(1.0).float()/((k == target).sum()**2)
        #print(inputx.shape, targetProb.shape, "ttk")
        # print(mask.shape) bs 1 1 48 11
        mask = torch.cat([mask.expand(mask.shape[0], 9, *mask.shape[2:]), torch.ones_like(mask)], dim = 1)
        # generalized dice loss
        #print(inputx.shape) # bs, 10(layers), 224, 48, 11
        sumDims = (0,2,3,4)
        if mask is None:
            GDL = 1.0-2.0*((inputx*targetProb).sum(dim=sumDims)*W).sum()/((inputx+targetProb).sum(dim=sumDims)*W).sum()
        else:
            GDL = 1.0-2.0*((inputx*targetProb*mask).sum(dim=sumDims)*W).sum()/(((inputx+targetProb)*mask).sum(dim=sumDims)*W).sum()
        
        return GDL
        
# support multiclass generalized Dice Loss
class GeneralizedDiceLoss():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target, mask = None):
        '''
         target: for N surfaces, surface 0 in its exact location marks as 1, surface N-1 in its exact location marks as N.
                region pixel between surface i and surface i+1 marks as i.

        :param inputx: float32 tensor, size of (B,K,H,W), where K is the number of classes. inputx is softmax probability along dimension K.
        :param target: long tensor, size of (B,H,W), where each element has a long value of [0,K) indicate the belonging class
        :return: a float scalar of mean dice over all classes and over batchSize.

        '''
        B,K,H,W,D = inputx.shape
        #assert (B,H,W) == target.shape
        #assert K == target.max()+1
        device = inputx.device

        # convert target of size(B,H,W) into (B,K,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()
        for k in range(0,K):
            targetProb[:,k,:,:,:] = torch.where(k ==target, torch.ones_like(target), targetProb[:,k,:,:,:])

        # compute weight of each class
        W = torch.zeros(K, device=device, dtype=torch.float64).float()
        for k in range(0,K):
            W[k] = torch.tensor(1.0).float()/((k == target).sum()**2)
        #print(inputx.shape, targetProb.shape, "ttk")
        # generalized dice loss
        sumDims = (0,2,3,4)
        if mask is None:
            GDL = 1.0-2.0*((inputx*targetProb).sum(dim=sumDims)*W).sum()/((inputx+targetProb).sum(dim=sumDims)*W).sum()
        else:
            GDL = 1.0-2.0*((inputx*targetProb*mask).sum(dim=sumDims)*W).sum()/(((inputx+targetProb)*mask).sum(dim=sumDims)*W).sum()
        
        return GDL

        
class MultiLayerCrossEntropyLoss_lesion():
    def __init__(self, weight=None):
        self.m_weight = weight  # B,N,H,W, where N is nLayer, instead of num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target, mask = None):
        '''
         pixel-wised multiLayer cross entropy.
        :param inputx:  N-layer probability of size: B,N,H,W
        :param target:  long tensor, size of (B,H,W), where each element has a long value of [0,N-1] indicate the belonging class
        :return: a scalar of loss
        '''
        B,N,H,W, D = inputx.shape
        assert (B,H,W,D) == target.shape
        device = inputx.device

        # convert target of size(B,H,W) into (B,N,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()  # size: B,N,H,W
        for k in range(0, N): # N layers
            targetProb[:, k, :, :, :] = torch.where(k == target, torch.ones_like(target).float(), targetProb[:, k, :, :, :])
        
        mask = torch.cat([mask.expand(mask.shape[0], 9, *mask.shape[2:]), torch.ones_like(mask)], dim = 1)

        e = 1e-6
        # 1e-8 is not ok, A=(1-e)*torch.ones_like(inputx) will still 1. and (1-A).log() will get -inf.
        inputx = inputx+e
        inputx = torch.where(inputx>=1, (1-e)*torch.ones_like(inputx).float(), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log()+(1-targetProb)*(1-inputx).log()))
        else:
            loss = -(targetProb * inputx.log()+(1-targetProb)*(1-inputx).log())
        if mask is not None:
            loss = loss * mask
        return loss.mean()
        
# support multiclass CrossEntropy Loss
class MultiLayerCrossEntropyLoss():
    def __init__(self, weight=None):
        self.m_weight = weight  # B,N,H,W, where N is nLayer, instead of num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target, mask = None):
        '''
         pixel-wised multiLayer cross entropy.
        :param inputx:  N-layer probability of size: B,N,H,W
        :param target:  long tensor, size of (B,H,W), where each element has a long value of [0,N-1] indicate the belonging class
        :return: a scalar of loss
        '''
        B,N,H,W, D = inputx.shape
        assert (B,H,W,D) == target.shape
        device = inputx.device

        # convert target of size(B,H,W) into (B,N,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device).float()  # size: B,N,H,W
        for k in range(0, N): # N layers
            targetProb[:, k, :, :, :] = torch.where(k == target, torch.ones_like(target).float(), targetProb[:, k, :, :, :])

        e = 1e-6
        # 1e-8 is not ok, A=(1-e)*torch.ones_like(inputx) will still 1. and (1-A).log() will get -inf.
        inputx = inputx+e
        inputx = torch.where(inputx>=1, (1-e)*torch.ones_like(inputx).float(), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log()+(1-targetProb)*(1-inputx).log()))
        else:
            loss = -(targetProb * inputx.log()+(1-targetProb)*(1-inputx).log())
        if mask is not None:
            loss = loss * mask
        return loss.mean()
 
class MultiSurfaceCrossEntropyLoss():
    def __init__(self,  weight=None):
        self.m_weight = weight   # B,N,H,W, where N is the num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target, mask = None):
        '''
        multiclass surface location cross entropy.
        this loss expect the corresponding prob at target location has maximum.

        :param inputx:  softmax probability along H dimension, in size: B,N,H,W
        :param target:  size: B,N,W; indicates surface location
        :return: a scalar
        '''
        B,N,H,W, D = inputx.shape
        #assert (B,N,W) == target.shape
        device = inputx.device

        targetIndex = (target +0.5).long().unsqueeze(dim=-3) # size: B,N,1,W, D

        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device)  # size: B,N,H,W
        #print(targetIndex.shape, targetProb.shape)
        targetProb.scatter_(2, targetIndex, torch.ones_like(targetIndex))

        e = 1e-6
        inputx = inputx + e
        #print(inputx.shape, target.shape)
        inputx = torch.where(inputx >= 1, (1 - e) * torch.ones_like(inputx), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log()))
        else:
            loss = -(targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log())
        if mask is not None:
            loss = loss * mask
        return loss.mean()
        
class Grad_2d:
    """
    N-D gradient loss.
    """

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
        #print(y_pred.shape)
        
        d = torch.mean(dx, (0,2,3)) * 0.1 + torch.mean(dy, (0,2,3))
        #print(d)
        d = torch.mean(d * self.weight)
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return d

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

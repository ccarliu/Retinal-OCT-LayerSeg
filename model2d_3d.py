import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *

class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)

        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        
        # We only focus on shift in one dimension, so the other two shift is zero.
        oshift1 = torch.zeros_like(flow)
        oshift2 = torch.zeros_like(flow)
        flow = torch.cat([flow, oshift1, oshift2],1)

        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def computeMuVariance(x, layerMu=None, layerConf=None): # without square weight
    '''
    Compute the mean and variance along H direction of each surface.
    '''
    
    A =3.0  # weight factor to balance surfaceMu and LayerMu.

    device = x.device
    B,N,H,W,D = x.size() # Num is the num of surface for each patientD

    # compute mu
    Y = torch.arange(H).view((1,1,H,1,1)).expand(x.size()).to(device=device, dtype=torch.int16)

    # use slice method to compute P*Y
    for b in range(B):
        if 0==b:
            PY = (x[b,]*Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (x[b,]*Y[b,]).unsqueeze(dim=0)))
            
    mu = torch.sum(PY, dim=-3, keepdim=True) # size: B,N,1,W
    del PY  # hope to free memory.
    Mu = mu.expand(x.size())

    # this slice method is to avoid using big GPU memory .
    for b in range(B):
        if 0==b:
            sigma2 = torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-3,keepdim=False).unsqueeze(dim=0)
        else:
            sigma2 = torch.cat((sigma2, torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-3,keepdim=False).unsqueeze(dim=0)))

    return mu.squeeze(dim=-3),sigma2

def logits2Prob(x, dim):
    # convert logits to probability for input x
    xMaxDim, _ = torch.max(x, dim=dim, keepdim=True)
    xMaxDim = xMaxDim.expand_as(x)
    prob = F.softmax(x - xMaxDim, dim=dim)  # using inputMaxDim is to avoid overflow.
    return prob

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 4:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)
    
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out
        
class LUConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm2d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out

def _make_nConv_2d(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv2d(in_channel, depth,act)
        layer2 = LUConv2d(depth, depth,act)
    else:
        layer1 = LUConv2d(in_channel, depth,act)
        layer2 = LUConv2d(depth, depth,act)

    return nn.Sequential(layer1,layer2)
        
def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, depth,act)
        layer2 = LUConv(depth, depth,act)
    else:
        layer1 = LUConv(in_channel, depth,act)
        layer2 = LUConv(depth, depth,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act, pol = True):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv_2d(in_channel, depth,act)
        self.maxpool = nn.MaxPool2d(2)
        self.current_depth = depth
        self.pol = pol

    def forward(self, x):
        if not self.pol:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=(2,2,1), stride=(2,2,1))
        self.ops = _make_nConv(outChans*2,outChans, act, double_chnnel=False)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.final_conv(x)
        return out

class UNet3D_dual_up_hc(nn.Module):

    # down_sample is 2d and up_sample is 3d
    
    def __init__(self, n_layer=9,act='relu', input_size = (320, 48, 40)):
        super(UNet3D_dual_up_hc, self).__init__()
        self.n_layer = n_layer
        
        self.down_tr32 = DownTransition(1,32,act)
        self.down_tr64 = DownTransition(32,64,act)
        self.down_tr128 = DownTransition(64,128,act)
        self.down_trf = DownTransition(128,128,act, pol = False)

        self.up_tr256 = UpTransition(128, 128, 2,act)
        self.up_tr128 = UpTransition(128, 64, 1,act)
        self.up_tr64 = UpTransition(64,32,0,act)
        
        self.up1_tr256 = UpTransition(128, 128, 2,act)
        self.up1_tr128 = UpTransition(128, 64, 1,act)
        self.up1_tr64 = UpTransition(64,32,0,act)
        
        self.out_tr = OutputTransition(32, n_layer)
        self.out_tr2 = OutputTransition(32, n_layer+1)
        self.out_tr3 = OutputTransition(32, 1)
        
        self.out_pool = nn.AdaptiveAvgPool2d(1)
        
        # trans for each level features
        self.transformer = SpatialTransformer2((input_size[0], input_size[1], input_size[2])) 
        self.transformer2 = SpatialTransformer2((input_size[0]//2, input_size[1]//2, input_size[2]))
        self.transformer4 = SpatialTransformer2((input_size[0]//4, input_size[1]//4, input_size[2]))
        self.transformer8 = SpatialTransformer2((input_size[0]//8, input_size[1]//8, input_size[2]))

    def forward(self, xr):  
        # 3d -> 2d
        bs = xr.shape[0]
        x = xr.permute(0,4,1,2,3)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        
        # feature extractor
        out32, skip_out32 = self.down_tr32(x)
        out64,skip_out64 = self.down_tr64(out32)
        out128,skip_out128 = self.down_tr128(out64)
        outf,skip_outf = self.down_trf(out128)
    
        # 2d -> 3d
        skip_out128 = torch.reshape(skip_out128, (bs, skip_out128.shape[0] // bs, skip_out128.shape[1], skip_out128.shape[2], skip_out128.shape[3])).permute(0,2,3,4,1)
        outf = torch.reshape(outf, (bs, outf.shape[0] // bs, outf.shape[1], outf.shape[2], outf.shape[3])).permute(0,2,3,4,1)
        skip_out64 = torch.reshape(skip_out64, (bs, skip_out64.shape[0] // bs, skip_out64.shape[1], skip_out64.shape[2], skip_out64.shape[3])).permute(0,2,3,4,1)
        skip_out32 = torch.reshape(skip_out32, (bs, skip_out32.shape[0] // bs, skip_out32.shape[1], skip_out32.shape[2], skip_out32.shape[3])).permute(0,2,3,4,1)
        
        # alignment network
        out_up1_128 = self.up1_tr256(outf.detach(),skip_out128.detach())
        out_up1_64 = self.up1_tr128(out_up1_128, skip_out64.detach())
        out_up1_32 = self.up1_tr64(out_up1_64, skip_out32.detach())
        flow_out = self.out_tr3(out_up1_32).squeeze()
        
        # flow
        flow = torch.mean(flow_out, dim = [1,2], keepdim = True)
        flow = flow - torch.mean(flow, dim = 3, keepdim = True).detach()
        flows = flow.expand(xr.squeeze().shape)

        x_out = self.transformer(xr, flows.unsqueeze(1)).squeeze(1)        
        
        # flows for each level of direct connection feature
        flows18 = flow.unsqueeze(1).expand((outf.shape[0], 1, outf.shape[2], outf.shape[3], outf.shape[4])) * 1/8
        flows14 = flow.unsqueeze(1).expand((skip_out128.shape[0], 1, skip_out128.shape[2], skip_out128.shape[3], skip_out128.shape[4])) * 1/4
        flows12 = flow.unsqueeze(1).expand((skip_out64.shape[0], 1, skip_out64.shape[2], skip_out64.shape[3], skip_out64.shape[4])) * 1/2

        # layer segmentation network
        outfs = self.transformer8(outf, flows18)
        skip_out128s = self.transformer4(skip_out128, flows14)
        out_up_128 = self.up_tr256(outfs,skip_out128s)
        skip_out64s = self.transformer2(skip_out64, flows12)
        out_up_64 = self.up_tr128(out_up_128, skip_out64s)
        skip_out32s = self.transformer(skip_out32, flows.unsqueeze(1))
        out_up_32 = self.up_tr64(out_up_64, skip_out32s)
        
        # out
        s = self.out_tr(out_up_32)
        layer_out = self.out_tr2(out_up_32)
        layerProb = logits2Prob(layer_out, 1)
        out = logits2Prob(s, 2)

        mu, sigma2 = computeMuVariance(out, layerMu=None, layerConf=None)
        S = mu.clone()
        
        # relu for Topology Guarantee
        for i in range(1,3):
            S[:,i,:, :] = torch.where(S[:, i,:, :]< S[:,i-1,:,:], S[:,i-1,:, :], S[:,i,:,:])

        return S, out, layerProb, mu, x_out, flow
        
class UNet3D_dual_up(nn.Module):

    # down_sample is 2d and up_sample is 3d
    
    def __init__(self, n_layer=3,act='relu', input_size = (320, 48, 40)):
        super(UNet3D_dual_up, self).__init__()
        self.n_layer = n_layer
        
        self.down_tr32 = DownTransition(1,32,act)
        self.down_tr64 = DownTransition(32,64,act)
        self.down_tr128 = DownTransition(64,128,act)
        self.down_trf = DownTransition(128,128,act, pol = False)

        self.up_tr256 = UpTransition(128, 128, 2,act)
        self.up_tr128 = UpTransition(128, 64, 1,act)
        self.up_tr64 = UpTransition(64,32,0,act)
        
        self.up1_tr256 = UpTransition(128, 128, 2,act)
        self.up1_tr128 = UpTransition(128, 64, 1,act)
        self.up1_tr64 = UpTransition(64,32,0,act)
        
        self.out_tr = OutputTransition(32, n_layer)
        self.out_tr2 = OutputTransition(32, n_layer+1)
        self.out_tr3 = OutputTransition(32, 1)
        
        self.out_pool = nn.AdaptiveAvgPool2d(1)
        self.transformer = SpatialTransformer2((input_size[0], input_size[1], input_size[2])) 
        self.transformer2 = SpatialTransformer2((input_size[0]//2, input_size[1]//2, input_size[2]))
        self.transformer4 = SpatialTransformer2((input_size[0]//4, input_size[1]//4, input_size[2]))
        self.transformer8 = SpatialTransformer2((input_size[0]//8, input_size[1]//8, input_size[2]))

    def forward(self, xr):  
        # 3d -> 2d
        bs = xr.shape[0]
        x = xr.permute(0,4,1,2,3)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        
        # feature extractor
        out32, skip_out32 = self.down_tr32(x)
        out64,skip_out64 = self.down_tr64(out32)
        out128,skip_out128 = self.down_tr128(out64)
        outf,skip_outf = self.down_trf(out128)
    
        # 2d -> 3d
        skip_out128 = torch.reshape(skip_out128, (bs, skip_out128.shape[0] // bs, skip_out128.shape[1], skip_out128.shape[2], skip_out128.shape[3])).permute(0,2,3,4,1)
        outf = torch.reshape(outf, (bs, outf.shape[0] // bs, outf.shape[1], outf.shape[2], outf.shape[3])).permute(0,2,3,4,1)
        skip_out64 = torch.reshape(skip_out64, (bs, skip_out64.shape[0] // bs, skip_out64.shape[1], skip_out64.shape[2], skip_out64.shape[3])).permute(0,2,3,4,1)
        skip_out32 = torch.reshape(skip_out32, (bs, skip_out32.shape[0] // bs, skip_out32.shape[1], skip_out32.shape[2], skip_out32.shape[3])).permute(0,2,3,4,1)
        
        # alignment network
        out_up1_128 = self.up1_tr256(outf.detach(),skip_out128.detach())
        out_up1_64 = self.up1_tr128(out_up1_128, skip_out64.detach())
        out_up1_32 = self.up1_tr64(out_up1_64, skip_out32.detach())
        flow_out = self.out_tr3(out_up1_32).squeeze()
        
        # flow
        flow = torch.mean(flow_out, dim = [1,2], keepdim = True)
        flow = flow - torch.mean(flow, dim = 3, keepdim = True).detach()
        flows = flow.expand(xr.squeeze().shape)

        x_out = self.transformer(xr, flows.unsqueeze(1)).squeeze(1)        
        
        # flows for each level of direct connection feature
        flows18 = flow.unsqueeze(1).expand((outf.shape[0], 1, outf.shape[2], outf.shape[3], outf.shape[4])) * 1/8
        flows14 = flow.unsqueeze(1).expand((skip_out128.shape[0], 1, skip_out128.shape[2], skip_out128.shape[3], skip_out128.shape[4])) * 1/4
        flows12 = flow.unsqueeze(1).expand((skip_out64.shape[0], 1, skip_out64.shape[2], skip_out64.shape[3], skip_out64.shape[4])) * 1/2

        # layer segmentation network
        outfs = self.transformer8(outf, flows18)
        skip_out128s = self.transformer4(skip_out128, flows14)
        out_up_128 = self.up_tr256(outfs,skip_out128s)
        skip_out64s = self.transformer2(skip_out64, flows12)
        out_up_64 = self.up_tr128(out_up_128, skip_out64s)
        skip_out32s = self.transformer(skip_out32, flows.unsqueeze(1))
        out_up_32 = self.up_tr64(out_up_64, skip_out32s)
        
        # out
        s = self.out_tr(out_up_32)
        layer_out = self.out_tr2(out_up_32)
        layerProb = logits2Prob(layer_out, 1)
        out = logits2Prob(s, 2)

        mu, sigma2 = computeMuVariance(out, layerMu=None, layerConf=None)
        S = mu.clone()
        
        # relu for Topology Guarantee
        for i in range(1,3):
            S[:,i,:, :] = torch.where(S[:, i,:, :]< S[:,i-1,:,:], S[:,i-1,:, :], S[:,i,:,:])

        return S, out, layerProb, mu, x_out, flow
        
class UNet2d_3d(nn.Module):
    # down_sample is 2d and up_sample is 3d, no align branch

    def __init__(self, n_layer=3,act='relu'):
        super(UNet2d_3d, self).__init__()
        self.n_layer = n_layer
        
        self.down_tr32 = DownTransition(1,32,act)
        self.down_tr64 = DownTransition(32,64,act)
        self.down_tr128 = DownTransition(64,128,act)
        self.down_trf = DownTransition(128,128,act, pol = False)

        self.up_tr256 = UpTransition(128, 128, 2,act)
        self.up_tr128 = UpTransition(128, 64, 1,act)
        self.up_tr64 = UpTransition(64,32,0,act)
        
        self.up1_tr256 = UpTransition(128, 128, 2,act)
        self.up1_tr128 = UpTransition(128, 64, 1,act)
        self.up1_tr64 = UpTransition(64,32,0,act)
        
        self.out_tr = OutputTransition(32, n_layer)
        self.out_tr2 = OutputTransition(32, n_layer+1)
        self.out_tr3 = OutputTransition(32, 1)
        
        self.out_pool = nn.AdaptiveAvgPool2d(1)
        self.transformer = SpatialTransformer2((320, 48, 40)) 
        self.transformer2 = SpatialTransformer2((160, 24, 40))
        self.transformer4 = SpatialTransformer2((80, 12, 40))
        self.transformer8 = SpatialTransformer2((40, 6, 40))

    def forward(self, xr):  
        bs = xr.shape[0]
        x = xr.permute(0,4,1,2,3)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
        out32, skip_out32 = self.down_tr32(x)
        out64,skip_out64 = self.down_tr64(out32)
        out128,skip_out128 = self.down_tr128(out64)
        outf,skip_outf = self.down_trf(out128)
    
        skip_out128 = torch.reshape(skip_out128, (bs, skip_out128.shape[0] // bs, skip_out128.shape[1], skip_out128.shape[2], skip_out128.shape[3])).permute(0,2,3,4,1)
        outf = torch.reshape(outf, (bs, outf.shape[0] // bs, outf.shape[1], outf.shape[2], outf.shape[3])).permute(0,2,3,4,1)
        skip_out64 = torch.reshape(skip_out64, (bs, skip_out64.shape[0] // bs, skip_out64.shape[1], skip_out64.shape[2], skip_out64.shape[3])).permute(0,2,3,4,1)
        skip_out32 = torch.reshape(skip_out32, (bs, skip_out32.shape[0] // bs, skip_out32.shape[1], skip_out32.shape[2], skip_out32.shape[3])).permute(0,2,3,4,1)
              
        out_up_128 = self.up_tr256(outf,skip_out128)
        out_up_64 = self.up_tr128(out_up_128, skip_out64)
        out_up_32 = self.up_tr64(out_up_64, skip_out32)
        
        s = self.out_tr(out_up_32)
        layer_out = self.out_tr2(out_up_32)
        layerProb = logits2Prob(layer_out, 1)
        out = logits2Prob(s, 2)

        mu, sigma2 = computeMuVariance(out, layerMu=None, layerConf=None)
        S = mu.clone()
        
        
        for i in range(1,3):
            S[:,i,:, :] = torch.where(S[:, i,:, :]< S[:,i-1,:,:], S[:,i-1,:, :], S[:,i,:,:])

        return S, out, layerProb, mu
        
        
        



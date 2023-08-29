import os
import argparse
import torch
import cv2
import sys 
import csv
import numpy as np
from numpy import *
import torch.nn as nn
from torch.utils.data.sampler import  WeightedRandomSampler
from model2d_3d import *
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import scipy.ndimage
import scipy.io as io

from dataset.dataset import *
from loss.losses import *

# for visualization
from torch.utils.tensorboard import SummaryWriter  

# Set arguments and hyper-parameters.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--data_dir', default="/data3/lh/boe/ori/", type=str, help='Directory to load PCam data.')
parser.add_argument('--label_dir', default='./label', type=str, help='Directory to load label files.')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, help='Directory to save checkpoint.')

parser.add_argument('--gpu', default='0', type=str, help='GPU Devices to use.')
parser.add_argument('--ck_name', default="2d3d_5_inter20_123_noncc_047_fold0_1.008%.t7", type=str, help='GPU Devices to use.')
parser.add_argument('--batch_size', default=7, type=int, help='Batch size.')

parser.add_argument('--lr', default=3e-4, type=float, help='Starting learning rate.')
parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay.')
parser.add_argument('--lr_decay_step', default=1, type=int, help='Learning rate decay step.')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 penalty for regularization.')
parser.add_argument('--weight_g', default=0.3, type=float, help='L2 penalty for regularization.')


parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch.')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint or not.')
parser.add_argument('--store_last', action='store_true', help='store the last model.')
parser.add_argument('--resume_last', action='store_true', help='resume the last model.')
parser.add_argument('--name', default='layers', type=str, help='The id of this train')
parser.add_argument('--valid_fold', default=0, type=int, help='Starting epoch.')

parser.add_argument('--norm', type=str, default='inorm', dest='norm')
parser.add_argument('--ny_in', type=int, default=320, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=400, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=40, dest='nch_in')

parser.add_argument('--ny_out', type=int, default=320, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=400, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=40, dest='nch_out')
parser.add_argument('--patch_size', type=int, default=48, dest='patch_size')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')

args = parser.parse_args()

class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', axis = 1):
        super().__init__()

        self.mode = mode
        self.axis = axis
        #print(size)
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        #print(vectors)
        grids = torch.meshgrid(vectors)
        #print(grids)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        #print(grid.shape)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, fmode = 'bilinear'):
        # We only focus on shift in one dimension, so the other two shift is zero.
        oshift1 = torch.zeros_like(flow)
        oshift2 = torch.zeros_like(flow)
        if self.axis == 1:
            flow = torch.cat([flow[:,:1,:,:,:], oshift1, oshift2],1)
        else:
            flow = torch.cat([flow[:,:1,:,:,:], flow[:,1:,:,:,:], oshift2[:,:1,:,:,:]],1)

        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        # print(new_locs.shape, self.grid.shape, flow.shape)
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

        return F.grid_sample(src, new_locs, align_corners=True, mode=fmode, padding_mode='border')

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ck_name = args.ck_name
# load checkpoint
checkpoint = torch.load("./checkpoint/"+ ck_name)

model = checkpoint["model"].module

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
model = model.to(device)
print(dir(model))
if len(args.gpu) > 1:
    model = nn.DataParallel(model)
    
# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = MSE().loss
elif args.image_loss == 'msce':
    image_loss_func = MultiSurfaceCrossEntropyLoss()
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

grad_loss = Grad_2d(weight = torch.tensor(np.array([1.0, 0.5, 1.5]).astype(np.float32)))
Gloss = GeneralizedDiceLoss()
smoothl1loss = nn.SmoothL1Loss()
mlloss = MultiLayerCrossEntropyLoss()
bce_Bscan = MSE_bscan()

# set best loss
best_loss = 10000
ncc = NCC_oct_metric().loss
transformer = SpatialTransformer2((224, 48, 41))
patch_size = args.patch_size

def load_data(data_dir, label_dir):
    valid_set = sd_oct_flatten_ori_align_doe(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, step = 10, asize = 224, patch_size = patch_size)
    
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return valid_loader

def get_new_label(mask, flow):
    
    new_label = (mask - flow[:,0].unsqueeze(1).unsqueeze(1))
    new_label[new_label > 223] = 223
    new_label[new_label < 0] = 0
    
    new_label = transformer(new_label.unsqueeze(1), flow[:, 1:, :].unsqueeze(1).unsqueeze(1).expand(new_label.shape[0], 1, new_label.shape[1], new_label.shape[2], new_label.shape[3])).squeeze()
    return new_label

def valid(epoch, valid_loader):
    print("valid:")
    global best_loss

    model.eval()
    surface_ces = []
    loss_sl1s = []
    loss_grads = []
    loss_gds = []
    loss_nccs = []

    flows = []
    result = []
    logits = []
    target = []
    metric = [] # mean_dis
    
    store_path = ""
    if store_path != "" and not os.path.exists(store_path):
        os.mkdir(store_path)
    
    names = []
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name, _, _) in enumerate(valid_loader):
            data = data.to(device)
            mask = mask.float()
            layer_gt = layer_gt.float().to(device)
            s, logit, layerProb, mu, x, flow = model(data)
            
            logit = transformer(logit.detach().cpu(), -1 * flow.cpu().detach().unsqueeze(1).expand((logit.shape[0], 1, logit.shape[2], logit.shape[3], logit.shape[4])))
            
            result.append(s.cpu()[:,:,:,:] + flow.detach().cpu()[:,:,:,:])
            logits.append(logit)
            target.append(mask[:,:,:,:])
            flows.append(flow.detach().cpu()[:,:,:,:])
            names.extend(name) 
            continue
            
    result = torch.cat(result).numpy()
    target = torch.cat(target).numpy()
    flows = torch.cat(flows).numpy()
    logits = torch.cat(logits).numpy()
    cat_metric(names, result, target, flows, logits)
    
 
def cat_metric(name, result, target, flows, logits):
    print(flows.shape)
    name_list = list(set([l[:-6] for l in name]))
    print(len(name_list), name_list[:5])
    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
            
    deno = np.zeros((len(name_list), nc, 800, 41))  # denominator
    mulo = np.zeros((len(name_list), nc, 800, 41)) # molecule
    
    denot = np.zeros((len(name_list), nc, 800, 11))   # denominator
    mulot = np.zeros((len(name_list), nc, 800, 11))   # molecule
    
    ft = np.zeros((len(name_list), 1, 1, 41))   # denominator
    mft = np.zeros((len(name_list),1,1,1))   # molecule
    
    logits_f = np.zeros((len(name_list), 8, 224, 800, 41))
    logitsmulot = np.zeros((len(name_list), 8, 224, 800, 41))   # molecule
    
    for idx, l in enumerate(name):
        cname = l[:-6]
        widx = int(l[-3:])
        nameidx = name_list.index(cname)
        deno[nameidx, :, widx: widx + patch_size, :] = deno[nameidx, :, widx: widx + patch_size, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + patch_size, :] = denot[nameidx, :, widx: widx + patch_size, :] + target[idx,:,:,:]
        
        mulo[nameidx, :, widx: widx + patch_size, :] += 1
        mulot[nameidx, :, widx: widx + patch_size, :] += 1
        
        logits_f[nameidx, :, :, widx: widx + patch_size, :] = logits_f[nameidx, :, :, widx: widx + patch_size, :] + logits[idx]
        logitsmulot[nameidx, :, :, widx: widx + patch_size, :] += 1
        
        ft[nameidx, :, :, :] += flows[idx, :,:,:]
        mft[nameidx, :,:,:] += 1
    
    deno = deno / mulo
    denot = denot / mulot
    ft = ft / mft
    logits_f = logits_f / logitsmulot
    
    file = open(ck_name[:-4] + ".csv", "w")
    file_c = csv.writer(file)
    save_path = "./result/" + ck_name[:-4] + "new_doe"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, l in enumerate(name_list):        
        cpath = os.path.join(save_path, l)
        io.savemat(cpath, {"prediction":deno[idx], "gt":denot[idx], "flow": ft[idx]})#, "logits": logits_f[idx]})
        

if __name__ == '__main__':
    valid_loader= load_data(args.data_dir, args.label_dir)
    # valid and store results
    valid(0, valid_loader)
   
    # get the metric
    prediction_path = "./checkpoint/" + ck_name[:-4] + "new_doe"
    target_path = args.data_dir
    
    prediction_files = glob.glob(os.path.join(prediction_path, "*.mat"))
    results = []
    
    for iidx, l in enumerate(sorted(prediction_files)):
        print(l)
        cname = l.split("/")[-1]
        
        tpath = os.path.join(target_path, cname)
        
        c_data = scio.loadmat(l)
        c_pre = c_data["prediction"]
        c_flow = c_data["flow"]
        
        t_data = scio.loadmat(tpath)
        cur_seg = t_data["manualLayers1"]
        fluid = t_data["automaticFluidDME"]
        auto_layer = t_data["automaticLayersDME"]
       
        nansBscan = np.isnan(cur_seg).astype(int)  # size: 8x512x61
        nansBscan = np.sum(nansBscan, axis=(0,1))  # size: 61
        
        cidx = []
        for ii,l1 in enumerate(nansBscan):
            if l1 < 8*512:
                cidx.append(ii)
        
        cur_seg = cur_seg.transpose(1,2,0)
        auto_layer = auto_layer.transpose(1,2,0)
        z_indexes, y_indexes, x_indexes = np.nonzero(~np.isnan(cur_seg))

        zmin = 128
        zmax = 640
        cur_seg = cur_seg[zmin:zmax, cidx, :]
        auto_layer = auto_layer[zmin:zmax, cidx, :]
        
        bv = int(np.nanmin(cur_seg) - 10)
        
        fluid = fluid[bv:bv+224, zmin:zmax, cidx]
        fluid = fluid.sum(0)
        mask = fluid==0
        mask = np.array([mask])
        mask = mask.transpose(1,2,0)
        mask = np.concatenate([mask for l in range(8)], 2)
        
        cur_seg -= bv
        auto_layer -= bv
        
        temp = cidx[0]
        cidx = [l-temp for l in cidx]
        c_pre = c_pre[:,:,cidx]
        c_pres = c_pre.transpose(1,2,0)[:cur_seg.shape[0]]
        
        nan_mask = np.logical_or(np.isnan(cur_seg), np.isnan(auto_layer))
        
        inter = c_pres[:,:,7] - c_pres[:,:,6]
        c_pres[:,:,6][inter>12] = (c_pres[:,:,7]-6)[inter>12]
        inter = c_pres[:,:,6] - c_pres[:,:,5]
        c_pres[:,:,5][inter>12] = (c_pres[:,:,6]-6)[inter>12]
            
        for layer in range(nan_mask.shape[2]):
            c_pre = c_pres[:,:,layer]
            c_gt = cur_seg[:,:,layer]
            cresult = np.abs(np.round(c_pre) - c_gt)

            cresult = np.nanmean(cresult[~nan_mask[:,:,layer]])
            results.append(cresult)
        continue
        
    results = np.array(results).reshape(5,8)
    print(results.mean(0) * 3.87, results.std(0) * 3.87)
    print(results.mean() * 3.87, results.mean(1).std() * 3.87)

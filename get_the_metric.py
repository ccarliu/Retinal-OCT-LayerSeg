import os
import argparse
import torch
import cv2
import sys 
import csv
import numpy as np
from numpy import *
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import  WeightedRandomSampler
from model2d_3d import *
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import scipy.ndimage
import scipy.io as io

from dataset import *
from losses import *

# for visualization
from torch.utils.tensorboard import SummaryWriter  

# Set arguments and hyper-parameters.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--data_dir', default='/data3/layer_segmentation/a2a_oct', type=str, help='Directory to load PCam data.')
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

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')

args = parser.parse_args()


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
model = model.to(device)

if len(args.gpu) > 1:
    model = nn.DataParallel(model)
    


def load_data(data_dir, label_dir):
    valid_set = sd_oct_flattenp_align(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, step = 20)
    
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return valid_loader


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
    target = []
    metric = [] # mean_dis
    
    store_path = ""
    if store_path != "" and not os.path.exists(store_path):
        os.mkdir(store_path)
    
    names = []
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name) in enumerate(valid_loader):
            data = data.to(device)
            mask = mask.float()
            layer_gt = layer_gt.float().to(device)
            s, logits, layerProb, mu, x, flow = model(data)
            
            
            #print(flow[:, 1].abs().mean(), mean_dis)
            result.append(s.cpu()[:,:,:,:] + flow.detach().cpu()[:,:,:,:])
            target.append(mask[:,:,:,:])
            flows.append(flow.detach().cpu()[:,:,:,:])
            names.extend(name) 
            
            
    print()
    result = torch.cat(result).numpy()
    target = torch.cat(target).numpy()
    flows = torch.cat(flows).numpy()
    cat_metric(names, result, target, flows)
    print(mean(metric), mean(loss_nccs), mean(surface_ces), mean(loss_sl1s), np.mean(loss_grads, 0), mean(loss_gds))
    
 
def cat_metric(name, result, target, flows):
    
    # to concat the patches to original volume to calculate the final metric, each A-scan will have more than one segmentation, depend on the step of patch(dataset)
    
    name_list = list(set([l[:-6] for l in name]))

    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
            
    deno = np.zeros((len(name_list), nc, 400, 40))  # denominator
    mulo = np.zeros((len(name_list), nc, 400, 40))  # molecule
    
    denot = np.zeros((len(name_list), nc, 400, 40))   # denominator
    mulot = np.zeros((len(name_list), nc, 400, 40))   # molecule
    
    ft = np.zeros((len(name_list), 1, 1, 40))   # denominator
    mft = np.zeros((len(name_list),1,1,1))   # molecule
    
    for idx, l in enumerate(name):

        cname = l[:-6]
        widx = int(l[-3:])

        nameidx = name_list.index(cname)

        deno[nameidx, :, widx: widx + 48, :] = deno[nameidx, :, widx: widx + 48, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + 48, :] = denot[nameidx, :, widx: widx + 48, :] + target[idx,:,:,:]
        
        mulo[nameidx, :, widx: widx + 48, :] += 1
        mulot[nameidx, :, widx: widx + 48, :] += 1
        
        ft[nameidx, :, :, :] += flows[idx, :,:,:]
        mft[nameidx, :,:,:] += 1
    
    deno = deno / mulo
    
    # this target is the same as the target in origin volume.
    denot = denot / mulot
    
    ft = ft / mft
    
    rmean = np.abs(np.round(deno)-np.round(denot))
    print(rmean.mean() * 3.24)
    print(np.mean(rmean, (0,2,3)) * 3.24)
    AMD_mean = rmean[AMD_list]
    NOR_mean = rmean[NOR_list]
    print("AMD:")
    print(np.mean(AMD_mean, (0,2,3)) * 3.24)
    
    print("NOR:")
    print(np.mean(NOR_mean, (0,2,3)) * 3.24)
    
    means = np.mean(rmean, (2,3))

    


 
if __name__ == '__main__':
    valid_loader= load_data(args.data_dir, args.label_dir)
    valid(0, valid_loader)

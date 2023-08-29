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
import time

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
parser.add_argument('--batch_size', default=6, type=int, help='Batch size.')

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
parser.add_argument('--image-loss', default='msce', help='image reconstruction loss - can be mse or ncc (default: mse)')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load checkpoint
args.ck_name = "test_2_089_fold0_0.914%.t7"
ck_path = "./checkpoint/"
checkpoint = torch.load("/home/lh/layer_segmentation_97/checkpoint_a2a/" + args.ck_name)

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
# set best loss
best_loss = 10000
ncc = NCC_oct_metric().loss
bce_Bscan = MSE_bscan()
meandis = nn.L1Loss()
# set visualization writer
logpt = "./log/log10"
if not os.path.exists(logpt):
    os.mkdir(logpt)
    
def load_data(data_dir, label_dir):
    #train_set = sd_oct_flattenp(data_dir, label_dir, usage = 'train', valid_fold = args.valid_fold)
    valid_set = sd_oct_flattenp_align(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, step = 10)
    
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True, pin_memory=True, num_workers=4)
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
    
    store_path = os.path.join("/data3/layer_segmentation/", args.ck_name[:-3])
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    
    names = []
    
    start = time.clock()
    
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name) in enumerate(valid_loader):
            data = data[:,:,:,:,:].to(device)
            mask = mask[:,:,:,:].float()
            layer_gt = layer_gt.float().to(device)
            s, logits, layerProb, mu, x, flow = model(data)
            #print(flow[:, 1])
            #continue
            result.append(s.cpu()[:,:,:,:] + flow.detach().cpu()[:,:,:,:])
            target.append(mask[:,:,:,:])
            flows.append(flow.detach().cpu()[:,:,:,:])
            names.extend(name) 
            continue
            
    print()
    
    end = time.clock()
    print("time: ", start - end)
    
    result = torch.cat(result).numpy()
    target = torch.cat(target).numpy()
    flows = torch.cat(flows).numpy()
    cat_metric(names, result, target, flows)
    
 
def cat_metric(name, result, target, flows):
    print(flows.shape)
    name_list = list(set([l[:-6] for l in name]))
    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
            
    deno = np.zeros((len(name_list), nc, 400, 41))  # denominator
    mulo = np.zeros((len(name_list), nc, 400, 41))  # molecule
    
    denot = np.zeros((len(name_list), nc, 400, 41))   # denominator
    mulot = np.zeros((len(name_list), nc, 400, 41))   # molecule
    
    ft = np.zeros((len(name_list), 1, 1, 41))   # denominator
    mft = np.zeros((len(name_list),1,1,1))   # molecule
    
    for idx, l in enumerate(name):
        #print(result[idx], target[idx])
        cname = l[:-6]
        widx = int(l[-3:])
        #print(widx)
        nameidx = name_list.index(cname)
        #print(nameidx)
        deno[nameidx, :, widx: widx + 48, :] = deno[nameidx, :, widx: widx + 48, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + 48, :] = denot[nameidx, :, widx: widx + 48, :] + target[idx,:,:,:]
        
        mulo[nameidx, :, widx: widx + 48, :] += 1
        mulot[nameidx, :, widx: widx + 48, :] += 1
        
        ft[nameidx, :, :, :] += flows[idx, :,:,:]
        mft[nameidx, :,:,:] += 1
    
    deno = deno / mulo
    denot = denot / mulot
    ft = ft / mft
    
    rmean = np.abs((deno)-(denot))
    print(rmean.mean() * 3.24)
    print(np.mean(rmean, (0,2,3)) * 3.24)
    AMD_mean = rmean[AMD_list]
    NOR_mean = rmean[NOR_list]
    print("AMD:")
    print(np.mean(AMD_mean, (0,2,3)) * 3.24)
    
    print("NOR:")
    print(np.mean(NOR_mean, (0,2,3)) * 3.24)
    
    means = np.mean(rmean, (2,3))
    #for idx, l in enumerate(name_list):
     #   print(l, means[idx])
    '''
    
    file = open(args.ck_name[:-3] + ".csv", "w")
    file_c = csv.writer(file)
    save_path = os.path.join("/data3/layer_segmentation/", args.ck_name[:-3] + "_result")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, l in enumerate(name_list):
        types = "control"
        if "AMD" in l:
            types = "AMD"
        
        file_c.writerow([l, types, *means[idx]])
        cpath = os.path.join(save_path, l)
        io.savemat(cpath, {"prediction":deno[idx], "gt":denot[idx], "flow": ft[idx]})
    '''
        

 
if __name__ == '__main__':
    valid_loader= load_data(args.data_dir, args.label_dir)
    valid(0, valid_loader)


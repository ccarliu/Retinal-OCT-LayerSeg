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
import scipy.io as io


from dataset.dataset import *
from loss.losses import *

from torch.utils.tensorboard import SummaryWriter   

# Set arguments and hyper-parameters.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--data_dir', default='/data3/layer_segmentation/a2a_oct', type=str, help='Directory to load PCam data.')
parser.add_argument('--label_dir', default='./label', type=str, help='Directory to load label files.')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, help='Directory to save checkpoint.')

parser.add_argument('--gpu', default='0', type=str, help='GPU Devices to use.')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size.')

parser.add_argument('--lr', default=3e-4, type=float, help='Starting learning rate.')
parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay.')
parser.add_argument('--lr_decay_step', default=1, type=int, help='Learning rate decay step.')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 penalty for regularization.')
parser.add_argument('--weight_g', default=0.3, type=float, help='L2 penalty for regularization.')
parser.add_argument('--patch_length', default=48, type=int, help='Seed.')

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

parser.add_argument('--ck_name', default="2d3d_5_inter20_123_noncc_047_fold0_1.008%.t7", type=str, help='GPU Devices to use.')


args = parser.parse_args()

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args.ck_name = "jhh.t7"

# load checkpoint
checkpoint = torch.load(os.path.join("./checkpoint/", args.ck_name))
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

    
def load_data(data_dir, label_dir):
    valid_set = hc_oct_noalign(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, inter = 10, morechannel = False, patch_length = args.patch_length)
    
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
    result = []
    flows = []
    target = []
    metric = [] # mean_dis
    store_path = args.ck_name[:-3] + "_align"
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    names = []
    momun = torch.zeros([49])
    idx = 0
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name) in enumerate(valid_loader):
            data = data[:,:,:,:,:].to(device)
            mask = mask[:,:,:,:].float()
            
            data = data.to(device)
            mask = mask.float()
            layer_gt = layer_gt.float().to(device)
            data[torch.isnan(data)] = 0
            s, logits, layerProb, mu,x, flow = model(data)
            idx += 1

            if torch.isnan(flow).sum() > 0:
                
                print("nan__", torch.isnan(flow).sum(), torch.isnan(data).sum(), name)
                
            
            result.append(s.cpu()[:,:,:,:]+flow.cpu()[:,:1,:,:])
            target.append(mask[:,:,:,:])
            flows.append(flow.cpu()[:,:1,:,:])
            names.extend(name) 
            '''
            if True:
                s1 = x.detach().cpu().numpy()
                out1 = flow.detach().cpu().numpy()
                cpath = os.path.join(store_path, name[0])
                io.savemat(cpath, {"img":s1.squeeze(), "flow" : out1, "imgr": data.cpu().numpy(), "mask": mask.numpy()})
            '''
                        
            continue
            
            
    result = torch.cat(result).numpy()
    target = torch.cat(target).numpy()
    flows = torch.cat(flows).numpy()
    cat_metrica(names, result, target, flows)
    
       
def cat_metrica(name, result, target, flows):
    rootpath = "/data3/layer_segmentation/afterfp2/"
    name_list = list(set([l[:-6] for l in name]))
    print(len(name_list), name_list[:5])

    n, nc, w, h = result.shape

    deno = np.zeros((len(name_list), nc, 1024, 49))  # denominator
    mulo = np.zeros((len(name_list), nc, 1024, 49))  # molecule
    
    denot = np.zeros((len(name_list), nc, 1024, 49))   # denominator
    mulot = np.zeros((len(name_list), nc, 1024, 49))   # molecule

    ft = np.zeros((len(name_list), 1, 1, 49))
    mft = np.zeros((len(name_list),1,1,1)) 
    
    for idx, l in enumerate(name):
        cname = l[:-6]
        widx = int(l[-3:])
        nameidx = name_list.index(cname)
        deno[nameidx, :, widx: widx + args.patch_length, :] = deno[nameidx, :, widx: widx + args.patch_length, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + args.patch_length, :] = denot[nameidx, :, widx: widx + args.patch_length, :] + target[idx,:,:,:]
        ft[nameidx, :, :, :] += flows[idx, 0,:,:]
        mulo[nameidx, :, widx: widx + args.patch_length, :] += 1
        mulot[nameidx, :, widx: widx + args.patch_length, :] += 1
        mft[nameidx, :,:,:] += 1
        
    #print(mulot[0,2,:,1])
    #print(deno[0,2,30:100,1])
    #print(deno[0,2,905:1000,1])
    #deno = deno[:,:,3:1021,:]
    #denot = denot[:,:,3:1021,:]
    #mulo = mulo[:,:,3:1021,:]
    #mulot = mulot[:,:,3:1021,:]
    
    ft = ft / mft

    deno = deno / mulo
    denot = denot / mulot
    print(np.abs(deno - denot).mean())
    means = np.mean(np.abs(np.round(deno)-np.round(denot)), (2,3))
    for idx, l in enumerate(name_list):
        print(l, means[idx])
    rmean = np.abs((deno)-(denot))
    print(rmean.mean() * 3.9, (rmean*3.9).std())
    
    print(np.mean(rmean, (0,2,3)) * 3.9, np.std((rmean*3.9), (0,2,3)))

    


 
if __name__ == '__main__':
    valid_loader= load_data(args.data_dir, args.label_dir)
    valid(0, valid_loader)

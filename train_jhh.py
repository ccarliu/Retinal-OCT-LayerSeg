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
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from dataset import *
from loss.losses import *

import logging


# Set arguments and hyper-parameters.
parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--data_dir', default='/data3/layer_segmentation/a2a_oct', type=str, help='Directory to load PCam data.')
parser.add_argument('--label_dir', default='./label', type=str, help='Directory to load label files.')
parser.add_argument('--ckpt_dir', default='./nni_best', type=str, help='Directory to save checkpoint.')

parser.add_argument('--gpu', default='0', type=str, help='GPU Devices to use.')
parser.add_argument('--batch_size', default=6, type=int, help='Batch size.')
parser.add_argument('--label_inter', default=1, type=int, help='Sparse label inter.')
parser.add_argument('--patch_length', default=48, type=int, help='Seed.')
parser.add_argument('--patch_inter', default=40, type=int, help='Seed.')
parser.add_argument('--seed', default=42, type=int, help='Seed.')

parser.add_argument('--lr', default=0.003, type=float, help='Starting learning rate.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='L2 penalty for regularization.')
parser.add_argument('--weight_g', default=2, type=float, help='Loss weight of Smooth loss.')
parser.add_argument('--ncc_weight', default=1, type=float, help='Loss weight of normlized cross correlation.')

parser.add_argument('--optimizer', default="adam", type=str, help='The type of optimizer')
parser.add_argument('--scheduler_type', default='cosine', type=str, help='GPU Devices to use.')
parser.add_argument('--T_max', default=40, type=float, help='Period of lr scheduler.')
parser.add_argument('--step', default=10, type=float, help='Period of lr scheduler.')
parser.add_argument('--gamma', default=0.9, type=float, help='Starting learning rate.')

parser.add_argument('--eta_min', default=3e-7, type=float, help='Min lr of lr scheduler.')
parser.add_argument('--random_patch_i', default='n', type=str, help='If use random patch, index, just for nni')
parser.add_argument('--weight_t', default='o2', type=str, help='Smooth label weight type')
parser.add_argument('--morec_i', default='n', type=str, help='If use more channel for input, index, just for nni')



parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch.')
parser.add_argument('--epochs', default=120, type=int, help='Number of training epochs.')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint or not.')
parser.add_argument('--random_patch', action='store_true', help='If use random patch.')
parser.add_argument('--morec', action='store_true', help='If use more channel for input.')

parser.add_argument('--store_last', action='store_true', help='store the last model.')
parser.add_argument('--resume_last', action='store_true', help='resume the last model.')
parser.add_argument('--name', default='layers', type=str, help='The id of this train')
parser.add_argument('--valid_fold', default=0, type=int, help='valid fold.')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

args = parser.parse_args()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

if args.random_patch_i == "t":
    args.random_patch = True
if args.morec_i == "t":
    args.morec = True

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)


# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.morec:
    input_c = 3
else:
    input_c = 1
    
if args.label_inter>1:
    model = UNet3D_dual_up_hc(n_layer = 9, input_c = input_c, input_size=[128,args.patch_length,49])
else:   
    model = UNet3D_dual_up_hc(n_layer = 9, input_size=[128,args.patch_length,49])

# set optimizer
if args.optimizer == "adam":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay = args.weight_decay)

if args.scheduler_type == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
elif args.scheduler_type == "reducelronpla" :
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

model = model.to(device)
model = nn.DataParallel(model)


image_loss_func = MultiSurfaceCrossEntropyLoss()


args.weight_o2 = [0.36, 0.33, 0.42, 0.40, 0.54, 0.78, 1.04, 0.70, 0.98]
args.weight_o3 = [0.58, 0.49, 1.1, 0.87, 0.84, 2.74, 4, 2.3, 3.8]

grad_loss = Grad_2d(weight = torch.tensor(np.array(args.weight_o3).astype(np.float32)).to(device))
bce_Bscan = MSE_bscan()
Gloss = GeneralizedDiceLoss()
smoothl1loss = nn.SmoothL1Loss()
mlloss = MultiLayerCrossEntropyLoss()
ncc = NCC_oct().loss
meandis = nn.L1Loss()


# set best loss
best_loss = 10000

# set visualization writer
logpt = "./log/" + args.name
if not os.path.exists(logpt):
    os.mkdir(logpt)

def load_data(data_dir, label_dir):
    train_set = hc_oct_noalign(data_dir, label_dir, usage = 'train', valid_fold = args.valid_fold, random_patch = args.random_patch, inter = args.patch_inter, morechannel = args.morec, patch_length = args.patch_length)
    valid_set = hc_oct_noalign(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, random_patch = args.random_patch, inter = args.patch_inter, morechannel = args.morec, patch_length = args.patch_length)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, valid_loader

def train(epoch, train_loader):
    print("train:")
    model.train()
    surface_ces = []
    loss_sl1s = []
    loss_grads = []
    loss_gds = []
    metric = [] # mean_dis
    
    loss_nccs = []
    loss_mses = []
    
    for idx, (data, mask, layer_gt, _) in enumerate(train_loader):
        data = data.to(device)
        mask = mask.float().to(device)

        optimizer.zero_grad()
        
        s, logits, layerProb, mu, x, flow = model(data)

        new_label = (mask - flow.detach())
        new_label[new_label > 127] = 127
        new_label[new_label < 0] = 0
        layer_gt = []
        for l in new_label:
            layer_gt.append(getLayerLabels(l, 128).unsqueeze(0))
        layer_gt = torch.cat(layer_gt, dim = 0)

        lossgd = Gloss(layerProb[:,:,:,:,::args.label_inter], layer_gt[:,:,:,::args.label_inter].float())  # diceloss
        lossgd += mlloss(layerProb[:,:,:,:,::args.label_inter], layer_gt[:,:,:,::args.label_inter].float())   # layer_cross_entropy
        surface_ce = image_loss_func(logits[:,:,:,:,::args.label_inter], new_label[:,:,:,::args.label_inter])  # surface_cross_entropy
        loss_grad = grad_loss.loss(mask, mu) # grad_loss
        # if not using full label, use pseudo label for alignment.
        if args.label_inter == 1:
            lossmse = bce_Bscan.loss(mask[:,8] - flow)
        else:
            lossmse = bce_Bscan.loss(mu.detach()[:,8] + flow.detach() - flow)
            
        if epoch<5 and args.label_inter > 1:
            weight_mse = 0
        else:
            weight_mse = 1
                
        reg_loss = torch.mean(torch.pow(torch.abs(flow),2))

        ncc_loss = ncc(x)
        sl1loss = smoothl1loss(mu[:,:,:,::args.label_inter], new_label[:,:,:,::args.label_inter])   # smooth_l1_loss # why use smooth l1
        main_loss =  surface_ce + sl1loss  + lossmse * weight_mse + loss_grad * args.weight_g + lossgd + ncc_loss * args.ncc_weight# + reg_loss * 0.001
        main_loss.backward()
        
        '''
        for name, parms in model.named_parameters():	
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
             ' -->grad_value:',parms.grad)
             '''
        optimizer.step()
        metric.append(meandis(s, mask-flow.detach()).detach().cpu().numpy())
        loss_mses.append(lossmse.detach().cpu().numpy())
        loss_sl1s.append(sl1loss.detach().cpu().numpy())
        loss_grads.append(loss_grad.detach().cpu().numpy())
        surface_ces.append(surface_ce.detach().cpu().numpy())
        loss_nccs.append(ncc_loss.detach().cpu().numpy())
        smooth_loss = sum(loss_sl1s[-100:]) / min(len(loss_sl1s), 100)
        
        if idx % 8 == 0:
            print(idx, sl1loss.item(), metric[-1], mean(surface_ces), loss_grad.item(), surface_ce.item(), ncc_loss.item(), lossmse.item(), end = "\r")
            
    loss = mean(loss_sl1s)
    print(mean(metric), mean(surface_ces), mean(loss_sl1s), mean(loss_grads))




def valid(epoch, valid_loader):
    print("valid:")
    global best_loss

    model.eval()
    surface_ces = []
    loss_sl1s = []
    loss_grads = []
    loss_gds = []
    loss_nccs = []
    loss_mses = []
    metric = [] # mean_dis
    
    names = []
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name) in enumerate(valid_loader):
        
            data = data.to(device)
            mask = mask.float()
                    
            s, logits, layerProb, mu, x, flow = model(data)
            flow = flow.detach().cpu()
            #print(flow)
            new_label = (mask - flow.detach())
            new_label[new_label > 127] = 127
            new_label[new_label < 0] = 0
            layer_gt = []
            for l in new_label:
                layer_gt.append(getLayerLabels(l, 128).unsqueeze(0))
            layer_gt = torch.cat(layer_gt, dim = 0)    
            lossgd = Gloss(layerProb.detach().cpu(), layer_gt.float())  # diceloss
            lossgd += mlloss(layerProb.detach().cpu(), layer_gt.float())   # layer_cross_entropy
            surface_ce = image_loss_func(logits.detach().cpu(), new_label)  # surface_cross_entropy
            loss_grad = grad_loss.loss(mask, mu[:,:,:,:].detach()) # grad_loss
            lossmse = bce_Bscan.loss(mask - flow)
            sl1loss = smoothl1loss(mu[:,:,:,:].detach().cpu(), new_label)   # smooth_l1_loss # why use smooth l1

            ncc_loss = ncc(x)

            mean_dis = np.abs(s.detach().cpu().numpy()[:,:,:,:] - mask.numpy()[:,:,:,:] + flow.numpy()).mean()
            metric.append(mean_dis)            
            loss_sl1s.append(sl1loss.detach().numpy())
            loss_grads.append(loss_grad.detach().cpu().numpy())
            loss_nccs.append(ncc_loss.detach().cpu().numpy())
            loss_mses.append(lossmse.detach().cpu().numpy())
            surface_ces.append(surface_ce.detach().cpu().numpy())
            names.append(name)
            
            smooth_loss = sum(metric[-100:]) / min(len(metric), 100)
            if idx % 10 == 0:
                print(idx, smooth_loss, ncc_loss, end = "\r")
    
    #print(metric)    
    print(mean(metric), mean(surface_ces), mean(loss_sl1s), mean(loss_grads), mean(loss_nccs), mean(loss_mses))

    loss = mean(metric)
    if loss < best_loss:
        best_loss = loss
        save_checkpoint(epoch)
    return loss

def save_checkpoint(epoch, name = None):
    ''' Save checkpoint if accuracy is higher than before.
    
    # Arguments
        epoch (int): Current epoch.
    '''
    # Save model and global variables into checkpoint.
    global best_loss
    print('==> Saving checkpoint...')
    state = {
        'model': model,
        'epoch': epoch,
        'acc': best_loss,
        'args': args,
    }
    if name == None:
        checkpoint_name = args.name + "_" +  str(epoch).zfill(3) + '_fold' + str(args.valid_fold) + "_"+str(round(best_loss, 3)) + '%.t7'
    else:
        checkpoint_name = args.name + "_" + name + ".t7"
    torch.save(state, os.path.join(args.ckpt_dir, checkpoint_name))    

if __name__ == '__main__':
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True

    train_loader, valid_loader= load_data(args.data_dir, args.label_dir)

    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\n************** Epoch: %d **************' % epoch)
        train(epoch, train_loader)
        print()
        valid_loss = valid(epoch, valid_loader)
        if epoch % 1 == 0 and args.random_patch:
            train_loader.dataset.random_patch()
        scheduler.step(valid_loss)


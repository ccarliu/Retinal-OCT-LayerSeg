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
parser.add_argument('--label_dir', default='./label', type=str, help='Directory to load label files.')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, help='Directory to save checkpoint.')

parser.add_argument('--gpu', default='0', type=str, help='GPU Devices to use.')
parser.add_argument('--batch_size', default=9, type=int, help='Batch size.')

parser.add_argument('--lr', default=0.001, type=float, help='Starting learning rate.')
parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay.')
parser.add_argument('--lr_decay_step', default=1, type=int, help='Learning rate decay step.')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 penalty for regularization.')
parser.add_argument('--weight_g', default=0.3, type=float, help='L2 penalty for regularization.')


parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch.')
parser.add_argument('--epochs', default=90, type=int, help='Number of training epochs.')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint or not.')
parser.add_argument('--store_last', action='store_true', help='store the last model.')
parser.add_argument('--resume_last', action='store_true', help='resume the last model.')
parser.add_argument('--test', action='store_true', help='resume the last model.')

parser.add_argument('--name', default='layers', type=str, help='The id of this train')
parser.add_argument('--valid_fold', default=0, type=int, help='Starting epoch.')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.resume_last:
    check = torch.load("")
    model = check["model"].module
    
else:
    model = UNet3D_dual_up(n_layer = 3)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model = model.to(device)
model = nn.DataParallel(model)

# prepare image loss
image_loss_func = MultiSurfaceCrossEntropyLoss()
grad_loss = Grad_2d(weight = torch.tensor(np.array([0, 0.8, 1.5]).astype(np.float32)).to(device))
smoothl1loss = nn.SmoothL1Loss()
meandis = nn.L1Loss()

# loss for segmentation branch
Gloss = GeneralizedDiceLoss()
mlloss = MultiLayerCrossEntropyLoss()

# loss for alignment
bce_Bscan = MSE_bscan()
ncc = NCC_oct().loss

# set best loss
best_loss = 10000

# set visualization writer
logpt = "./log/" + args.name
if not os.path.exists(logpt):
    os.mkdir(logpt)
writer = SummaryWriter(logpt)

def load_data(label_dir):
    valid_set = sd_oct_flattenp_align(label_dir, usage = 'valid', valid_fold = args.valid_fold)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    if args.test:
        return valid_loader, valid_loader
    train_set = sd_oct_flattenp_align(label_dir, usage = 'train', valid_fold = args.valid_fold)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True, pin_memory=True, num_workers=4)
    return train_loader, valid_loader

def train(epoch, train_loader):
    print("train:")
    model.train()
    surface_ces = []
    loss_sl1s = []
    loss_grads = []
    loss_gds = []

    loss_nccs = []
    loss_mses = []

    metric = [] # mean_dis
    
    for idx, (data, mask, _, _) in enumerate(train_loader):
        data = data.to(device)
        mask = mask.float().to(device)
        optimizer.zero_grad()
        
        s, logits, layerProb, mu, x, flow = model(data)

        # for the branch 2, the layer_label need to be align
        new_label = (mask - flow.detach())
        new_label[new_label > 319] = 319
        new_label[new_label < 0] = 0
        layer_gt = []
        for l in new_label:
            layer_gt.append(getLayerLabels(l, 320).unsqueeze(0))
        layer_gt = torch.cat(layer_gt, dim = 0)
        
        lossgd = Gloss(layerProb, layer_gt.float())  # diceloss
        lossgd += mlloss(layerProb, layer_gt.float())   # layer_cross_entropy
        surface_ce = image_loss_func(logits[:,:,:,:], new_label)  # surface_cross_entropy
        loss_grad = grad_loss.loss(mask, mu[:,:3,:,:]) # grad_loss
        lossmse = bce_Bscan.loss(mask - flow)

        ncc_loss = ncc(x)
        sl1loss = smoothl1loss(mu, new_label)   # smooth_l1_loss # why use smooth l1
        main_loss =  surface_ce + sl1loss + ncc_loss + lossgd + loss_grad * * args.weight_g + lossmse
 
        main_loss.backward()

        optimizer.step()
        metric.append(meandis(s[:,:,:,:], mask[:,:3,:,:] - flow.detach()).detach().cpu().numpy())
        loss_mses.append(lossmse.detach().cpu().numpy())
        loss_sl1s.append(sl1loss.detach().cpu().numpy())
        loss_grads.append(loss_grad.detach().cpu().numpy())
        loss_gds.append(lossgd.detach().cpu().numpy())
        surface_ces.append(surface_ce.detach().cpu().numpy())
        loss_nccs.append(ncc_loss.detach().cpu().numpy())
        smooth_loss = sum(loss_sl1s[-100:]) / min(len(loss_sl1s), 100)
        print(idx, sl1loss.item(), metric[-1], mean(surface_ces), surface_ce.item(), loss_grad.item(), ncc_loss.item(), end = "\r")
        
    loss = mean(loss_sl1s)
    print(mean(metric), mean(surface_ces), mean(loss_sl1s))
    writer.add_scalar('loss_sl1s/train', mean(loss_sl1s), epoch)
    writer.add_scalar('loss_grads/train', mean(loss_grads), epoch)
    writer.add_scalar('loss_mses/train', mean(loss_mses), epoch)
    writer.add_scalar('mean_dis/train', mean(metric), epoch)
    writer.add_scalar('surface_ce/train', mean(surface_ces), epoch)
    writer.add_scalar('loss_gds/train', mean(loss_gds), epoch)
    writer.add_scalar('ncc/train', mean(loss_nccs), epoch)


def valid(epoch, valid_loader, test = False):
    print("valid:")
    global best_loss

    model.eval()
    surface_ces = []
    loss_sl1s = []
    loss_grads = []
    loss_gds = []
    loss_nccs = []
    loss_mses = []
    if test:
        store_path = "/data3/layer_segmentation/final"
        if not os.path.exists(store_path):
            os.mkdir(store_path)
    metric = [] # mean_dis
    
    names = []
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name) in enumerate(valid_loader):
            data = data.to(device)
            mask = mask.float()
            s, logits, layerProb, mu, x, flow = model(data)
            
            new_label = (mask - flow.detach().cpu())
            new_label[new_label > 319] = 319
            new_label[new_label < 0] = 0
            
            surface_ce = image_loss_func(logits.detach().cpu(), new_label)  # surface_cross_entropy
            loss_grad = grad_loss.loss(mask, mu[:,:3,:,:].detach()) # grad_loss
            lossmse = bce_Bscan.loss(mask - flow.detach().cpu())

            ncc_loss = ncc(x)
            sl1loss = smoothl1loss(mu[:,:3,:,:].detach().cpu(), new_label)   # smooth_l1_loss # why use smooth l1

            mean_dis = np.abs(s.detach().cpu().numpy()[:,:,:,:] - new_label.numpy()).mean()
            metric.append(mean_dis)
            
            if test:
                res = x.detach().cpu().numpy()
                out = flow.detach().cpu().numpy()
                cpath = os.path.join(store_path, name[0])
                io.savemat(cpath, {"img":res.squeeze(), "flow" : out, "imgr": data.cpu().numpy()})
            
            loss_sl1s.append(sl1loss.detach().numpy())
            loss_grads.append(loss_grad.detach().cpu().numpy())
            loss_nccs.append(ncc_loss.detach().cpu().numpy())
            loss_mses.append(lossmse.detach().cpu().numpy())
            surface_ces.append(surface_ce.detach().cpu().numpy())
            names.append(name) 
            
            smooth_loss = sum(metric[-100:]) / min(len(metric), 100)
            print(idx, smooth_loss, end = "\r")
    
    print(mean(metric), mean(surface_ces), mean(loss_sl1s), mean(loss_grads), mean(loss_nccs))
    
    writer.add_scalar('loss_sl1s/valid', mean(loss_sl1s), epoch)
    writer.add_scalar('loss_grads/valid', mean(loss_grads), epoch)
    writer.add_scalar('loss_mses/valid', mean(loss_mses), epoch)
    writer.add_scalar('mean_dis/valid', mean(metric), epoch)
    writer.add_scalar('surface_ce/valid', mean(surface_ces), epoch)
    writer.add_scalar('ncc/valid', mean(loss_nccs), epoch)
    metric = mean(metric)
    if metric < best_loss:
        best_loss = metric
        save_checkpoint(epoch)
    return metric

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
    }
    if name == None:
        checkpoint_name = args.name + "_" +  str(epoch).zfill(3) + '_fold' + str(args.valid_fold) + "_"+str(round(best_loss, 3)) + '%.t7'
    else:
        checkpoint_name = args.name + "_" + name + ".t7"
    torch.save(state, os.path.join(args.ckpt_dir, checkpoint_name))    

if __name__ == '__main__':

    train_loader, valid_loader= load_data(args.label_dir)
    
    if args.test:
        valid(0, valid_loader, True)
        exit(0)
        
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\n************** Epoch: %d **************' % epoch)
        train(epoch, train_loader)
        print()
        valid_loss = valid(epoch, valid_loader)

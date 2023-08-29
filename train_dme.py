import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import torch
import cv2
import sys 
import csv
import numpy as np
from numpy import *
import torch.nn as nn
from torch.utils.data.sampler import  WeightedRandomSampler
from models import *
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

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
parser.add_argument('--batch_size', default=6, type=int, help='Batch size.')

parser.add_argument('--lr', default=0.001, type=float, help='Starting learning rate.')
parser.add_argument('--scheduler_type', default='cosine', type=str, help='GPU Devices to use.')
parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay.')
parser.add_argument('--lr_decay_step', default=1, type=int, help='Learning rate decay step.')
parser.add_argument('--label_inter', default=1, type=int, help='Sparse label inter.')

parser.add_argument('--weight_decay', default=0.0001, type=float, help='L2 penalty for regularization.')
parser.add_argument('--weight_g', default=1, type=float, help='L2 penalty for regularization.')


parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch.')
parser.add_argument('--epochs', default=120, type=int, help='Number of training epochs.')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint or not.')
parser.add_argument('--store_last', action='store_true', help='store the last model.')
parser.add_argument('--resume_last', action='store_true', help='resume the last model.')
parser.add_argument('--test', action='store_true', help='resume the last model.')
parser.add_argument('--patch_size', type=int, default=48, dest='patch_size')


parser.add_argument('--name', default='layers', type=str, help='The id of this train')
parser.add_argument('--valid_fold', default=0, type=int, help='Starting epoch.')

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

    def forward(self, src, flow):
        # We only focus on shift in one dimension, so the other two shift is zero.
        oshift1 = torch.zeros_like(flow, device = flow.device)
        oshift2 = torch.zeros_like(flow, device = flow.device)
        if self.axis == 1:
            flow = torch.cat([oshift1, flow, oshift2],1)

        # new locations
        new_locs = self.grid.to(flow.device) + flow
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

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode='reflection')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

Ascan_size = 224

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

patch_size = args.patch_size

if args.resume_last and False:
    check = torch.load("./checkpoint/fp_intermask_054_fold0_0.968%.t7")
    model = check["model"].module
    
else:
    model = UNet3D_dual_up_lesion(n_layer = 8, input_size = (Ascan_size, patch_size, 41))
    model.apply(weight_init)
    
# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = args.weight_decay)

if args.scheduler_type == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
elif args.scheduler_type == "reducelronpla" :
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

model = model.to(device)
model = nn.DataParallel(model)

# prepare image loss
image_loss_func = MultiSurfaceCrossEntropyLoss()
grad_loss = Grad_2d(weight = torch.tensor(np.array([0.15, 0.15, 0.18, 0.18, 0.20, 0.11, 0.11, 0.11]).astype(np.float32)).to(device) * 0.3)

#grad_loss = Grad_2d(weight = torch.tensor(np.array([0, 0.3, 0.5]).astype(np.float32)).to(device))
smoothl1loss = nn.SmoothL1Loss(reduction = "none")
meandis = nn.L1Loss(reduction = "none")

# loss for segmentation branch
Gloss = GeneralizedDiceLoss_lesion()
mlloss = MultiLayerCrossEntropyLoss_lesion()

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

def load_data(data_dir, label_dir):
    valid_set = sd_oct_flatten_ori_align_doe(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, asize = Ascan_size, patch_size = patch_size)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last = True)

    if args.test:
        return valid_loader, valid_loader
    train_set = sd_oct_flatten_ori_align_doe_lesion(data_dir, label_dir, usage = 'train', valid_fold = args.valid_fold, step = 30, asize = Ascan_size, patch_size = patch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True, pin_memory=True, num_workers=4, drop_last = True)
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
    
    for idx, (data, mask, _, lesion, cidx, nan_mask) in enumerate(train_loader):
        data = data.to(device)
        lesion = lesion.to(device)
        mask = mask.float().to(device)
        
        
        optimizer.zero_grad()
        s, logits, layerProb, mu, x, flow, lesion_a = model(data, lesion)
        
        cidx = [ll[0] for ll in cidx]
        
        # for the branch 2, the layer_label need to be align
        flow_t = flow.detach().clone()
        new_label = (mask - flow_t[:,:,:,cidx])
        new_label[new_label > Ascan_size-3] = Ascan_size-3
        new_label[new_label < 3] = 3
        new_label[torch.isnan(new_label)] = 0
        layer_gt = []
        lesion_a = lesion_a[:,:,:,cidx]
     
        for iidx, l in enumerate(new_label):
            
            cgt = getLayerLabels(l, Ascan_size)
            
            cgt[lesion_a[iidx] > 0.5] = 8+1
            layer_gt.append(cgt.unsqueeze(0))

        layer_gt = torch.cat(layer_gt, dim = 0)
        
        lossgd = Gloss(layerProb[:,:,:,:,cidx], layer_gt.float(), mask = nan_mask.unsqueeze(1).unsqueeze(1).to(layerProb.device))  # diceloss
        lossgd += mlloss(layerProb[:,:,:,:,cidx], layer_gt.float(), mask = nan_mask.unsqueeze(1).unsqueeze(1).to(layerProb.device))   # layer_cross_entropy
        surface_ce = image_loss_func(logits[:,:,:,:,cidx], new_label[:,:,:,:], mask = nan_mask.unsqueeze(1).unsqueeze(1).to(layerProb.device))  # surface_cross_entropy
        
        if torch.isnan(lossgd):
            lossgd = surface_ce * 0.1
        
        loss_grad = grad_loss.loss(mask, mu) # grad_loss
        if args.label_inter == 1 and False:
            lossmse = bce_Bscan.loss(mask - flow)
        else:
            lossmse = bce_Bscan.loss(mu.detach() + flow.detach() - flow)
        ncc_loss = ncc(x)
        sl1loss = smoothl1loss(mu[:,:,:,cidx], new_label[:,:,:,:])# smooth_l1_loss # why use smooth l1
        sl1loss = (sl1loss * nan_mask.unsqueeze(1).to(layer_gt.device)).mean()
        reg_loss = (flow * flow).mean()
        main_loss =  surface_ce + sl1loss + lossgd  + lossmse * 1 + loss_grad * args.weight_g  + reg_loss * 0.01
 
        main_loss.backward()

        optimizer.step()
        metric.append(meandis(s[:,:,:,cidx], mask[:,:,:,:] - flow[:,:,:,cidx].detach()).detach().cpu().numpy().mean())
        loss_mses.append(lossmse.detach().cpu().numpy())
        loss_sl1s.append(sl1loss.detach().cpu().numpy())
        loss_grads.append(loss_grad.detach().cpu().numpy())
        loss_gds.append(lossgd.detach().cpu().numpy())
        surface_ces.append(surface_ce.detach().cpu().numpy())
        loss_nccs.append(ncc_loss.detach().cpu().numpy())
        smooth_loss = sum(loss_sl1s[-100:]) / min(len(loss_sl1s), 100)
        print(idx, sl1loss.mean().item(), metric[-1], mean(loss_gds), mean(surface_ces), surface_ce.item(), loss_grad.item(), ncc_loss.item(), lossmse.item(), end = "\r")
        
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
    metric = [] 
    
    names = []
    with torch.no_grad():
        for idx, (data, mask, layer_gt, name, cidx, nan_mask) in enumerate(valid_loader):
            data = data.to(device)
            mask = mask.float().to(device)
            s, logits, layerProb, mu, x, flow = model(data)
            
            cidx = [ll[0] for ll in cidx]
            flow_t = flow.detach().clone()
            new_label = (mask - flow_t[:,:,:,cidx])
            new_label[new_label > Ascan_size-1] = Ascan_size-1
            new_label[new_label < 0] = 0
            
            surface_ce = image_loss_func(logits[:,:,:,:,cidx], new_label[:,:,:,:], mask = nan_mask.unsqueeze(1).unsqueeze(1).to(layerProb.device))  # surface_cross_entropy
            loss_grad = grad_loss.loss(mask, mu) # grad_loss
            lossmse = bce_Bscan.loss(new_label)

            ncc_loss = ncc(x)
            sl1loss = smoothl1loss(mu[:,:,:,cidx], new_label[:,:,:,:])# smooth_l1_loss # why use smooth l1
            sl1loss = (sl1loss * nan_mask.unsqueeze(1).to(sl1loss.device)).mean().cpu()
        
            mean_dis = torch.abs(s.detach().cpu()[:,:,:,cidx] - new_label.cpu())
            metric.append(mean_dis[nan_mask.unsqueeze(1).expand_as(mean_dis).bool()].mean())
            
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
    
    print(mean(metric), mean(surface_ces), mean(loss_sl1s), mean(loss_grads), mean(loss_nccs), mean(loss_mses))
    
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
    else:
        save_checkpoint(epoch, "last")
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
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 
    train_loader, valid_loader= load_data(args.data_dir, args.label_dir)
    
    if args.test:
        valid(0, valid_loader, True)
        exit(0)
        
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\n************** Epoch: %d **************' % epoch)
        train(epoch, train_loader)
        print()
        valid_loss = valid(epoch, valid_loader)

        scheduler.step(valid_loss)
        train_loader.dataset.load_data()


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
from apex import amp
import scipy.io as io


from dataset import *
from losses import *

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

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device.
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ck_name = "reproduce_final_grad1_060_fold0_0.734%.t7"
ck_name = "reproduce_final_grad1_7_063_fold0_0.74%.t7"
ck_name = "reproduce_final_grad0_8_038_fold0_0.719%.t7"
ck_name = "reproduce_final_grad01_8_038_fold0_0.726%.t7"
#ck_name = "reproduce_final_grado201_8_053_fold0_0.73%.t7"
ck_name = "reproduce_final_grado2005_8_038_fold0_0.73%.t7"
ck_name = "reproduce_final_grado4_w1_8_046_fold0_0.729%.t7"
ck_name = "reproduce_final_grado4_w1_9_040_fold0_0.724%.t7"
ck_name = "reproduce_final_grado4_w1_9_035_fold0_0.728%.t7"
#ck_name = "/home/lh/layer_segmentation_97/nni_best/reproduce_final_grado2005_8_038_fold0_0.73%.t7""

# load checkpoint
checkpoint = torch.load(os.path.join("./nni_best/", ck_name))
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

grad_loss = Grad_2d(weight = torch.tensor(np.array([1.0, 1, 1,1,1,1,1,1,1]).astype(np.float32)))
Gloss = GeneralizedDiceLoss()
smoothl1loss = nn.SmoothL1Loss()
mlloss = MultiLayerCrossEntropyLoss()
# set best loss
best_loss = 10000
ncc = NCC_oct_metric().loss

meandis = nn.L1Loss()

    
def load_data(data_dir, label_dir):
    #train_set = sd_oct_flattenp(data_dir, label_dir, usage = 'train', valid_fold = args.valid_fold)
    valid_set = hc_oct_noalign(data_dir, label_dir, usage = 'valid', valid_fold = args.valid_fold, inter = 10, morechannel = False, patch_length = args.patch_length)
    
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
    result = []
    flows = []
    target = []
    metric = [] # mean_dis
    store_path = ck_name[:-3] + "_align"
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
            s, logits, layerProb, mu,x, flow = model(data)
            #print(flow[:,1])
            #momun += flow[:,1].mean(0).squeeze().cpu()
            idx += 1
            #continue
            #print(flow)
            if torch.isnan(flow).sum() > 0:
                
                print("nan__", torch.isnan(flow).sum(), name)
                
            
            #print(s.shape, flow.shape, (s.cpu()[:,:,:,:]+flow.cpu()[:,0,:,:]).shape)
            result.append(s.cpu()[:,:,:,:]+flow.cpu()[:,:1,:,:])
            target.append(mask[:,:,:,:])
            flows.append(flow.cpu()[:,:1,:,:])
            names.extend(name) 
            
            if True:
                s1 = x.detach().cpu().numpy()
                out1 = flow.detach().cpu().numpy()
                cpath = os.path.join(store_path, name[0])
                io.savemat(cpath, {"img":s1.squeeze(), "flow" : out1, "imgr": data.cpu().numpy(), "mask": mask.numpy()})
            
            ncc_loss = ncc(x)
            
            continue
            lossgd = Gloss(layerProb, layer_gt)  # diceloss
            #lossgd += mlloss(layerProb, layer_gt)   # layer_cross_entropy
            surface_ce = image_loss_func(logits.cpu(), mask)  # surface_cross_entropy
            loss_grad = grad_loss.loss(mask, s.cpu()) # grad_loss
            
            sl1loss = smoothl1loss(mu.cpu(), mask)   # smooth_l1_loss # why use smooth l1
            #print(ncc(x), name, flow)
            
            metric.append(meandis(s.cpu(), mask-flow.detach().cpu()).detach())
            loss_sl1s.append(sl1loss.detach().numpy())
            loss_grads.append(loss_grad.detach().numpy())
            loss_gds.append(lossgd.detach().cpu().numpy())
            surface_ces.append(surface_ce.detach().cpu().numpy())
            


            smooth_loss = sum(metric[-100:]) / min(len(metric), 100)
            print(idx, smooth_loss, end = "\r")
            
    print()
    print(len(names))
    #print(momun / idx)
    result = torch.cat(result).numpy()
    target = torch.cat(target).numpy()
    flows = torch.cat(flows).numpy()
    print(result.shape)
    #cat_metric2(names, result, target)
    cat_metrica(names, result, target, flows)
    print(target.shape)    
    #print(mean(metric), mean(surface_ces), mean(loss_sl1s), mean(loss_grads), mean(loss_gds))
    
def cat_metric2(name, result, target):
    rootpath = "/data3/layer_segmentation/afterfp2/"
    name_list = list(set([l[:-6] for l in name]))
    print(len(name_list), name_list[:5])
    '''
    segs = []
    for l in name_list:
        cpath = os.path.join(rootpath, l)
        cur_data = scio.loadmat(cpath)
        cur_seg = np.round(cur_data["nlayer"][:,:,:])
        segs.append(np.transpose(cur_seg, (2,0,1)))
    segs = np.array(segs)
    '''
    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
    deno = np.zeros((len(name_list), nc, 1024, 49))  # denominator
    mulo = np.zeros((len(name_list), nc, 1024, 49))  # molecule
    
    denot = np.zeros((len(name_list), nc, 1024, 49))   # denominator
    mulot = np.zeros((len(name_list), nc, 1024, 49))   # molecule
    
    for idx, l in enumerate(name):
        #print(result[idx], target[idx])
        cname = l[:-6]
        widx = int(l[-3:])
        nameidx = name_list.index(cname)
        #print(nameidx)
        deno[nameidx, :, widx: widx + args.patch_length, :] = deno[nameidx, :, widx: widx + args.patch_length, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + args.patch_length, :] = denot[nameidx, :, widx: widx + args.patch_length, :] + target[idx,:,:,:]
        
        mulo[nameidx, :, widx: widx + args.patch_length, :] += 1
        mulot[nameidx, :, widx: widx + args.patch_length, :] += 1
    deno = deno[:,:,30:1000,:]
    denot = denot[:,:,30:1000,:]
    mulo = mulo[:,:,30:1000,:]
    mulot = mulot[:,:,30:1000,:]

    
    # print(mulot[0,2,2,:])
    # print(denot.shape, mulo.shape)
    deno = deno / mulo
    denot = denot / mulot
    print(np.abs(deno - denot).mean())
    means = np.mean(np.abs(np.round(deno)-np.round(denot)), (2,3))
    for idx, l in enumerate(name_list):
        print(l, means[idx])
    rmean = np.abs(np.round(deno)-np.round(denot))
    print(rmean.mean() * 3.9)
    print(np.mean(rmean, (0,2,3)) * 3.9)
    AMD_mean = rmean[AMD_list]
    NOR_mean = rmean[NOR_list]
    print("AMD:")
    print(np.mean(AMD_mean, (0,2,3)) * 3.9)
    
    print("NOR:")
    print(np.mean(NOR_mean, (0,2,3)) * 3.9)

        
def cat_metrica(name, result, target, flows):
    print(target.shape)
    rootpath = "/data3/layer_segmentation/afterfp2/"
    print(flows.shape)
    name_list = list(set([l[:-6] for l in name]))
    print(len(name_list), name_list[:5])
    '''
    segs = []
    for l in name_list:
        cpath = os.path.join(rootpath, l)
        cur_data = scio.loadmat(cpath)
        cur_seg = np.round(cur_data["nlayer"][:,:,:])
        segs.append(np.transpose(cur_seg, (2,0,1)))
    segs = np.array(segs)
    '''
    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
    deno = np.zeros((len(name_list), nc, 1024, 49))  # denominator
    mulo = np.zeros((len(name_list), nc, 1024, 49))  # molecule
    
    denot = np.zeros((len(name_list), nc, 1024, 49))   # denominator
    mulot = np.zeros((len(name_list), nc, 1024, 49))   # molecule

    ft = np.zeros((len(name_list), 1, 1, 49))
    mft = np.zeros((len(name_list),1,1,1)) 
    
    for idx, l in enumerate(name):
        #print(result[idx], target[idx])
        cname = l[:-6]
        widx = int(l[-3:])
        nameidx = name_list.index(cname)
        #print(nameidx)
        deno[nameidx, :, widx: widx + args.patch_length, :] = deno[nameidx, :, widx: widx + args.patch_length, :] + result[idx,:,:,:]
        denot[nameidx, :, widx: widx + args.patch_length, :] = denot[nameidx, :, widx: widx + args.patch_length, :] + target[idx,:,:,:]
        ft[nameidx, :, :, :] += flows[idx, 0,:,:]
        mulo[nameidx, :, widx: widx + args.patch_length, :] += 1
        mulot[nameidx, :, widx: widx + args.patch_length, :] += 1
        mft[nameidx, :,:,:] += 1
    #print(mulot[0,2,:,1])
    #print(deno[0,2,30:100,1])
    #print(deno[0,2,905:1000,1])
    # deno = deno[:,:,3:1021,:]
    # denot = denot[:,:,3:1021,:]
    # mulo = mulo[:,:,3:1021,:]
    # mulot = mulot[:,:,3:1021,:]
    ft = ft / mft
    # print(mulot[0,2,:,1])
    # print(mulot[0,2,2,:])
    # print(denot.shape, mulo.shape)
    deno = deno / mulo
    denot = denot / mulot
    print(np.abs(deno - denot).mean())
    means = np.mean(np.abs(np.round(deno)-np.round(denot)), (2,3))
    for idx, l in enumerate(name_list):
        print(l, means[idx])
    rmean = np.abs((deno)-(denot))
    print(rmean.mean() * 3.9, (rmean*3.9).std())
    
    print(np.mean(rmean, (0,2,3)) * 3.9, np.std((rmean*3.9), (0,2,3)))
    AMD_mean = rmean[AMD_list]
    NOR_mean = rmean[NOR_list]
    print("AMD:")
    print(np.mean(AMD_mean, (0,2,3)) * 3.9)
    
    print("NOR:")
    print(np.mean(NOR_mean, (0,2,3)) * 3.9)
    file = open(ck_name[:-3]+".csv", "w")
    file_c = csv.writer(file)
    save_path = "/data3/layer_segmentation/" + ck_name[:-3]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, l in enumerate(name_list):
        types = "control"
        if "AMD" in l:
            types = "AMD"
        
        file_c.writerow([l, types, *means[idx]])
        cpath = os.path.join(save_path, l + ".mat")
        io.savemat(cpath, {"prediction":deno[idx], "gt": denot[idx], "flow": ft[idx]})
    
def cat_metric(name, result, target):
    rootpath = "/data3/layer_segmentation/afterfp2/"
    name_list = list(set([l[:-9] for l in name]))
    print(len(name_list), name_list[:5])
    '''
    segs = []
    for l in name_list:
        cpath = os.path.join(rootpath, l)
        cur_data = scio.loadmat(cpath)
        cur_seg = np.round(cur_data["nlayer"][:,:,:])
        segs.append(np.transpose(cur_seg, (2,0,1)))
    segs = np.array(segs)
    '''
    n, nc, w, h = result.shape
    AMD_list = []
    NOR_list = []
    
    for idx,l in enumerate(name_list):
        if "AMD" in l:
            AMD_list.append(idx)
        else:
            NOR_list.append(idx)
    deno = np.zeros((len(name_list), nc, 1024, 48))  # denominator
    mulo = np.zeros((len(name_list), nc, 1024, 48))  # molecule
    
    denot = np.zeros((len(name_list), nc, 1024, 48))   # denominator
    mulot = np.zeros((len(name_list), nc, 1024, 48))   # molecule
    
    for idx, l in enumerate(name):
        #print(result[idx], target[idx])
        cname = l[:-9]
        widx = int(l[-6:-3])
        hidx = (int(l[-3:])-1) * 24
        print(widx, hidx)
        nameidx = name_list.index(cname)
        #print(nameidx)
        deno[nameidx, :, widx: widx + 48, hidx:hidx + 24] = deno[nameidx, :, widx: widx + 48, hidx:hidx + 24] + result[idx,:,:,::2]
        denot[nameidx, :, widx: widx + 48, hidx:hidx + 24] = denot[nameidx, :, widx: widx + 48, hidx:hidx + 24] + target[idx,:,:,::2]
        
        mulo[nameidx, :, widx: widx + 48, hidx:hidx + 24] += 1
        mulot[nameidx, :, widx: widx + 48, hidx:hidx + 24] += 1
    deno = deno[:,:,60:1000,:]
    denot = denot[:,:,60:1000,:]
    mulo = mulo[:,:,60:1000,:]
    mulot = mulot[:,:,60:1000,:]
    
    print(mulot[0,2,:,1])
    print(mulot[0,2,2,:])
    print(denot.shape, mulo.shape)
    deno = np.round(deno / mulo)
    denot = np.round(denot / mulot)
    print(np.abs(deno - denot).mean())
    means = np.mean(np.abs(np.round(deno)-np.round(denot)), (2,3))
    for idx, l in enumerate(name_list):
        print(l, means[idx])
    rmean = np.abs(np.round(deno)-np.round(denot))
    print(rmean.mean() * 3.9, rmean.std()*3.9)
    
    print(np.mean(rmean, (0,2,3)) * 3.9, np.std(rmean, (0,2,3)) * 3.9)
    AMD_mean = rmean[AMD_list]
    NOR_mean = rmean[NOR_list]
    print("AMD:")
    print(np.mean(AMD_mean, (0,2,3)) * 3.9)
    
    print("NOR:")
    print(np.mean(NOR_mean, (0,2,3)) * 3.9)
    file = open("result.csv", "w")
    file_c = csv.writer(file)
    save_path = "/data3/layer_segmentation/predictiontest/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx, l in enumerate(name_list):
        types = "control"
        if "AMD" in l:
            types = "AMD"
        
        file_c.writerow([l, types, *means[idx]])
        cpath = os.path.join(save_path, l)
        io.savemat(cpath, {"prediction":deno[idx], "gt": denot[idx]})
        


 
if __name__ == '__main__':
    valid_loader= load_data(args.data_dir, args.label_dir)
    valid(0, valid_loader)
    exit(0)
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\n************** Epoch: %d **************' % epoch)
        train(epoch, train_loader)
        print()
        valid_loss = valid(epoch, valid_loader)
        #scheduler.step()with open('redirect.txt', 'w') as f:
        '''
        old_stdout = sys.stdout
        with open("a" + str(epoch) + ".log", 'w') as f:
                
            sys.stdout = f
            for name, parms in model.named_parameters():
                print(name, parms)
                '''
        scheduler.step(valid_loss)

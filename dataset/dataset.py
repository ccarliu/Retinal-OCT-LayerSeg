
import os
import csv
import glob
import torch
import torch.nn.functional as F
import random
import torch.utils.data
import scipy.io as scio
import warnings
warnings.filterwarnings('ignore')
from numpy import *
import numpy as np
import scipy.ndimage

from sklearn import preprocessing

def getLayerLabels(surfaceLabels, height):

    if surfaceLabels is None:
        return None

    H = height
    device = surfaceLabels.device
    N, W, D = surfaceLabels.shape  # N is the number of surface
    layerLabels = torch.zeros((H, W, D), dtype=torch.long, device=device)
    surfaceLabels = (surfaceLabels + 0.5).long()  # let surface height match grid
    surfaceCodes = torch.tensor(range(1, N + 1), device=device).unsqueeze(dim=-1).unsqueeze(dim=-1).expand_as(surfaceLabels)
    layerLabels.scatter_(0, surfaceLabels, surfaceCodes)  #surface 0 in its exact location marks as 1, surface N-1 mark as N.

    for i in range(1,H):
        layerLabels[i,:, :] = torch.where(0 == layerLabels[i,:, :], layerLabels[i-1,:, :], layerLabels[i,:, :])

    return layerLabels

 
class sd_oct_flattenp_align(torch.utils.data.Dataset):
    """ 
        dataset for a2a_sd_oct.
        the train and valid sets are splited by the label.csv.
    """
    
    def __init__(self, label_dir, data_name = "a2a_oct", valid_fold = 0, usage = "train", transform=None, step=40):
        super(sd_oct_flattenp_align, self).__init__()

        self.data_path = []
        self.data = []
        self.seg = []
        self.name = []
        self.step = step
        self.usage = usage
        self.valid_fold = valid_fold

        # load file path and split train set and valid set.
        label_file = "labelflattenedp.csv"
        label_file_path = os.path.join(label_dir, label_file)
        label = open(label_file_path, "r")
        csv_reader = csv.reader(label)
    
        for path, fold in csv_reader:
            path = path.replace("afterfp2","flattened")
            if int(fold) == valid_fold and self.usage == "valid":
                self.data_path.append(path)
            elif int(fold) != valid_fold and self.usage == "train":
                self.data_path.append(path)
        print("load " + self.usage + " " + str(len(self.data_path)))
        
        # load data and seg
        for l in self.data_path:
            try:
                cur_data = scio.loadmat(l)
            except:
                print(l)
                continue
                
            cur_img = cur_data["data"][0,0]["flat_vol"][:, 300:700, 18:58]
            cur_seg = cur_data["data"][0,0]["bds"][300:700, 18:58, :]
            
            if cur_img.shape[0] != 320:
                print(cur_img.shape[0])
                continue
            if sum(np.isnan(cur_seg)) > 0:
                continue
            if cur_seg.min() < 5 or cur_seg.max() > 315:
                continue

            # normalization
            mu = np.mean(cur_img, axis=0)
            sigma = np.std(cur_img, axis=0)
            cur_img = (cur_img - mu) / sigma
            step = self.step

            for i in range((cur_img.shape[1] - 48) // step):
                cur_idx = i*step
                self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
                cur_idx = i*step
                self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + 48,:])
                self.seg.append(np.round(cur_seg).astype(np.float32)[cur_idx : cur_idx + 48,:,:])
            cur_idx = 400 - 48
            self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
            self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + 48,:])
            self.seg.append(np.round(cur_seg).astype(np.float32)[cur_idx : cur_idx + 48,:,:])
                
        print("load " + self.usage + " " + str(len(self.data)))

    def __getitem__(self, index):
        patch = self.data[index]
        name = self.name[index]
        seg = self.seg[index]
        
        layer_gt = getLayerLabels(torch.tensor(seg).permute(2,0,1), 320)
        return torch.tensor(patch).unsqueeze(0), torch.tensor(seg).permute(2,0,1), layer_gt, name

    def __len__(self):
        return len(self.data)

class sd_oct_flatten_ori_align_doe(torch.utils.data.Dataset):
    """ 
        dataset for a2a_sd_oct.
        the root should be "/data3/layer_segmentation/a2a_oct".
        the train and valid sets are splited by the label.csv.
    """
    
    def __init__(self, root, label_dir, data_name = "a2a_oct", valid_fold = 0, usage = "train", transform=None, inte_dis = False,step=30, test = False, asize = 224, patch_size = 48):
        super(sd_oct_flatten_ori_align_doe, self).__init__()

        self.data_path = []
        self.data = []
        self.seg = []
        self.label_idx = []
        self.name = []
        self.step = step
        self.test = test 
        self.inte_dis = inte_dis
        self.usage = usage
        self.valid_fold = valid_fold
        self.asize = asize
        self.patch_size = patch_size

        # load file path and split train set and valid set.
        files = glob.glob(os.path.join(root, "*.mat"))
        if self.usage == "train":
            self.data_path = sorted(files)[5:]
        else:
            self.data_path = sorted(files)[:5]
    
        print("load " + self.usage + " " + str(len(self.data_path)))
        self.load_data()
        
    def load_data(self):
    
        self.data = []
        self.seg = []
        self.label_idx = []
        self.name = []
        
        maxi = 0
        mini = 512
        # load data and seg
        for l in self.data_path:
            try:
                matInfo = scio.loadmat(l)
            except:
                print(l)
                continue
            
            
            cur_img = matInfo['images']            # size: (496, 768, 61)  -> (496, 512, 61)
            cur_seg = matInfo['manualLayers1']     # original paper uses MJA's segmentation result.  size: (8, 768, 61) -> (8, 512, 61)

            nansBscan = np.isnan(cur_seg).astype(int)  # size: 8x512x61
            nansBscan = np.sum(nansBscan, axis=(0,1))  # size: 61

            cidx = []
            for ii,l1 in enumerate(nansBscan):
                if l1 < 8*512:
                    cidx.append(ii)
            


            ll = cidx[0]
            cur_img = cur_img[:,:,ll:ll+41]
            cur_seg = cur_seg.transpose(1,2,0)
            z_indexes, y_indexes, x_indexes = np.nonzero(~np.isnan(cur_seg))
            zmin, ymin, xmin = [max(0, int(np.min(arr))) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            
            # random 1: for data augmentation
            if self.usage == "train":
                tempi = np.random.randint(0,10)
                cur_seg = cur_seg[zmin+tempi:zmax, cidx, :]
                cur_img = cur_img[:, zmin+tempi:zmax, :]
            else:
                tempi = 0
                zmin = 128
                zmax = 640
                cur_seg = cur_seg[zmin+tempi:zmax, cidx, :]
                cur_img = cur_img[:, zmin+tempi:zmax, :]
            
            
                       
            # random 2: for dataset augmentation
            if self.usage == "train":
                bv = np.random.randint(np.max([np.nanmax(cur_seg) - self.asize + 5, 0]), np.nanmin(cur_seg)-5)#int(np.nanmin(cur_seg) - 10)
            else:
                bv = int(np.nanmin(cur_seg) - 10)
            
            cur_img = cur_img[bv : bv + self.asize]
            
            cur_seg -= bv
            
            
            temp = cidx[0]
            cidx = [l-temp for l in cidx]
            
            if cur_seg.max() > maxi:
                maxi = cur_seg.max()
            if cur_seg.min() < mini:
                mini = cur_seg.min()
            

            # normalization
            mu = np.mean(cur_img, axis=0)
            sigma = np.std(cur_img, axis=0)
            cur_img = (cur_img - mu) / sigma
            
            # patch
            step = self.step
            for i in range((cur_img.shape[1] - self.patch_size) // step):
                cur_idx = i*step
                self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
                cur_idx = i*step
                self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + self.patch_size,:])
                self.seg.append((cur_seg).astype(np.float32)[cur_idx : cur_idx + self.patch_size,:,:])
                self.label_idx.append(cidx)
            cur_idx = zmax - zmin - tempi - self.patch_size
            self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
            self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + self.patch_size,:])
            self.label_idx.append(cidx)
            self.seg.append((cur_seg).astype(np.float32)[cur_idx : cur_idx + self.patch_size,:,:])
            
        print("load " + self.usage + " " + str(len(self.data)))

    def __getitem__(self, index):
        patch = self.data[index]
        name = self.name[index]
        seg = self.seg[index]
        cidx = self.label_idx[index]
                
        # data augmenation        
        if self.usage == "train":
            if np.random.random() > 0.7:
                patch = patch[:, ::-1, :].copy()
                seg = seg[::-1, :, :].copy()
        
        patch = torch.tensor(patch).unsqueeze(0)
        
        nan_mask = (np.isnan(seg).sum(-1) == 0).astype(np.float32)
        seg[np.isnan(seg)] = 110
        
        layer_gt = getLayerLabels(torch.tensor(seg).permute(2,0,1), self.asize)
        return patch, torch.tensor(seg).permute(2,0,1), layer_gt, name, cidx, nan_mask

    def __len__(self):
        return len(self.data)

class sd_oct_flatten_ori_align_doe_lesion(torch.utils.data.Dataset):
    """ 
        dataset for a2a_sd_oct.
        the root should be "/data3/layer_segmentation/a2a_oct".
        the train and valid sets are splited by the label.csv.
    """
    
    def __init__(self, root, label_dir, data_name = "a2a_oct", valid_fold = 0, usage = "train", transform=None, inte_dis = False,step=30, test = False, asize = 224, patch_size = 48):
        super(sd_oct_flatten_ori_align_doe_lesion, self).__init__()

        self.data_path = []
        self.data = []
        self.seg = []
        self.label_idx = []
        self.name = []
        self.step = step
        self.test = test 
        self.inte_dis = inte_dis
        self.usage = usage
        self.valid_fold = valid_fold
        self.asize = asize
        self.patch_size = patch_size

        # load file path and split train set and valid set.
        files = glob.glob(os.path.join(root, "*.mat"))
        if self.usage == "train":
            self.data_path = sorted(files)[5:]
        else:
            self.data_path = sorted(files)[:5]
    
        self.load_data()
        
    def load_data(self):
    
        self.data = []
        self.seg = []
        self.label_idx = []
        self.name = []
        self.lesions = []
        
        maxi = 0
        mini = 512
        
        # load data and seg
        for l in self.data_path:
            try:
                matInfo = scio.loadmat(l)
            except:
                print(l)
                continue
           
            
            cur_img = matInfo['images']                # size: (496, 768, 61)  -> (496, 512, 61)
            cur_seg = matInfo['manualLayers1']          # original paper uses MJA's segmentation result.  size: (8, 768, 61) -> (8, 512, 61)
            cur_lesion = matInfo['manualFluid1']
            
            nansBscan = np.isnan(cur_seg).astype(int)  # size: 8x512x61
            nansBscan = np.sum(nansBscan, axis=(0,1))  # size: 61

            cidx = []
            for ii,l1 in enumerate(nansBscan):
                if l1 < 8*512:
                    cidx.append(ii)


            ll = cidx[0]

            cur_img = cur_img[:,:,ll:ll+41]
            cur_lesion = cur_lesion[:, :, ll:ll+41]

            cur_seg = cur_seg.transpose(1,2,0)
            z_indexes, y_indexes, x_indexes = np.nonzero(~np.isnan(cur_seg))
            zmin, ymin, xmin = [max(0, int(np.min(arr))) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            
            # random 1: for data augmentation
            if self.usage == "train":
                tempi = np.random.randint(0,self.patch_size // 2)
            else:
                tempi = 0
            
            cur_seg = cur_seg[zmin+tempi:zmax, cidx, :]
            cur_img = cur_img[:, zmin+tempi:zmax, :]
            cur_lesion = cur_lesion[:, zmin+tempi:zmax, :]
            
            
            # random 2: for data augmentation
            if self.usage == "train":
                bv = np.random.randint(np.max([np.nanmax(cur_seg) - self.asize + 5, 0]), np.nanmin(cur_seg)-5)#int(np.nanmin(cur_seg) - 10)
            else:
                bv = int(np.nanmin(cur_seg) - 10)
            
            cur_img = cur_img[bv : bv + self.asize]
            cur_lesion = cur_lesion[bv : bv + self.asize]
            cur_lesion[cur_lesion > 0.5] = 1
            
            cur_seg -= bv            
            cur_seg[cur_seg < 0] = 0  
            cur_seg[cur_seg > self.asize] = self.asize
            
            
            temp = cidx[0]
            cidx = [l-temp for l in cidx]
            
            if cur_seg.max() > maxi:
                maxi = cur_seg.max()
            if cur_seg.min() < mini:
                mini = cur_seg.min()
            
            # normalization
            mu = np.mean(cur_img, axis=0)
            sigma = np.std(cur_img, axis=0)
            cur_img = (cur_img - mu) / sigma
            
            
            # patch
            step = self.step
            for i in range((cur_img.shape[1] - self.patch_size) // step):
                cur_idx = i*step
                self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
                cur_idx = i*step
                self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + self.patch_size,:])
                self.seg.append((cur_seg).astype(np.float32)[cur_idx : cur_idx + self.patch_size,:,:])
                self.label_idx.append(cidx)
                self.lesions.append((cur_lesion).astype(np.float32)[:, cur_idx:cur_idx + self.patch_size, :])
            cur_idx = zmax - zmin - tempi - self.patch_size
            self.name.append(l.split("/")[-1]+str(i).zfill(3)+str(cur_idx).zfill(3))
            self.data.append(cur_img.astype(np.float32)[:,cur_idx : cur_idx + self.patch_size,:])
            self.label_idx.append(cidx)
            self.seg.append((cur_seg).astype(np.float32)[cur_idx : cur_idx + self.patch_size,:,:])
            self.lesions.append((cur_lesion).astype(np.float32)[:, cur_idx:cur_idx + self.patch_size, :])
            
        print("load " + self.usage + " " + str(len(self.data)))

    def __getitem__(self, index):
        patch = self.data[index]
        name = self.name[index]
        seg = self.seg[index]
        cidx = self.label_idx[index]
        clesion = self.lesions[index]
        
        # for data augmentation
        if self.usage == "train":
            if np.random.random() > 0.7:
                patch = patch[:, ::-1, :].copy()
                seg = seg[::-1, :, :].copy()
                clesion = clesion[:, ::-1, :].copy()
        
        patch = torch.tensor(patch).unsqueeze(0)
        clesion = torch.tensor(clesion).unsqueeze(0)
        
        nan_mask = (np.isnan(seg).sum(-1) == 0).astype(np.float32)
        seg[np.isnan(seg)] = 110
        clesion[np.isnan(clesion)] = 0
        
        layer_gt = getLayerLabels(torch.tensor(seg).permute(2,0,1), self.asize)
        if self.usage == "train":
            return patch, torch.tensor(seg).permute(2,0,1), layer_gt, clesion, cidx, nan_mask
        else:
            return patch, torch.tensor(seg).permute(2,0,1), layer_gt, name, cidx, nan_mask

    def __len__(self):
        return len(self.data)
        


        

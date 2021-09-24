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
        the root should be "/data3/layer_segmentation/a2a_oct".
        the train and valid sets are splited by the label.csv.
    """
    
    def __init__(self, label_dir, data_name = "a2a_oct", valid_fold = 0, usage = "train", transform=None, inte_dis = False,step=40):
        super(sd_oct_flattenp_align, self).__init__()

        self.data_path = []
        self.data = []
        self.seg = []
        self.name = []
        self.step = step
        self.inte_dis = inte_dis
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
        maxi = 0
        mini = 512
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
            if cur_seg.max() > maxi:
                maxi = cur_seg.max()
            if cur_seg.min() < mini:
                mini = cur_seg.min()

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
        


        



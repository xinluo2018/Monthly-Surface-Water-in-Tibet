
import glob
import torch
import torch.nn as nn
from model.loss import FocalLoss
from utils.img_aug import rotate, flip, torch_noise, missing, numpy2tensor
from utils.img_aug import colorjitter, bandjitter

## -------- root directory --------  ## 
root = "/home/yons/Desktop/developer-luo/SWatNet"

# ------------ data paths -------------- #
# --- scene path for training ---
paths_as = sorted(glob.glob(root+'/data/s1_ascend/*'))
paths_des = sorted(glob.glob(root+'/data/s1_descend/*'))
paths_truth = sorted(glob.glob(root+'/data/s1_truth/*'))
# --- patch path for validation ---
paths_patch_val = sorted(glob.glob(root+'/data/val_patches/*'))

## ----------Training configuration------- ##
## parameter setting
epoch = 200
lr = 0.001
batch_size = 16

## loss function
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()
loss_focal = FocalLoss()

## label_smooth
def label_smooth(image_label, label_smooth = 0.1):
    label_smooth = 0.1
    image_label =  image_label + label_smooth
    image_label = torch.clamp(image_label, label_smooth, 1-label_smooth)
    return image_label

## --------- data pre-processing -------- ##
s1_min = [-57.78, -70.37, -58.98, -68.47]  # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]   # as-vv, as-vh, des-vv, des-vh

transforms_tra = [
        colorjitter(prob=0.5),
        bandjitter(prob=0.5),
        rotate(prob=0.3), 
        flip(prob=0.3), 
        missing(prob=0.3, ratio_max = 0.25),
        numpy2tensor(), 
        torch_noise(prob=0.3, std_min=0.005, std_max=0.1),
            ]


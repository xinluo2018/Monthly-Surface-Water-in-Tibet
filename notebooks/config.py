## author: xin luo, 
## created: 2021.7.8
## modify: 2021.10.13

import torch
import torch.nn as nn
from model.loss import FocalLoss
from dataloader.img_aug import missing_band_p, rotate, flip, torch_noise, missing_region, numpy2tensor
from dataloader.img_aug import missing_line, missing_band
from dataloader.img_aug import colorjitter, bandjitter

## ------------- Path -------------- ##
# -------- root directory --------  #
root_tb_data = '/myDrive/tibet-water'
root_proj = "/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet"

# ------------ data directory -------------- #
# --- scene dir path for training ---
dir_as = root_proj + '/data/dset/s1_ascend'
dir_des = root_proj + '/data/dset/s1_descend'
dir_truth = root_proj + '/data/dset/s1_truth'

# --- patch dir for validation ---
dir_patch_val = root_proj + '/data/dset/val_patches'


## --------- data loader -------- ##
s1_min = [-57.78, -70.37, -58.98, -68.47]  # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]   # as-vv, as-vh, des-vv, des-vh
# s1_min_per = [-25.95684, -38.16688, -29.07279, -38.11438]  # as-vv, as-vh, des-vv, des-vh
# s1_max_per = [-2.58706, -12.21354, -2.91177, -12.68447]   # as-vv, as-vh, des-vv, des-vh


def missing_line_aug(prob = 0.3):    # implemented in the parallel_loader.py
    return missing_line(prob=prob)

transforms_tra = [
        colorjitter(prob=0.3, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small
        # bandjitter(prob=0.3),     # numpy-based
        rotate(prob = 0.3),         # numpy-based
        flip(prob = 0.3),           # numpy-based
        missing_region(prob = 0.3, ratio_max = 0.2),   # numpy-based
        missing_band_p(prob = 0.3, ratio_max = 0.5),    # numpy-based
        numpy2tensor(), 
        torch_noise(prob=0.3, std_min=0, std_max=0.1),      # tensor-based
            ]

## ---------- model training ------- ##
# ----- parameter setting
epoch = 200
lr = 0.001  # select
# lr = 0.01
batch_size = 16  # select

# ----- loss function
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()
loss_focal = FocalLoss()

# ----- label_smooth
def label_smooth(image_label, label_smooth = 0.1):
    image_label =  image_label + label_smooth
    image_label = torch.clamp(image_label, label_smooth, 1-label_smooth)
    return image_label



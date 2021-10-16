## author: xin luo, 
## created: 2021.7.8
## modify: 2021.10.13

import torch
import torch.nn as nn
from model.loss import FocalLoss
from dataloader.img_aug import missing_band_p, rotate, flip, torch_noise, missing_region, numpy2tensor
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

transforms_tra = [
        colorjitter(prob=0.5),    # numpy-based
        # bandjitter(prob=0.5),     # numpy-based
        rotate(prob=0.5),         # numpy-based
        flip(prob=0.5),           # numpy-based
        missing_region(prob=0.2, ratio_max = 0.25),   # numpy-based
        missing_band_p(prob=0.2, ratio_max = 1),   # numpy-based
        numpy2tensor(), 
        torch_noise(prob=0.5, std_min=0.005, std_max=0.1),      # tensor-based
                ]

## ---------- model training ------- ##
# ----- parameter setting
epoch = 200
lr = 0.001
batch_size = 16

# ----- loss function
loss_ce = nn.CrossEntropyLoss()
loss_bce = nn.BCELoss()
loss_focal = FocalLoss()

# ----- label_smooth
def label_smooth(image_label, label_smooth = 0.1):
    label_smooth = 0.1
    image_label =  image_label + label_smooth
    image_label = torch.clamp(image_label, label_smooth, 1-label_smooth)
    return image_label



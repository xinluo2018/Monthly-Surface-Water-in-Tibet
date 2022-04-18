## author: xin luo, 
## created: 2021.7.8, modify: xxxx
## modify: 2021.10.13

import torch
import torch.nn as nn
from model.loss import FocalLoss
from dataloader.img_aug import missing_band_p, rotate, flip, torch_noise, missing_region, numpy2tensor
from dataloader.img_aug import missing_line, missing_band
from dataloader.img_aug import colorjitter, bandjitter

## ------------- Path -------------- ##
# -------- root directory --------  #
root_tb_data = '/WD-myBook/tibet-water'
root_proj = "/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet"

# ------------ data directory -------------- #
# --- scene dir path for training ---
dir_as = root_proj + '/data/dset/s1_ascend'
dir_des = root_proj + '/data/dset/s1_descend'
dir_truth = root_proj + '/data/dset/s1_truth'

## -------- train/validation data spliting --------
# val_ids = [0,2,7,10,14,18,23,31,35,38]   ## !!!note: the scene id is start with 1 in our dataset.
# tra_ids = list(set([i for i in range(39)])-set((val_ids)))

val_ids = ['03','05','08','11','15','19','24','31','37','39']
tra_ids= ['01','02','04','06','07','09','10','12','13','14','16',
          '17','18','20','21','22','23','25','26','27','28','29',
          '30','32','33','34','35','36','38']


# --- patch dir for validation ---
dir_patch_val = root_proj + '/data/dset/s1_val_patches'

## --------- data loader -------- ##
s1_min = [-63.00, -70.37, -59.01, -69.94]  # as-vv, as-vh, des-vv, des-vh
s1_max = [30.61, 13.71, 29.28, 17.60]      # as-vv, as-vh, des-vv, des-vh

def missing_line_aug(prob = 0.2):    # implemented in the parallel_loader.py
    return missing_line(prob=prob)

transforms_tra = [
        ### !!!note: line missing is in the paraller_loader.py
        colorjitter(prob=0.25, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small
        # bandjitter(prob=0.3),     # numpy-based
        rotate(prob = 0.25),         # numpy-based
        flip(prob = 0.25),           # numpy-based
        missing_region(prob = 0.25, ratio_max = 0.2),   # numpy-based
        missing_band_p(prob = 0.25, ratio_max = 0.2),    # numpy-based
        numpy2tensor(), 
        torch_noise(prob=0.25, std_min=0, std_max=0.1),      # tensor-based
            ]

## ---------- model training ------- ##
# ----- parameter setting
epoch = 300  # select
lr = 0.0002  # select
# epoch = 5  # tmp
# lr = 0.00001  # tmp
batch_size = 16  # select

# ----- loss function
loss_ce = nn.CrossEntropyLoss()  
loss_bce = nn.BCELoss()    # selected
loss_focal = FocalLoss()

# ----- label_smooth
def label_smooth(image_label, label_smooth = 0.1):
    image_label =  image_label + label_smooth
    image_label = torch.clamp(image_label, label_smooth, 1-label_smooth)
    return image_label



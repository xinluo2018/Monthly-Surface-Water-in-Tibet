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
dir_as = root_proj + '/data/dset/s1_ascend_clean'
dir_des = root_proj + '/data/dset/s1_descend_clean'
dir_truth = root_proj + '/data/dset/s1_truth_clean'

## -------- train/validation data spliting --------

# val_ids = ['03','06','08','11','15','19','24','31','39']
# tra_ids= ['01','02','04','05','07','09','10','12','13','14','16',
        #   '17','18','20','21','22','23','25','26','27','28','29',
        #   '30','32','33','34','35','36','37','38']
val_ids = ['01','02','03','04','05','06','07']
tra_ids= ['08','09','10','11','12','13','14','15','16','17','18',
          '19','20','21','22','23','24','25','26','27','28','29',
          '30','31','32','33','34','35','36','37','38','39']


# --- patch dir for validation ---
dir_patch_val = root_proj + '/data/dset/s1_val_patches'

## --------- data loader -------- ##
s1_min = [-63.00, -70.37, -59.01, -69.94]  # as-vv, as-vh, des-vv, des-vh
s1_max = [30.61, 13.71, 29.28, 17.60]      # as-vv, as-vh, des-vv, des-vh

def missing_line_aug(prob=0.3):    # implemented in the parallel_loader.py
    return missing_line(prob=prob)

transforms_tra = [
        ### !!!note: line missing is in the paraller_loader.py
        colorjitter(prob=0.3, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small
        # bandjitter(prob=0.2),     # numpy-based 
        rotate(prob=0.3),           # numpy-based
        flip(prob=0.3),             # numpy-based
        missing_region(prob=0.3, ratio_max=0.2),    # numpy-based
        missing_band_p(prob=0.3, ratio_max=0.2),    # numpy-based
        numpy2tensor(), 
        torch_noise(prob=0.3, std_min=0, std_max=0.1),      # tensor-based
            ]

## ---------- model training ------- ##
# ----- parameter setting
lr = 0.0002  # if use lr_scheduler;
batch_size = 16  # select

# ----- loss function
loss_ce = nn.CrossEntropyLoss()  
loss_bce = nn.BCELoss()    # selected
loss_focal = FocalLoss()

## ----- label_smooth
def label_smooth(image_label, label_smooth = 0.1):
    image_label =  image_label + label_smooth
    image_label = torch.clamp(image_label, label_smooth, 1-label_smooth)
    return image_label


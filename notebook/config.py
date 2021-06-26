
# root directory
root = "/home/yons/Desktop/developer-luo/SWatNet"

import sys
sys.path.append(root)
from utils.img_aug import rotate, flip, torch_noise, missing, numpy2tensor

##----------Training parameter------- ##
epoch = 100
lr = 0.001
batch_size = 16

##---------data pre-processing-------- ##
s1_min = [-57.78, -70.37, -58.98, -68.47]
s1_max = [25.98, 10.23, 29.28, 17.60]

transforms_tra = [
        rotate(prob=0.5), 
        flip(prob=0.5), 
        missing(prob=0.5, ratio_max = 0.25),
        numpy2tensor(), 
        torch_noise(prob=0.5, std_min=0.001, std_max=0.1)
            ]


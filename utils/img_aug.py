'''!!! note: all the input patch are multi-scale patches'''
import torch
import random
import copy
import numpy as np


class numpy2tensor:
    '''des: np.array to torch.Tensor
    '''
    def __call__(self, patch_group, ptruth):
        patch_group_tensor = [torch.from_numpy(patch.copy()).float() for patch in patch_group]
        ptruth_tensor = torch.from_numpy(ptruth.copy())
        return patch_group_tensor, ptruth_tensor

#### ----------- numpy-based --------------
#### --------------------------------------

class rotate:
    '''numpy-based
       des: randomly rotation with given probability
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else:
            k = random.randint(1,3)
            patches_rot = [np.rot90(patch, k, [1, 2]) for patch in patches]
            truth_rot = np.rot90(truth, k, [0, 1])
            return patches_rot, truth_rot

class flip: 
    '''numpy-based
       des: randomly flip with given probability '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''input image, truth are np.array'''
        if random.random() > self.p:
            return patches, truth
        else: 
            if random.random() < 0.5:            # up <-> down
                patches_flip = [np.flip(patch, 1) for patch in patches] 
                truth_flip = np.flip(truth, 0)
            else:                                # left <-> right
                patches_flip = [np.flip(patch, 2) for patch in patches]
                truth_flip = np.flip(truth, 1)
            return patches_flip, truth_flip

class missing:
    '''numpy-based
       des: randomly stripe missing'''
    def __init__(self, prob=0.5, ratio_max = 0.25):
        self.p = prob
        self.ratio_max = ratio_max
    def __call__(self, patches, truth):
        if random.random() > self.p:
            return patches, truth
        else:
            patches_miss = copy.deepcopy(patches)
            h_img = truth.shape[0]
            miss_max = int(h_img*self.ratio_max)
            for i in range(len(patches)):
                h_miss = random.randint(0, miss_max)
                w_miss = random.randint(0, miss_max)
                start_h = random.randint(0, h_img-h_miss)
                start_w = random.randint(0, h_img-h_miss)
                patches_miss[i][:,start_h:start_h+h_miss, start_w:start_w+w_miss] = 0
            return patches_miss, truth

class noise:
    '''numpy-based 
       !!! slower than torch-based 
       des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max =0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else:
            patches_noisy = []
            for i in range(len(patches)):
                std = random.uniform(self.std_min, self.std_max)
                noise = np.random.normal(loc=0, scale=std, size=patches[i].shape)
                patches_noisy.append(patches[i]+noise)
            return patches_noisy, truth


#### -------------- torch-based ----------------
#### -------------------------------------------
class torch_rotate:
    '''torch-based
       des: randomly rotation with given probability
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else:
            k = random.randint(1,3)
            patches_rot = [torch.rot90(patch, k, [1, 2]) for patch in patches]
            truth_rot = torch.rot90(truth, k, [0, 1])
        return patches_rot, truth_rot

class torch_noise:
    '''torch-based: faster than numpy-based
       des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max =0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else: 
            patches_noisy = []
            for i in range(len(patches)):
                std = random.uniform(self.std_min, self.std_max)
                noise = torch.normal(mean=0, std=std, size=patches[i].size())
                patches_noisy.append(patches[i].add(noise))
            return patches_noisy, truth




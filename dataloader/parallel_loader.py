## author: xin luo
## create: 2021.9.15
## des: 

import torch
import threading as td
from queue import Queue
from dataloader.preprocess import crop_scales, crop
from dataloader.img_aug import missing_line, missing_band
import numpy as np
import gc

def scenes2patches(scene_list, truth_list, transforms, scales=[2048, 512, 256]):    
    patch_list, ptruth_list = [],[]
     #'''convert image to patches group'''
    zip_data = list(zip(scene_list, truth_list))
    for scene, truth in zip_data:
        patches_group, truth = crop_scales(scales=scales, threads=False)(scene, truth)
        for transform in transforms:
            patches_group, truth = transform(patches_group, truth)
        truth = torch.unsqueeze(truth,0)
        patch_list.append(patches_group), ptruth_list.append(truth)
    return patch_list, ptruth_list


def job_scenes2patches(q, scene_list, truth_list, transforms):
    patch_list, ptruth_list = scenes2patches(scene_list, truth_list, transforms, scales=[2048, 512, 256])
    q.put((patch_list, ptruth_list))


def threads_read(scene_list, truth_list, transforms, num_thread=20):
    '''multi-thread reading training data
        cooperated with the job function
    '''
    patch_lists, ptruth_lists = [], []
    q = Queue()
    threads = [td.Thread(target=job_scenes2patches, args=(q, scene_list, \
                                    truth_list, transforms)) for i in range(num_thread)]
    start = [t.start() for t in threads]
    join = [t.join() for t in threads]
    for i in range(num_thread):
        patch_list, ptruth_list = q.get()
        patch_lists += patch_list
        ptruth_lists += ptruth_list
    return patch_lists, ptruth_lists


class threads_scene_dset(torch.utils.data.Dataset):
    ''' des: dataset (patch and the truth) parallel reading from RAM memory
        input: 
            patch_list, truth_list are lists (consist of torch.tensor).
            num_thread: number of threads
    '''
    def __init__(self, scene_list, truth_list, transforms, num_thread):

        self.scene_list = scene_list
        self.truth_list = truth_list 
        self.num_thread = num_thread
        # !!! significantly improve the performance. 
        self.scene_list = [missing_line(prob=0.3)(scene=scene) for scene in self.scene_list]
        self.patches_list, self.ptruth_list = threads_read(\
                                        scene_list, truth_list, transforms, num_thread)
        self.transforms = transforms
    def __getitem__(self, index): 
        '''load patches and truths'''
        patch = self.patches_list[index]
        truth = self.ptruth_list[index]
        if index == len(self.patches_list)-1:          ## update the dataset
            del self.patches_list, self.ptruth_list
            gc.collect()
            # !!! significantly improve the performance.
            self.scene_list = [missing_line(prob=0.3)(scene=scene) for scene in self.scene_list]
            # self.scene_list = [missing_band(prob=0.1)(scene=scene) for scene in self.scene_list]

            self.patches_list, self.ptruth_list = threads_read(\
                            self.scene_list, self.truth_list, self.transforms, self.num_thread)
        return patch, truth

    def __len__(self):
        return len(self.patches_list)



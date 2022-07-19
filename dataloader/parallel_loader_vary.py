## author: xin luo
## create: 2021.9.15
## des: compared with paraller_loader, the patch size is varing 
#       from 128 to the minimal of image height/width, and the size
#       then be resized to 256x256. in fact, we made the input resolution
#       varing 


from numpy.lib.function_base import delete
import torch
import threading as td
from queue import Queue
from dataloader.preprocess import crop, crop_scales
import random
import numpy as np
import cv2
import gc

def scenes2patches_vary(scene_list, truth_list, transforms):  
    '''des: comparing to scenes2patches, the num of patch is only one, 
            and the patch size is random generated'''  
    patch_list, ptruth_list = [],[]
     #'''convert image to patches group'''
    zip_data = list(zip(scene_list, truth_list))
    for scene, truth in zip_data:
        scene, truth = scene[:, 892:-892, 892:-892], truth[892:-892, 892:-892]
        size_crop = random.randint(128, min(truth.shape))    ## patch size: > 128, <min(truth.shape)
        patch, ptruth = crop(scene, truth, size=(size_crop, size_crop))
        ## resize to (256, 256)
        patch = [cv2.resize(patch[i], dsize=(256, 256), \
                            interpolation=cv2.INTER_LINEAR) for i in range(patch.shape[0])]
        patch = np.array(patch)
        ptruth = cv2.resize(ptruth, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        for transform in transforms:
            patch, ptruth = transform(patch, ptruth)
        ptruth = torch.unsqueeze(ptruth, 0)
        patch_list.append(patch), ptruth_list.append(ptruth)
    return patch_list, ptruth_list


def job_scenes2patches_vary(q, scene_list, truth_list, transforms):
    '''match to scenes2patches_'''
    patch_list, ptruth_list = scenes2patches_vary(scene_list, truth_list, transforms)
    q.put((patch_list, ptruth_list))


def threads_read_vary(scene_list, truth_list, transforms, num_thread=20):
    '''multi-thread reading training data
        cooperated with the job function
    '''
    patch_lists, ptruth_lists = [], []
    q = Queue()
    threads = [td.Thread(target=job_scenes2patches_vary, args=(q, scene_list, \
                                    truth_list, transforms)) for i in range(num_thread)]
    start = [t.start() for t in threads]
    join = [t.join() for t in threads]
    for i in range(num_thread):
        patch_list, ptruth_list = q.get()
        patch_lists += patch_list
        ptruth_lists += ptruth_list
    return patch_lists, ptruth_lists

class threads_scene_dset_vary(torch.utils.data.Dataset):
    ''' des: dataset (patch and the truth) parallel reading from RAM memory
        input: 
            patch_list, truth_list are lists (consist of torch.tensor).
            num_thread: number of threads
    '''
    def __init__(self, scene_list, truth_list, transforms, num_thread):
        self.scene_list = scene_list
        self.truth_list = truth_list 
        self.num_thread = num_thread
        self.transforms = transforms
        self.patches_list, self.ptruth_list = threads_read_vary(\
                                self.scene_list, self.truth_list, self.transforms, self.num_thread)

    def __getitem__(self, index): 
        '''load patches and truths'''
        patch = self.patches_list[index]
        ptruth = self.ptruth_list[index]
        if index == len(self.patches_list)-1:    ## update the dataset
            del self.patches_list, self.ptruth_list
            gc.collect()
            self.patches_list, self.ptruth_list = threads_read_vary(\
                                self.scene_list, self.truth_list, self.transforms, self.num_thread)
        return patch, ptruth

    def __len__(self):
        return len(self.scene_list)*self.num_thread


## author: xin luo
## create: 2021.9.9
## des: simple pre-processing for the dset data(image and truth pair).

import numpy as np
import random
import cv2
from utils.geotif_io import readTiff
import threading as td
from queue import Queue

## max and min value for each band of the s1 image.
s1_min = [-57.78, -70.37, -58.98, -68.47]  # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]


class normalize:
    '''normalization with the given per-band max and min values'''
    def __init__(self, max, min):
        '''max, min: list, values corresponding to each band'''
        self.max, self.min = max, min
    def __call__(self, image, truth):
        for band in range(image.shape[-1]):
            image[:,:,band] = (image[:,:,band]-self.min[band])/(self.max[band]-self.min[band]+0.0001)
        image = np.clip(image, 0., 1.) 
        return image, truth


def read_normalize(paths_as, paths_des, paths_truth):
    ''' des: data (s1 ascending, s1 descending and truth) reading 
             and preprocessing
        input: 
            ascend image, descend image and truth image paths
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    for i in range(len(paths_as)):
        ## --- data reading
        ascend, _ = readTiff(paths_as[i])
        descend, _ = readTiff(paths_des[i])
        truth, _ = readTiff(paths_truth[i])
        ## --- data normalization 
        scene = np.concatenate((ascend, descend), axis=-1)    
        scene, truth = normalize(max=s1_max, min=s1_min)(scene, truth)
        scene = scene.transpose(2,0,1)   # channel first
        scene[np.isnan(scene)]=0         # remove nan value
        scene_list.append(scene), truth_list.append(truth)
    return scene_list, truth_list


def crop(image, truth, size=(256, 256)):
    ''' numpy-based
        des: randomly crop corresponding to specific size
        input image and truth are np.array
        input size: (height, width)'''
    start_h = random.randint(0, truth.shape[0]-size[0])
    start_w = random.randint(0, truth.shape[1]-size[1])
    patch = image[:,start_h:start_h+size[0], start_w:start_w+size[1]]
    ptruth = truth[start_h:start_h+size[0], start_w:start_w+size[1]]
    return patch, ptruth


class crop_scales:
    ''' numpy-based
        des: randomly crop multiple-scale patches (from high to low)
        input scales: tuple or list (high -> low)
        we design a multi-thread processsing for resizing
    '''
    def __init__(self, scales=(2048, 512, 256), threads=False):
        self.scales = scales
        self.threads = threads

    def job(self, q, band):
        band_down = cv2.resize(src=band, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        q.put((band_down))

    def threads_process(self, patch):
        patch_down = []
        q = Queue()
        threads = [td.Thread(target=self.job, args=(q, patch[i])) for i in range(patch.shape[0])]
        start = [t.start() for t in threads]
        join = [t.join() for t in threads]
        for i in range(len(threads)):
            band_down = q.get()
            patch_down.append(band_down)
        patch_down = np.array(patch_down)
        return patch_down

    def __call__(self, image, truth):
        '''input image and turth are np.array'''
        patches_group = []
        patch_high, ptruth_high = crop(image, truth, size=(self.scales[0], self.scales[0]))
        patches_group.append(patch_high)
        for scale in self.scales[1:]:
            start_offset = (self.scales[0]-scale)//2
            patch_lower = patch_high[:, start_offset:start_offset+scale, \
                                                    start_offset:start_offset+scale]
            patches_group.append(patch_lower)
        ptruth = ptruth_high[start_offset:start_offset + scale, \
                                                    start_offset:start_offset+scale]        
        patches_group_down = []
        for patch in patches_group[:-1]:
            if self.threads:
                patch_down = self.threads_process(patch)
            else:
                patch_down=[cv2.resize(patch[num], dsize=(self.scales[-1], self.scales[-1]), \
                                    interpolation=cv2.INTER_LINEAR) for num in range(patch.shape[0])]
            patches_group_down.append(np.array(patch_down))
        patches_group_down.append(patch_lower)

        return patches_group_down, ptruth


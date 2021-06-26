'''simple pre-processing'''
import torch
import numpy as np
import random
import cv2
import threading as td
from queue import Queue

# class toTensor:
#     """Convert ndarrays to torch.Tensors (channel, height, width)."""
#     def __call__(self, image, truth):
#         image = image.astype(np.float32).transpose((2, 0, 1)) #（channels, height, width）
#         image = torch.from_numpy(image).float()
#         truth = torch.from_numpy(truth).long()
#         return image, truth

class normalize:
    '''normalization with the given per-band max and min values'''
    def __init__(self, max, min):
        '''max, min: list, values corresponding to each band'''
        self.max, self.min = max, min
    def __call__(self, image, truth):
        for band in range(image.shape[-1]):
            image[:,:,band] = (image[:,:,band]-self.min[band])/(self.max[band]-self.min[band]+0.0001)
        return image, truth


def crop(image, truth, size=(256, 256)):
    ''' numpy-based
        des: randomly crop corresponding to specific size
        input image and truth are np.array
        input size: (height, width)'''
    start_h = random.randint(0, truth.shape[0]-size[0])
    start_w = random.randint(0, truth.shape[1]-size[1])
    image_crop = image[:,start_h:start_h+size[0], start_w:start_w+size[1]]
    truth_crop = truth[start_h:start_h+size[0], start_w:start_w+size[1]]
    return image_crop, truth_crop

class crop_scales:
    ''' numpy-based
        des: randomly crop corresponding to specific size
        input scales: tuple or list (high -> low)
        we design a multi-thread processsing
    '''
    def __init__(self, scales=(2048, 512, 256), threads=True):
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

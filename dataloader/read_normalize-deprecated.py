## author: xin luo
## create: 2021.9.9

from utils.geotif_io import readTiff
import numpy as np

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


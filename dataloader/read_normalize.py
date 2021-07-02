from utils.preprocess import normalize
from utils.geotif_io import readTiff
import numpy as np

s1_min = [-57.78, -70.37, -58.98, -68.47]  # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]

def read_normalize(paths_as, paths_des, paths_truth):
    ''' des: data reading and preprocessing
        input: 
            ascend image, descend image and truth image paths
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    for i in range(len(paths_as)):
        ## --- data reading
        ascend, ascend_info = readTiff(paths_as[i])
        descend, descend_info = readTiff(paths_des[i])
        truth, truth_info = readTiff(paths_truth[i])
        ## --- data normalization 
        scene = np.concatenate((ascend, descend), axis=-1)    
        scene, truth = normalize(max=s1_max, min=s1_min)(scene, truth)
        scene = scene.transpose(2,0,1)   # channel first
        scene_list.append(scene), truth_list.append(truth)
    return scene_list, truth_list


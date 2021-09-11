
from utils.geotif_io import readTiff
import torch
import numpy as np
from dataloader.preprocess import crop_scales

class scene_path_dset(torch.utils.data.Dataset):
    '''sentinel-1 ascending/descending image and the truth reading
        from the provided data path (read data from disk).
        time record: data(15 s1_image) read->4.7s model train -> 0.07 s 
    '''
    def __init__(self, paths_ascend, paths_descend, paths_truth, transforms):
        self.paths_ascend = paths_ascend
        self.paths_descend = paths_descend
        self.paths_truth = paths_truth
        self.transforms = transforms
    def __getitem__(self, index):
        '''load images and truths'''
        ascend, ascend_info = readTiff(self.paths_ascend[index])
        descend, descend_info = readTiff(self.paths_descend[index])
        truth, truth_info = readTiff(self.paths_truth[index])
        scene = np.concatenate((ascend, descend), axis=-1)
        '''pre-processing (e.g., random crop)'''
        patch_group, truth = crop_scales(scales=[2048,512,256], threads=False)(scene, truth)
        for transform in self.transforms:
            patch_group, truth = transform(patch_group, truth)
        return patch_group, torch.unsqueeze(truth,0)
    def __len__(self):
        return len(self.paths_truth)

class scene_tensor_dset(torch.utils.data.Dataset):
    '''sentinel-1 scene and the corresponding truth reading
        from the torch.Tensor: read data from memory.
    '''
    def __init__(self, scene_tensor_list, truth_tensor_list, transforms, scales=[2048,512,256]):
        '''input arrs_scene, arrs_truth are list'''
        self.scene_tensor_list = scene_tensor_list
        self.truth_tensor_list = truth_tensor_list
        self.transforms = transforms
    def __getitem__(self, index):
        '''load images and truths'''
        scene = self.scene_tensor_list[index]
        truth = self.truth_tensor_list[index]
        '''pre-processing (e.g., random crop)'''
        patch_group, truth = crop_scales(scales=[2048,512,256], threads=False)(scene, truth)
        for transform in self.transforms:
            patch_group, truth = transform(patch_group, truth)
        return patch_group, torch.unsqueeze(truth,0)
    def __len__(self):
        return len(self.truth_tensor_list)


class patch_path_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from data paths (in SSD)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->1.2 s model train -> 2.9 s 
    '''
    def __init__(self, paths_patch):
        self.paths_patch = paths_patch
    def __getitem__(self, index):
        '''load patches and truths'''
        patch_pair = torch.load(self.paths_patch[index])
        patch = patch_pair[0]
        truth = patch_pair[1]
        return patch, truth
    def __len__(self):
        return len(self.paths_patch)

class patch_tensor_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from memory (in RAM)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->0.7 s model train -> 2.9 s 
    '''
    def __init__(self, patch_pair_list):
        self.patch_pair_list = patch_pair_list
    def __getitem__(self, index):
        '''load patches and truths'''
        patch = self.patch_pair_list[index][0]
        truth = self.patch_pair_list[index][1]
        return patch, truth
    def __len__(self):
        return len(self.patch_pair_list)


if __name__ =='__main__':
    import glob
    paths_patch = sorted(glob.glob('/home/yons/Desktop/developer-luo/SWatNet/data/tra_patches/*'))
    tra_dset = patch_path_dset(paths_patch=paths_patch)
    print(tra_dset[0][0][0].shape)

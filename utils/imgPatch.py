'''
author: xin luo, 
create: 2021.3.19
des: 1. remote sensing image to patches and 2. patches to remote sensing image
'''        

import numpy as np

class imgPatch():
    def __init__(self, img, patch_size, edge_overlay):
        ''' edge_overlay = left overlay or, right overlay
        edge_overlay should be an even number. '''
        if edge_overlay % 2 != 0:
            raise ValueError('Argument edge_overlay should be an even number')
        self.edge_overlay = edge_overlay        
        self.patch_size = patch_size
        self.img = img[:,:,np.newaxis] if len(img.shape) == 2 else img
        self.img_row = img.shape[0]
        self.img_col = img.shape[1]
        self.img_patch_row = np.nan    # valid when call toPatch
        self.img_patch_col = np.nan
        self.start_list = []       #  

    def toPatch(self):
        '''
        des: 
            convert img to patches. 
        return: 
            patch_list, contains all generated patches.
            start_list, contains all start positions(row, col) of the generated patches. 
        '''
        patch_list = []
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step),
                                          (self.edge_overlay, patch_step), (0,0)), 'constant')
        self.img_patch_row = (img_expand.shape[0]-self.edge_overlay)//patch_step
        self.img_patch_col = (img_expand.shape[1]-self.edge_overlay)//patch_step
        for i in range(self.img_patch_row):
            for j in range(self.img_patch_col):
                patch_list.append(img_expand[i*patch_step:i*patch_step+self.patch_size,
                                                        j*patch_step:j*patch_step+self.patch_size, :])
                self.start_list.append([i*patch_step-self.edge_overlay, j*patch_step-self.edge_overlay])
        return patch_list

    def higher_patch_crop(self, higher_patch_size):
        '''
        des: 
            crop the higher-scale patch (centered by the given low-scale patch)
                (!!note: the toPatch() should be firstly called when use higher_patch_crop())
        input:
            img, np.array, the original image
            patch_size, int, the lower-scale patch size
            crop_size, int, the higher-scale patch size
            start_list, list, the start position (row,col) corresponding to the original image (generated by the toPatch function)
        return: 
            higher_patch_list, list, contains higher-scale patches corresponding to the lower-scale patches.
        '''
        higher_patch_list = []
        radius_bias = higher_patch_size//2-self.patch_size//2
        patch_step = self.patch_size - self.edge_overlay
        img_expand = np.pad(self.img, ((self.edge_overlay, patch_step), (self.edge_overlay, patch_step), (0,0)), 'constant')
        img_expand_higher = np.pad(img_expand, ((radius_bias, radius_bias), (radius_bias, radius_bias), (0,0)), 'constant')
        start_list_new = list(np.array(self.start_list)+self.edge_overlay+radius_bias)
        for start_i in start_list_new:
            higher_row_start, higher_col_start = start_i[0]-radius_bias, start_i[1]-radius_bias
            higher_patch = img_expand_higher[higher_row_start:higher_row_start+higher_patch_size, \
                                                            higher_col_start:higher_col_start+higher_patch_size,:]
            higher_patch_list.append(higher_patch)
        return higher_patch_list

    def toImage(self, patch_list):
        '''
        des: 
            merge patches into one image. 
            (!!note: the toPatch() should be firstly called when use toImage())
        '''
        patch_list = [patch[self.edge_overlay//2:-self.edge_overlay//2, self.edge_overlay//2:-self.edge_overlay//2,:]
                                                        for patch in patch_list]
        patch_list = [np.hstack((patch_list[i*self.img_patch_col:i*self.img_patch_col+self.img_patch_col]))
                                                        for i in range(self.img_patch_row)]
        img_array = np.vstack(patch_list)
        img_array = img_array[self.edge_overlay//2:self.img_row+self.edge_overlay//2, \
            self.edge_overlay//2:self.img_col+self.edge_overlay//2,:]
        return img_array


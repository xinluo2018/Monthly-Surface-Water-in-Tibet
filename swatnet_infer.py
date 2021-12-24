'''
author: xin luo
creat: 2021.9.8
des:
    perform surface water mapping by using pretrained watnet
    through funtional api and command line, respectively.
example:
    funtional api:
        water_map = swatnet_infer(rsimg) 
        !!note: the rsimg value should be in [0,1]
    command line: 
        python swatnet_infer.py -as data/tibet_tiles/s1_ascend/*.tif -des data/tibet_tiles/s1_descend/*.tif -s 100
        !!note: 1. the .tif format image should be the original value, or scale * original value
                2. the ascending image and descending image should be pair-wise, that is, the last number in 
                   the image name should be the same. e.g., ...ascending_001.tif, ...descending_001.tif
'''

import os
import numpy as np
import argparse
import torch
from utils.imgPatch import imgPatch
import scipy.ndimage
import gc

from utils.geotif_io import readTiff, writeTiff
from model.seg_model.model_scales_gate import unet_scales_gate

### ----------- default path of the pretrained watnet model ----------- 
path_swatnet_w = ['model/pretrained/apply_to_tibet/model_gscales_app_base_weights.pth']
path_swatnet_w_as = ['model/pretrained/apply_to_tibet/model_gscales_as_app_base_weights.pth']
path_swatnet_w_des = ['model/pretrained/apply_to_tibet/model_gscales_des_app_base_weights.pth']

### ------------- super parameters ------------- 
s1_min = [-57.78, -70.37, -58.98, -68.47]   # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]       # as-vv, as-vh, des-vv, des-vh


# ------------- Device --------------- #
# device = torch.device('cuda:0')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

## ------------- pre-defined functions -------------
### get arguments
def get_args():

    description = 'surface water mapping by using pretrained watnet'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-as', metavar='ifile_as', dest= 'ifile_as', type=str, 
        nargs='+', default=[None],
        help=('ascending file(s) to process (.tiff)'))

    parser.add_argument(
        '-des', metavar='ifile_des', dest= 'ifile_des', type=str, 
        nargs='+', default=[None],
        help=('descending file(s) to process (.tiff)'))

    parser.add_argument(
        '-m', metavar='model', dest='model', type=str, 
        nargs='+', default=path_swatnet_w, 
        help=('pretrained swatnet model (based on both ascending and descending images) weight(torch, .pth)'))

    parser.add_argument(
        '-m_as', metavar='model_as', dest='model_as', type=str, 
        nargs='+', default=path_swatnet_w_as, 
        help=('pretrained swatnet model (based on ascending image only) weight(torch, .pth)'))

    parser.add_argument(
        '-m_des', metavar='model_des', dest='model_des', type=str, 
        nargs='+', default=path_swatnet_w_des, 
        help=('pretrained swatnet model (based on descending image only) weight(torch, .pth)'))

    parser.add_argument(
        '-o', metavar='odir', dest='odir', type=str, nargs='+', 
        default=[None], help=('directory to write'))

    parser.add_argument(
        '-s', metavar='scale_DN', dest='scale_DN', type=int, nargs='+', 
        default=[None], help=('DN scale of the sentinel-1 image, if None, the scale can be regard as 1'))

    return parser.parse_args()

def normalize(s1_img, s1_min, s1_max):
    '''normalization'''
    s1_img_nor = s1_img.copy()    
    for band in range(s1_img.shape[-1]):
        s1_img_nor[:,:,band] = (s1_img_nor[:,:,band] - s1_min[band])/(s1_max[band]-s1_min[band]+0.0001)
    s1_img_nor = np.clip(s1_img_nor, 0., 1.) 
    s1_img_nor[np.isnan(s1_img_nor)]=0         # remove nan value
    return s1_img_nor


def img2patchin(img, scales = [256, 512, 2048], overlay=60):
    ratio_mid, ratio_high = scales[1]//scales[0], scales[2]//scales[0],
    imgPat_ins = imgPatch(img=img, patch_size=scales[0], edge_overlay = overlay)
    patch_low_list = imgPat_ins.toPatch()
    patch_mid_list = imgPat_ins.higher_patch_crop(higher_patch_size=scales[1])
    patch_high_list = imgPat_ins.higher_patch_crop(higher_patch_size=scales[2])
    '''---- Resize multi-scale patches to the same size'''
    patch_mid2low_list = [scipy.ndimage.zoom(input=patch, zoom=(1/ratio_mid, \
                                            1/ratio_mid, 1), order=0) for patch in patch_mid_list]
    patch_high2low_list = [scipy.ndimage.zoom(input=patch, zoom=(1/ratio_high, \
                                            1/ratio_high, 1), order=0) for patch in patch_high_list]
    return patch_low_list, patch_mid2low_list, patch_high2low_list, imgPat_ins

def model_pred(model, model_as, model_des, inputs, \
                    idx_both_valid, idx_as_valid_only, idx_des_valid_only):
    '''---- Obtain the prediction patches----'''
    pred_patch_list = [i for i in range(len(inputs))]  # initial list

    with torch.no_grad():
        for idx in idx_both_valid:
            if device:
                in_data = [inp.to(device) for inp in inputs[idx]]
            pred_patch = model(in_data)[0]      ### for multi-scale patch            
            pred_patch = pred_patch.cpu()   # convert from gpu to cpu
            pred_patch_list[idx] = pred_patch
        for idx in idx_as_valid_only:
            if device:
                in_data = [inp.to(device) for inp in inputs[idx]]
            pred_patch = model_as(in_data)[0]      ### for multi-scale patch            
            pred_patch = pred_patch.cpu()   # convert from gpu to cpu
            pred_patch_list[idx] = pred_patch
        for idx in idx_des_valid_only:
            if device:
                in_data = [inp.to(device) for inp in inputs[idx]]
            pred_patch = model_des(in_data)[0]      ### for multi-scale patch            
            pred_patch = pred_patch.cpu()   # convert from gpu to cpu
            pred_patch_list[idx] = pred_patch
    return pred_patch_list

if __name__ == '__main__':

    args = get_args()
    ifile_as = args.ifile_as
    ifile_des = args.ifile_des
    path_swatnet_w = args.model[0]
    path_swatnet_w_as = args.model_as[0]
    path_swatnet_w_des = args.model_des[0]
    odir = args.odir[0]
    scale_DN = args.scale_DN[0]

    ''' Obtain pair-wise ascending/descending files.'''
    io_files = []
    if odir: dir_wat = odir
    else: dir_wat = '/'.join(ifile_as[0].split('/')[:-2]) + '/s1_water'
    if not os.path.exists(dir_wat):
        os.makedirs(dir_wat)

    ### TODO: should be simplified
    if ifile_as[0] and ifile_des[0] is not None:
        for i_as in ifile_as:
            for i_des in ifile_des:
                if i_as.split('/')[-1].split('_')[-1] == i_des.split('/')[-1].split('_')[-1]:
                    name_wat = i_as.split('/')[-1].split('.')[0] + '_water.tif'
                    name_wat = name_wat.replace('s1as', 's1')
                    out_wat_path = dir_wat + '/' + name_wat
                    io_files.append((i_as, i_des, out_wat_path))   
                else:
                    continue
    elif ifile_as[0] is not None:
        for i_as in ifile_as:
            name_wat = i_as.split('/')[-1].split('.')[0] + '_water.tif'
            out_wat_path = dir_wat + '/' + name_wat
            io_files.append((i_as, out_wat_path))   
    elif ifile_des[0] is not None:
        for i_des in ifile_des:
            name_wat = i_des.split('/')[-1].split('.')[0] + '_water.tif'
            out_wat_path = dir_wat + '/' + name_wat
            io_files.append((i_des, out_wat_path))   
    io_files = sorted(io_files)
    print(io_files)

    '''Model loading'''
    model = unet_scales_gate(num_bands=4,num_classes=2,scale_high=2048,scale_mid=512,scale_low=256)
    model_as = unet_scales_gate(num_bands=2,num_classes=2,scale_high=2048,scale_mid=512,scale_low=256)
    model_des = unet_scales_gate(num_bands=2,num_classes=2,scale_high=2048,scale_mid=512,scale_low=256)
    model.load_state_dict(torch.load(path_swatnet_w))
    model_as.load_state_dict(torch.load(path_swatnet_w_as))
    model_des.load_state_dict(torch.load(path_swatnet_w_des))
    if device:
        model.to(device); model = model.eval()
        model_as.to(device);  model_as = model_as.eval()        # load on gpu
        model_des.to(device); model_des = model_des.eval()
    print('...pretrained models have been loaded')

    ''' Loop for each file '''
    for i in range(len(io_files)):
        ### ---- 1. preprocessing
        ### ---- 1.1 data reading
        print('--- 1. data reading...')
        if ifile_as[0] and ifile_des[0] is not None:
            print('Ascending image:', io_files[i][0])
            print('Descending image:', io_files[i][1])
            s1_ascend, s1_info = readTiff(path_in = io_files[i][0])
            s1_descend, s1_info = readTiff(path_in = io_files[i][1])
        elif ifile_as[0] is not None:
            print('Ascending image:', io_files[i][0])
            s1_ascend, s1_info = readTiff(path_in = io_files[i][0])
            s1_descend = np.zeros_like(s1_ascend)
        elif ifile_des[0] is not None:
            print('Descending image:', io_files[i][0])
            s1_descend, s1_info = readTiff(path_in = io_files[i][0])
            s1_ascend = np.zeros_like(s1_descend)
        dif_as_des = (abs(np.count_nonzero(s1_ascend) - \
                                np.count_nonzero(s1_descend)))/s1_descend.size

        ### --- 1.2 get normalized s1_image
        if scale_DN: 
            s1_ascend, s1_descend = np.float32(s1_ascend)/scale_DN, np.float32(s1_descend)/scale_DN
        s1_img = np.concatenate((s1_ascend, s1_descend), axis=2)
        s1_img[s1_img==0]=['nan']
        s1_img_nor = normalize(s1_img, s1_min, s1_max)
        del s1_img, s1_ascend, s1_descend; gc.collect()
        print('...Input image have been normalized')

        ### ---- 2. image to patch ----
        print('--- 2. Convert input image to patch list...')
        patch_low_list, patch_mid_list, patch_high_list, imgPat_ins = \
                            img2patchin(s1_img_nor, scales = [256, 512, 2048], overlay=60)
        num_patch = len(patch_low_list)
        del s1_img_nor; gc.collect()
        ### Three category: ascending only, descending only, and combined ascending and descending image
        if ifile_as[0] is None:
            idx_des_valid_only = [i for i in range(num_patch)]
            idx_as_valid_only = idx_both_valid = []
        elif ifile_des[0] is None:
            idx_as_valid_only = [i for i in range(num_patch)]
            idx_des_valid_only = idx_both_valid = []
        ### below: both ascending and descending image exist
        elif dif_as_des > 0.05:    
            miss_as_per = [1 - np.count_nonzero(patch[:,:,0:2])/patch[:,:,0:2].size for patch in patch_low_list]
            miss_des_per = [1 - np.count_nonzero(patch[:,:,2:4])/patch[:,:,2:4].size for patch in patch_low_list]
            idx_all = np.arange(len(patch_low_list))
            idx_as_valid = np.argwhere(np.array(miss_as_per) < 0.3).squeeze()
            idx_des_valid = np.argwhere(np.array(miss_des_per) < 0.3).squeeze()
            idx_both_valid = np.intersect1d(idx_as_valid, idx_des_valid)
            idx_as_valid_only = np.setdiff1d(idx_as_valid, idx_both_valid)
            idx_des_valid_only = np.setdiff1d(idx_all, np.union1d(idx_both_valid, idx_as_valid_only))
        else:
            idx_both_valid = [i for i in range(num_patch)]
            idx_as_valid_only = idx_des_valid_only = []
        #### update the patch_list
        for idx in idx_as_valid_only:
            patch_low_list[idx] = patch_low_list[idx][:,:,0:2]
            patch_mid_list[idx] = patch_mid_list[idx][:,:,0:2]
            patch_high_list[idx] = patch_high_list[idx][:,:,0:2]
        for idx in idx_des_valid_only:
            patch_low_list[idx] = patch_low_list[idx][:,:,2:4]
            patch_mid_list[idx] = patch_mid_list[idx][:,:,2:4]
            patch_high_list[idx] = patch_high_list[idx][:,:,2:4]

        ## formating data from 3d to 4d torch.tensor
        patch_high_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                            float() for patch in patch_high_list]
        patch_mid_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                            float() for patch in patch_mid_list]
        patch_low_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).\
                                                            float() for patch in patch_low_list]
        inputs = tuple(zip(patch_high_list_, patch_mid_list_, patch_low_list_))
        del patch_high_list, patch_mid_list, patch_low_list;
        del patch_high_list_, patch_mid_list_, patch_low_list_; gc.collect()
        print('Number of multi-scale patches:', len(inputs))
        print('...Input image have been converted to patches list')

        ### ---- 3. surface water mapping ----
        print('--- 3. surface water mapping using swatnet model...')
        pred_patch_list = model_pred(model, model_as, model_des, inputs, \
                                idx_both_valid, idx_as_valid_only, idx_des_valid_only)
        pred_patch_list = [np.squeeze(patch, axis = 0).permute(1, 2, 0) for patch in pred_patch_list]
        pro_map = imgPat_ins.toImage(pred_patch_list)
        wat_map = np.where(pro_map>0.5, 1, 0)
        print('...Water map have been produced')

        ### ---- 4. write out the water map -----
        print('--- 4. write out the result image...')
        writeTiff(im_data = wat_map.astype(np.uint8), 
                    im_geotrans = s1_info['geotrans'], 
                    im_geosrs = s1_info['geosrs'], 
                    path_out = io_files[i][-1])
        print('--- write out -->')
        print(io_files[i][-1])
        del wat_map; gc.collect()



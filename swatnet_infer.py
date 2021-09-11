'''
author: xin luo,
creat: 2021.9.8
des: 
    perform surface water mapping by using pretrained watnet
    through funtional api and command line, respectively.
example:
    funtional api:
        water_map = swatnet_infer(rsimg) 
        !!note: rsimg value: [0,1]
    command line: 
        python swatnet_infer.py data/tibet_tiles/s1_ascend/*.tif -des data/tibet_tiles/s1_descend/*.tif
    !!note: 
        the rsimg value should be in [0,1], 
        while the .tif format image should be the original value
'''

import os
import numpy as np
import argparse
import torch
from utils.imgPatch import imgPatch
import scipy.ndimage
import gc
from utils.get_s1pair_nor import get_s1pair_nor
from utils.geotif_io import readTiff, writeTiff
from model.seg_model.model_scales_gate import unet_scales_gate

## default path of the pretrained watnet model
path_swatnet_w = 'model/pretrained/model_scales_gate_weights.pth'
s1_min = [-57.78, -70.37, -58.98, -68.47]   # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]       # as-vv, as-vh, des-vv, des-vh

# ----------- Device --------------- #
# device = torch.device('cuda:0')
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def get_args():

    description = 'surface water mapping by using pretrained watnet'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        'ifile_as', metavar='ifile_as', type=str, nargs='+',
        help=('ascending file(s) to process (.tiff)'))

    parser.add_argument(
        '-des', metavar='ifile_des', dest= 'ifile_des', type=str, nargs='+',
        help=('descending file(s) to process (.tiff)'))

    parser.add_argument(
        '-m', metavar='model', dest='model', type=str, 
        nargs='+', default=path_swatnet_w, 
        help=('pretrained swatnet model weight(torch, .pth)'))

    parser.add_argument(
        '-o', metavar='odir', dest='odir', type=str, nargs='+', 
        default=None, help=('directory to write'))

    return parser.parse_args()


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


def model_pred(model, inputs):
    '''---- Obtain the prediction patches----'''
    pred_patch_list = []
    for idx in range(len(inputs)):
        if device:
            in_data = [inp.to(device) for inp in inputs[idx]]
        with torch.no_grad():
            pred_patch = model(in_data)      ### for multi-scale patch
            if isinstance(pred_patch,tuple):
                pred_patch = pred_patch[0]
        pred_patch = pred_patch.cpu()   # convert from gpu to cpu
        pred_patch_list.append(pred_patch)
    return pred_patch_list


def swatnet_infer(s1_img, model):
    
    ''' des: surface water mapping by using pretrained watnet
        arg:
            img: np.array, sentinel-1 backscattering values(!!data value: 0-1): ; 
                 consist of 4 band (sascending VV, VH and descending VV, Vh)
            model: the loaded pytorched model.
        retrun:
            water_map: np.array.
    '''

    ### ---- 1. Convert remote sensing image to multi-scale patches ----
    print('--- convert image to multi-scale pathes input...')
    patch_low_list, patch_mid_list, patch_high_list, imgPat_ins = \
                            img2patchin(s1_img, scales = [256, 512, 2048], overlay=60)
    del s1_img
    gc.collect()

    ## formating data from 3d to 4d torch.tensor
    patch_high_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).float() \
                                                                    for patch in patch_high_list]
    patch_mid_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).float() \
                                                                    for patch in patch_mid_list]
    patch_low_list_ = [torch.from_numpy(patch.transpose(2,0,1)[np.newaxis,:]).float() \
                                                                    for patch in patch_low_list]
    inputs = tuple(zip(patch_high_list_, patch_mid_list_, patch_low_list_)) 
    print('number of multi-scale patches:', len(inputs))
    del patch_high_list_, patch_mid_list_, patch_low_list_
    gc.collect()

    ### ---- 2. prediction by pretrained model -----
    print('--- surface water mapping using swatnet model...')
    pred_patch_list = model_pred(model=model, inputs=inputs)
    del inputs
    gc.collect()

    ### ---- 3. Convert the patches to image ----
    print('--- comvert patch result to image result...')
    pred_patch_list = [np.squeeze(patch, axis = 0).permute(1, 2, 0) for patch in pred_patch_list]
    pro_map = imgPat_ins.toImage(pred_patch_list)
    wat_map = np.where(pro_map>0.5, 1, 0)

    return wat_map

if __name__ == '__main__':

    args = get_args()
    ifile_as = args.ifile_as
    ifile_des = args.ifile_des
    path_model_w = args.model
    odir = args.odir

    ''' Obtain pair-wise ascending/descending files.'''
    io_files = []
    if odir:
        dir_wat = odir
    else:
        dir_wat = '/'.join(ifile_as[0].split('/')[:-2]) + '/s1_water'
    for i_as in ifile_as:
        for i_des in ifile_des:
            if i_as.split('/')[-1].split('_')[-1] == i_des.split('/')[-1].split('_')[-1]:
                if not os.path.exists(dir_wat):
                    os.makedirs(dir_wat)
                name_wat = i_as.split('/')[-1].split('.')[0] + '_water.tif'
                name_wat = name_wat.replace('s1as', 's1')
                o_water = dir_wat + '/' + name_wat
                io_files.append((i_as, i_des, o_water))   
            else:
                continue
    io_files = sorted(io_files)

    '''Model loading'''
    model_name= 'model_scales_gate'
    # model = unet_scales(num_bands=4, num_classes=2, \
    #                         scale_high=2048, scale_mid=512, scale_low=256)
    model = unet_scales_gate(num_bands=4, num_classes=2, \
                            scale_high=2048, scale_mid=512, scale_low=256)
    model.load_state_dict(torch.load(path_model_w))
    if device:
        model.to(device)    # load on gpu
    model = model.eval()

    print(io_files)

    ''' Loop for each file '''
    for i in range(len(io_files)):
        
        ### ---- 1. preprocessing
        ### ---- 1.1 data reading
        print('--- data reading...')
        print(io_files[i][0])
        print(io_files[i][1])
        s1_ascend, s1_ascend_info = readTiff(path_in = io_files[i][0])
        s1_descend, _ = readTiff(path_in = io_files[i][1])
        ### --- 1.2 get normalized s1_image
        s1_img_nor = get_s1pair_nor(s1_as=s1_ascend, s1_des=s1_descend)
        print('image shape:', s1_img_nor.shape)
        del s1_ascend, s1_descend
        gc.collect()

        ### ---- 2. surface water mapping ----
        print('--- surface water mapping using swatnet model...')
        wat_map = swatnet_infer(s1_img=s1_img_nor, model=model)
        del s1_img_nor
        gc.collect()

        ### ---- 3. write out the water map -----
        print('--- write out the result image...')
        writeTiff(im_data = wat_map.astype(np.uint8), 
                    im_geotrans = s1_ascend_info['geotrans'], 
                    im_geosrs = s1_ascend_info['geosrs'], 
                    path_out = io_files[i][2])
        print('--- write out -->')
        print(io_files[i][2])
        del wat_map
        gc.collect()



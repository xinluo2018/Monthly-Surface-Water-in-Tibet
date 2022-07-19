## author: xin luo
## creat: 2022.4.3, modify: 2022.4.15
## des: model traing with the dset(traset or full dset)
## !!!note: the input dataset not to be normalized. the normalization process has been 
##          added to this script.

import sys
sys.path.append("/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet")
from notebooks import config
import numpy as np
import torch
import time
import pandas as pd
import glob
import argparse
from utils.metric import oa_binary, miou_binary
from dataloader.preprocess import read_normalize
from model.seg_model.model_scales_in import unet_scales
from model.seg_model.model_scales_gate import unet_scales_gate
from model.seg_model.deeplabv3_plus import deeplabv3plus
from dataloader.parallel_loader import threads_scene_dset
from dataloader.loader import patch_tensor_dset

def get_args():
  
    """ Get command-line arguments. """
    parser = argparse.ArgumentParser(
            description='model training')
    parser.add_argument(
            '--model_type', type=str, nargs='+',
            help='model type of the traiend model, gscales/scales/single',
            default=['gscales']
            )
    parser.add_argument(
            '--dataset', type=str, nargs='+',
            help='model type of the traiend model, dset/traset',
            default=['dset']
            )
    parser.add_argument(
            '--s1_orbit', type=str, nargs='+',
            help='model type of the traiend model, as/des/as_des',
            default=['as_des']
            )
    parser.add_argument(
            '--model_name', type=str, nargs='+',
            help='model name of the trained model, e.g., model_1',
            default=['model_1']
            )
    parser.add_argument(
            '--num_epoch', type=int, nargs='+',
            help='number of the training epoch, e.g., 300',
            default=[300]
            )

    return parser.parse_args()


'''------train step------'''
def train_step(model, loss_fn, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y.float())
    loss.backward()
    optimizer.step()
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------validation step------'''
def val_step(model, loss_fn, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = loss_fn(pred, y.float())
    miou = miou_binary(pred=pred, truth=y)
    oa = oa_binary(pred=pred, truth=y)
    return loss, miou, oa

'''------ train loops ------'''
def train_loops(model, loss_fn, optimizer, tra_loader, val_loader, epoches, lr_scheduler=None):
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    tra_loss_loops, tra_miou_loops = [], []
    val_loss_loops, val_miou_loops = [], []
    for epoch in range(epoches):
        start = time.time()
        tra_loss, val_loss = 0, 0
        tra_miou, val_miou = 0, 0
        tra_oa, val_oa = 0, 0

        '''----- 1. train the model -----'''
        for x_batch, y_batch in tra_loader:
            x_batch, y_batch = [batch.to(device) for batch in x_batch], y_batch.to(device)
            if model.name == 'deeplabv3plus':
              x_batch = x_batch[2]      # !!!note: x_batch[2] for single-scale model
            y_batch = config.label_smooth(y_batch) 
            loss, miou, oa = train_step(model=model, loss_fn=loss_fn, 
                                        optimizer=optimizer, x=x_batch, y=y_batch)
            tra_loss += loss.item()
            tra_miou += miou.item()
            tra_oa += oa.item()
        if lr_scheduler:
          lr_scheduler.step(tra_loss)         # if using ReduceLROnPlateau
          # lr_scheduler.step()          # if using StepLR scheduler.

        '''----- 2. validate the model -----'''
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = [batch.to(device).to(dtype=torch.float32) for batch in x_batch], y_batch.to(device)    
            if model.name == 'deeplabv3plus':
              x_batch = x_batch[2]          #!!!note: x_batch[2] for single-scale model
            loss, miou, oa = val_step(model=model, loss_fn=loss_fn, 
                                                      x=x_batch, y=y_batch)
            val_loss += loss.item()
            val_miou += miou.item()
            val_oa += oa.item()

        '''------ 3. print accuracy ------'''
        tra_loss = tra_loss/size_tra_loader
        val_loss = val_loss/size_val_loader
        tra_miou = tra_miou/size_tra_loader
        val_miou = val_miou/size_val_loader
        tra_oa = tra_oa/size_tra_loader
        val_oa = val_oa/size_val_loader
        tra_loss_loops.append(tra_loss), tra_miou_loops.append(tra_miou)
        val_loss_loops.append(val_loss), val_miou_loops.append(val_miou)

        format = 'Ep{}: Tra-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f}, Val-> Loss:{:.3f},Oa:{:.3f},Miou:{:.3f},Time:{:.1f}s'
        print(format.format(epoch+1, tra_loss, tra_oa, tra_miou, val_loss, val_oa, val_miou, time.time()-start))

    metrics = {'tra_loss':tra_loss_loops, 'tra_miou':tra_miou_loops, 'val_loss': val_loss_loops, 'val_miou': val_miou_loops}
    return metrics


if __name__ == '__main__':

    ## Configuration
    args = get_args()
    model_type = args.model_type[0]
    dataset = args.dataset[0]
    s1_orbit = args.s1_orbit[0]
    model_name = args.model_name[0]
    num_epoch = args.num_epoch[0]

    device = torch.device('cuda:1')
    torch.manual_seed(999)   # make the trianing replicable
    if s1_orbit == 'as': num_bands = 2; id_band_start, id_band_end = 0, 2
    if s1_orbit == 'des': num_bands = 2; id_band_start, id_band_end = 2, 4
    elif s1_orbit == 'as_des': num_bands = 4; id_band_start, id_band_end = 0, 4
    if model_type == 'gscales':
      model = unet_scales_gate(num_bands=num_bands, num_classes=2,\
                          scale_high=2048, scale_mid=512, scale_low=256).to(device)
    elif model_type == 'scales':
      model = unet_scales(num_bands=num_bands, num_classes=2, \
                          scale_high=2048, scale_mid=512, scale_low=256).to(device)
    elif model_type == 'single':
      model = deeplabv3plus(num_bands=num_bands, num_classes=2).to(device)
    print('Model type and name:', model_type + '/' + dataset + '/' + s1_orbit + '/' + model_name)

    ## Data paths 
    ### the whole dataset.
    paths_as = sorted(glob.glob(config.dir_as + '/*pad*.tif'))  ## ascending scenes
    paths_des = sorted(glob.glob(config.dir_des+'/*pad*.tif'))  ## descending scenes
    paths_truth = sorted(glob.glob(config.dir_truth+'/*pad*.tif'))   ## truth water 
    ### training part of the dataset.
    paths_tra_as, paths_tra_des, paths_tra_truth = [], [], []
    for tra_id in config.tra_ids:     ## select training scenes
      as_name = 'scene'+tra_id+'_s1as_pad.tif'
      des_name = 'scene'+tra_id+'_s1des_pad.tif'
      truth_name = 'scene'+tra_id+'_wat_truth_pad.tif'
      paths_tra_as.append(config.dir_as + '/' + as_name); 
      paths_tra_des.append(config.dir_des + '/' + des_name);
      paths_tra_truth.append(config.dir_truth + '/' + truth_name)
    ### validation part of the dataset (patch format)
    paths_patch_val = sorted(glob.glob(config.dir_patch_val+'/*'))   ## validatation patches

    '''--------- 1. Data loading --------'''
    if dataset == 'traset':   ## use the training scenes for training.
      paths_as_, paths_des_, paths_truth_ = paths_tra_as, paths_tra_des, paths_tra_truth
    elif dataset == 'dset':                     ## use the whole scenes for training.
      paths_as_, paths_des_, paths_truth_ = paths_as, paths_des, paths_truth

    '''----- 1.1 training data loading (from scenes path) '''
    tra_scenes, tra_truths = read_normalize(paths_as=paths_as_, paths_des=paths_des_, \
                                paths_truth=paths_truth_, max_bands=config.s1_max, min_bands=config.s1_min)
    # ### !!!!extract either ascending or descending image.
    tra_scenes = [s[id_band_start: id_band_end] for s in tra_scenes]   ## [0:2] -> ascending; [2:4] -> descending
    ''' ----- 1.2. Training data loading and auto augmentation'''
    tra_dset = threads_scene_dset(scene_list = tra_scenes, \
                                  truth_list = tra_truths, 
                                  transforms=config.transforms_tra, 
                                  num_thread=30)          ##  num_thread(30) patches per scene.
    print('size of training data:  ', tra_dset.__len__())

    ''' ----- 1.3. validation data loading (validation patches) ------ '''
    patch_list_val = [torch.load(path) for path in paths_patch_val]
    ## !!!extract either ascending or descending image for validation
    ## [id_band_start:id_band_end]; [0:2] -> ascending; [2:4] -> descending
    for i in range(len(patch_list_val)):
       for j in range(len(patch_list_val[0][0])):
          patch_list_val[i][0][j] = patch_list_val[i][0][j][id_band_start:id_band_end]   
    val_dset = patch_tensor_dset(patch_pair_list = patch_list_val)
    print('size of validation data:', val_dset.__len__())

    tra_loader = torch.utils.data.DataLoader(tra_dset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=16)

    ''' -------- 2. Model loading and training strategy ------- '''
    model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                  mode='min', factor=0.6, patience=20)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)

    ''' -------- 3. Model training for loops ------- '''
    metrics = train_loops(model=model,  
                        loss_fn=config.loss_bce, 
                        optimizer=optimizer,  
                        tra_loader=tra_loader,  
                        val_loader=val_loader,  
                        epoches=num_epoch,  
                        lr_scheduler=lr_scheduler,
                        )

    ''' -------- 4. trained model and accuracy metric saving  ------- '''
    # model saving
    path_weights = config.root_proj+'/model/trained_model/'+model_type+'/'+dataset+'/'+s1_orbit+'/'+model_name+'_weights.pth'
    torch.save(model.state_dict(), path_weights)
    print('Model weights are saved to --> ', path_weights)
    ## metrics saving
    path_metrics = config.root_proj+'/model/trained_model/'+model_type+'/'+dataset+'/'+s1_orbit+'/'+model_name+'_metrics.csv'
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(path_metrics, index=False, sep=',')
    metrics_df = pd.read_csv(path_metrics)
    print('Training metrics are saved to --> ', path_metrics)


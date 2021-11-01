## author: xin luo
## create: 2021.10.26
## Metrics for Semantic Segmentation

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


## batch-based, torch-based
def oa_binary(pred, truth):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa

def miou_binary(pred, truth):
    ''' des: calculate miou (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter/(area_union+0.0000001)
    miou = iou.mean()
    return miou


## image-based
def acc_matrix(cla_map,  sam_pixel=None, truth_map=None, id_label=None):
    ''' 
    Arguments: 
        cla_map: classification result of the full image
        truth_map: truth image (either truth_map or sam_pixel should be given)
        sam_pixel: array(num_samples,3), col 1,2,3 are the row,col and label.            
        id_label: 0,1,2..., the target class for calculating producer's or user's accuracy.
    Return: 
        the overall accuracy and confusion matrix
    '''
    if sam_pixel is not None:
        sam_result = []
        sam_label = sam_pixel[:,2]
        num_cla = sam_label.max()+1
        labels = list(range(num_cla))
        for i in range(sam_label.shape[0]):
            sam_result.append(cla_map[sam_pixel[i,0], sam_pixel[i,1]])            
        sam_result = np.array(sam_result)
    if truth_map is not None:
        sam_label = truth_map.flatten()  
        sam_result = cla_map.flatten()
        num_cla = sam_label.max()+1
        labels = list(range(num_cla))
    acc_oa = np.around(accuracy_score(sam_label, sam_result), 4)
    confus_mat = confusion_matrix(sam_label, sam_result, labels=labels)
    if id_label is not None:
        acc_prod=np.around(confus_mat[id_label, id_label]/confus_mat[id_label,:].sum(), 4)
        acc_user=np.around(confus_mat[id_label, id_label]/confus_mat[:,id_label].sum(), 4)
        return acc_oa, acc_prod, acc_user, confus_mat
    else:
        return acc_oa, confus_mat


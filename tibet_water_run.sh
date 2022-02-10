#!/bin/bash

## ----- configure data
## ascending data
as_path_1=/WD-myBook/tibet-water/tibet-202005/s1_ascend/*2??.tif
as_path_2=/WD-myBook/tibet-water/tibet-202005/s1_ascend/*3??.tif
## descending data
des_path=/WD-myBook/tibet-water/tibet-202005/s1_descend/*.tif

## ----- configure models
## surface water mapping using deep learning models
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_app_base_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_as_app_base_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_des_app_base_weights.pth
python swatnet_infer.py -as $as_path_1 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_2 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des



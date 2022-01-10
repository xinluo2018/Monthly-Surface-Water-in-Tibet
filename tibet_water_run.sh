#!/bin/bash

as_path_1=/myDrive/tibet-water/tibet-202012/s1_ascend/*_26?.tif
as_path_2=/myDrive/tibet-water/tibet-202012/s1_ascend/*_27?.tif
as_path_3=/myDrive/tibet-water/tibet-202012/s1_ascend/*_28?.tif
as_path_4=/myDrive/tibet-water/tibet-202012/s1_ascend/*_29?.tif
as_path_5=/myDrive/tibet-water/tibet-202012/s1_ascend/*_3??.tif
des_path=/myDrive/tibet-water/tibet-202012/s1_descend/*.tif

model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_app_base_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_as_app_base_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_des_app_base_weights.pth

python swatnet_infer.py -as $as_path_1 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_2 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_3 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_4 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_5 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des

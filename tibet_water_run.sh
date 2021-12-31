#!/bin/bash

as_path=/myDrive/tibet-water/tibet-202002/s1_ascend/*.tif
des_path_1=/myDrive/tibet-water/tibet-202002/s1_descend/*.tif
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_app_base_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_as_app_base_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_des_app_base_weights.pth

python swatnet_infer.py -as $as_path -des $des_path_1 -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path -des $des_path_2 -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path -des $des_path_3 -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path -des $des_path_4 -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path -des $des_path_5 -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path -des $des_path_6 -s 100 -m $model -m_as $model_as -m_des $model_des


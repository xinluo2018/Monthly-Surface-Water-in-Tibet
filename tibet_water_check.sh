#!/bin/bash

as_path_1=/WD-myBook/tibet-water/tibet-202008/s1_ascend/*218.tif
as_path_2=/WD-myBook/tibet-water/tibet-202008/s1_ascend/*219.tif
# as_path_3=/WD-myBook/tibet-water/tibet-202008/s1_ascend/*236.tif
as_path_4=/WD-myBook/tibet-water/tibet-202008/s1_ascend/*239.tif

## descending data
des_path=/WD-myBook/tibet-water/tibet-202008/s1_descend/*.tif

## run
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_base_5_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_as_base_1_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_des_base_2_weights.pth
python swatnet_infer.py -as $as_path_1 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_2 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path_3 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_4 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des


#!/bin/bash

## ----- configure data
## ascending data
as_path_1=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*185.tif
as_path_2=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*186.tif
as_path_3=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*187.tif
as_path_4=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*188.tif
as_path_5=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*189.tif
as_path_6=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*19?.tif
as_path_7=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*2??.tif
as_path_8=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*3??.tif
# as_path_9=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*179.tif

## descending data
des_path=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif

## ----- configure models
## surface water mapping using deep learning models
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_app_base_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_as_app_base_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/model_gscales_des_app_base_weights.pth
python swatnet_infer.py -as $as_path_1 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_2 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_3 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_4 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_5 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_6 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_7 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
python swatnet_infer.py -as $as_path_8 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des
# python swatnet_infer.py -as $as_path_9 -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des


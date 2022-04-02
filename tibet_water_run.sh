#!/bin/bash

## ----- 1. Configure models
## surface water mapping using deep learning models
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_base_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_as_base_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/pretrained/apply_to_tibet/gscales_app_des_base_weights.pth


## ----- 2. Configure data
## ascending data
as_path_1=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*_178.tif
# as_path_2=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*_19?.tif
# as_path_3=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*_2??.tif
# as_path_4=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*_3??.tif
# as_path_5=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*.tif
# as_path_6=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*.tif
# as_path_7=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*.tif
# as_path_8=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*.tif
# as_path_9=/WD-myBook/tibet-water/tibet-202003/s1_ascend/*.tif
# as_path_10=/WD-myBook/tibet-water/tibet-20203/s1_ascend/*.tif
# as_path_11=/WD-myBook/tibet-water/tibet-20203/s1_ascend/*.tif
# as_path_12=/WD-myBook/tibet-water/tibet-20203/s1_ascend/*.tif
## descending data
des_path_1=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_2=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_3=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_4=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_5=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_6=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_7=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_8=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_9=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_10=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_11=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif
# des_path_12=/WD-myBook/tibet-water/tibet-202003/s1_descend/*.tif

## ----- 3. Run the model
python swatnet_infer.py -as $as_path_1 -des $des_path_1 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_2 -des $des_path_1 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_3 -des $des_path_1 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_4 -des $des_path_1 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_5 -des $des_path_5 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_6 -des $des_path_6 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_7 -des $des_path_7 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_8 -des $des_path_8 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_9 -des $des_path_9 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_10 -des $des_path_10 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_11 -des $des_path_11 -s 100 -m $model -m_as $model_as -m_des $model_des
#python swatnet_infer.py -as $as_path_12 -des $des_path_12 -s 100 -m $model -m_as $model_as -m_des $model_des




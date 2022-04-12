#!/bin/bash

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

### ---- seleted tiles
## ascending image
paths_as='/WD-myBook/tibet-water/tibet-202008-del/s1_ascend/tibet_s1as_202008_tile_218.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_ascend/tibet_s1as_202008_tile_219.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_ascend/tibet_s1as_202008_tile_236.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_ascend/tibet_s1as_202008_tile_239.tif
            '
## descending image
paths_des='/WD-myBook/tibet-water/tibet-202008-del/s1_descend/tibet_s1des_202008_tile_218.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_descend/tibet_s1des_202008_tile_219.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_descend/tibet_s1des_202008_tile_236.tif
            /WD-myBook/tibet-water/tibet-202008-del/s1_descend/tibet_s1des_202008_tile_239.tif
            '

## ----- Run the model
## model path
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/as_des/model_1_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/as/model_5_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/des/model_1_weights.pth
dir_wat_chek=/WD-myBook/tibet-water/tibet-202008-del/s1_water_check

## running
python script/swatnet_infer.py -as $paths_as -des $paths_des -s 100 -m $model -m_as $model_as -m_des $model_des -o $dir_wat_chek


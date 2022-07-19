#!/bin/bash

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

### ---- seleted tiles
## ascending image
paths_as='/WD-myBook/tibet-water/tibet-202001/s1-ascend/tibet_s1as_202001_tile_270.tif
            /WD-myBook/tibet-water/tibet-202001/s1-ascend/tibet_s1as_202001_tile_271.tif
            /WD-myBook/tibet-water/tibet-202002/s1-ascend/tibet_s1as_202002_tile_114.tif
            /WD-myBook/tibet-water/tibet-202002/s1-ascend/tibet_s1as_202002_tile_267.tif
            /WD-myBook/tibet-water/tibet-202003/s1-ascend/tibet_s1as_202003_tile_140.tif
            /WD-myBook/tibet-water/tibet-202003/s1-ascend/tibet_s1as_202003_tile_155.tif
            /WD-myBook/tibet-water/tibet-202004/s1-ascend/tibet_s1as_202004_tile_133.tif
            /WD-myBook/tibet-water/tibet-202004/s1-ascend/tibet_s1as_202004_tile_270.tif
            /WD-myBook/tibet-water/tibet-202005/s1-ascend/tibet_s1as_202005_tile_146.tif
            /WD-myBook/tibet-water/tibet-202005/s1-ascend/tibet_s1as_202005_tile_276.tif
            /WD-myBook/tibet-water/tibet-202006/s1-ascend/tibet_s1as_202006_tile_096.tif
            /WD-myBook/tibet-water/tibet-202006/s1-ascend/tibet_s1as_202006_tile_162.tif
            /WD-myBook/tibet-water/tibet-202007/s1-ascend/tibet_s1as_202007_tile_276.tif
            /WD-myBook/tibet-water/tibet-202008/s1-ascend/tibet_s1as_202008_tile_218.tif
            /WD-myBook/tibet-water/tibet-202008/s1-ascend/tibet_s1as_202008_tile_236.tif
            /WD-myBook/tibet-water/tibet-202008/s1-ascend/tibet_s1as_202008_tile_269.tif
            /WD-myBook/tibet-water/tibet-202009/s1-ascend/tibet_s1as_202009_tile_213.tif
            /WD-myBook/tibet-water/tibet-202009/s1-ascend/tibet_s1as_202009_tile_241.tif
            /WD-myBook/tibet-water/tibet-202010/s1-ascend/tibet_s1as_202010_tile_085.tif
            /WD-myBook/tibet-water/tibet-202010/s1-ascend/tibet_s1as_202010_tile_258.tif
            /WD-myBook/tibet-water/tibet-202011/s1-ascend/tibet_s1as_202011_tile_087.tif
            /WD-myBook/tibet-water/tibet-202011/s1-ascend/tibet_s1as_202011_tile_114.tif
            /WD-myBook/tibet-water/tibet-202012/s1-ascend/tibet_s1as_202012_tile_023.tif
            /WD-myBook/tibet-water/tibet-202012/s1-ascend/tibet_s1as_202012_tile_091.tif
            '
## descending image
paths_des='/WD-myBook/tibet-water/tibet-202001/s1-descend/tibet_s1des_202001_tile_270.tif
            /WD-myBook/tibet-water/tibet-202001/s1-descend/tibet_s1des_202001_tile_271.tif
            /WD-myBook/tibet-water/tibet-202002/s1-descend/tibet_s1des_202002_tile_114.tif
            /WD-myBook/tibet-water/tibet-202002/s1-descend/tibet_s1des_202002_tile_267.tif
            /WD-myBook/tibet-water/tibet-202003/s1-descend/tibet_s1des_202003_tile_140.tif
            /WD-myBook/tibet-water/tibet-202003/s1-descend/tibet_s1des_202003_tile_155.tif
            /WD-myBook/tibet-water/tibet-202004/s1-descend/tibet_s1des_202004_tile_133.tif
            /WD-myBook/tibet-water/tibet-202004/s1-descend/tibet_s1des_202004_tile_270.tif
            /WD-myBook/tibet-water/tibet-202005/s1-descend/tibet_s1des_202005_tile_146.tif
            /WD-myBook/tibet-water/tibet-202005/s1-descend/tibet_s1des_202005_tile_276.tif
            /WD-myBook/tibet-water/tibet-202006/s1-descend/tibet_s1des_202006_tile_096.tif
            /WD-myBook/tibet-water/tibet-202006/s1-descend/tibet_s1des_202006_tile_162.tif
            /WD-myBook/tibet-water/tibet-202007/s1-descend/tibet_s1des_202007_tile_276.tif
            /WD-myBook/tibet-water/tibet-202008/s1-descend/tibet_s1des_202008_tile_218.tif
            /WD-myBook/tibet-water/tibet-202008/s1-descend/tibet_s1des_202008_tile_236.tif
            /WD-myBook/tibet-water/tibet-202008/s1-descend/tibet_s1des_202008_tile_269.tif
            /WD-myBook/tibet-water/tibet-202009/s1-descend/tibet_s1des_202009_tile_213.tif
            /WD-myBook/tibet-water/tibet-202009/s1-descend/tibet_s1des_202009_tile_241.tif
            /WD-myBook/tibet-water/tibet-202010/s1-descend/tibet_s1des_202010_tile_085.tif
            /WD-myBook/tibet-water/tibet-202010/s1-descend/tibet_s1des_202010_tile_258.tif
            /WD-myBook/tibet-water/tibet-202011/s1-descend/tibet_s1des_202011_tile_087.tif
            /WD-myBook/tibet-water/tibet-202011/s1-descend/tibet_s1des_202011_tile_114.tif
            /WD-myBook/tibet-water/tibet-202012/s1-descend/tibet_s1des_202012_tile_023.tif
            /WD-myBook/tibet-water/tibet-202012/s1-descend/tibet_s1des_202012_tile_091.tif
            '


## ----- Run the model

## model path (base)
model=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/as_des/model_1_weights.pth
model_as=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/as/model_5_weights.pth
model_des=/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet/model/trained_model/gscales/dset/des/model_1_weights.pth
dir_tiles_check=/WD-myBook/tibet-water/tibet-tiles-check




## running
python script/swatnet_infer.py -as $paths_as -des $paths_des -s 100 -m $model -m_as $model_as -m_des $model_des -o $dir_tiles_check


#!/bin/bash
## create: 2022.4.9
## des: surface water inference and accurace evaluation for the test sites by using 
##      1) the trained gscales, scales, and single-scale models. 
##      2) the trained ascending only-based and descending only-based gscales models.

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

# IDS_MODEL='1 2 3 4 5 6 7 8 9 10'
IDS_MODEL='1'

for ID_MODEL in $IDS_MODEL

do 
  ## ----- 1. Configure models
  ## surface water mapping using deep learning models
  model_gscales_as=model/trained_model/gscales/traset/as/model_${ID_MODEL}_weights.pth
  model_gscales_des=model/trained_model/gscales/traset/des/model_${ID_MODEL}_weights.pth
  model_gscales=model/trained_model/gscales/traset/as_des/model_${ID_MODEL}_weights.pth
  model_scales=model/trained_model/scales/traset/as_des/model_${ID_MODEL}_weights.pth
  model_single=model/trained_model/single/traset/as_des/model_${ID_MODEL}_weights.pth
  VALSITES='01 03 08 11 15 19 24 32 36 39'

  for I_VAL in $VALSITES
  do
    echo 'number of validation site: ' $I_VAL
    path_valsite_as=data/dset/s1_ascend/scene${I_VAL}_s1as.tif 
    path_valsite_des=data/dset/s1_descend/scene${I_VAL}_s1des.tif

    ## ---- model_gscales output
    dir_gscales_out=data/dset/valsite_wat_infer/gscales/as_des/model_$ID_MODEL
    python script/swatnet_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $model_gscales -o $dir_gscales_out

    ## ---- model_gscales_as output
    dir_gscales_as_out=data/dset/valsite_wat_infer/gscales/as/model_$ID_MODEL
    python script/swatnet_infer.py -as $path_valsite_as -s 1 -m_as $model_gscales_as -o $dir_gscales_as_out

    # ## ---- model_gscales_des output
    dir_gscales_des_out=data/dset/valsite_wat_infer/gscales/des/model_$ID_MODEL
    python script/swatnet_infer.py -des $path_valsite_des -s 1 -m_des $model_gscales_des -o $dir_gscales_des_out

    ## ---- model_scales output
    dir_scales_out=data/dset/valsite_wat_infer/scales/as_des/model_$ID_MODEL
    python script/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $model_scales -o $dir_scales_out

    # ---- model_single output
    dir_single_out=data/dset/valsite_wat_infer/single/as_des/model_$ID_MODEL
    python script/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $model_single -o $dir_single_out

  done

done


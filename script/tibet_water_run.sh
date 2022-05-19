#!/bin/bash
## author: xin luo
## create: 2022.5.10
## des: Surface water mapping for the tibet region by using the trained model.

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

## ----- 1. Models
## surface water mapping using deep learning models
model=model/trained_model/gscales/dset/as_des/model_22_weights.pth
model_as=model/trained_model/gscales/dset/as/model_25_weights.pth
model_des=model/trained_model/gscales/dset/des/model_21_weights.pth

dates='202001 202002 202003 202004 202005 202006 202007 202008 202009 202010 202011 202012'

for date in $dates
do
  echo 'Date:' $date
  # ----- 2. Data
  # --- ascending data
  as_path=/WD-myBook/tibet-water/tibet-$date/s1-ascend/*.tif
  des_path=/WD-myBook/tibet-water/tibet-$date/s1-descend/*.tif
  echo 'as_path:' $as_path
  echo 'des_path:' $des_path

  ## ----- 3. Run the model
  python script/swatnet_infer.py -as $as_path -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des

done



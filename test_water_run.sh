#!/bin/bash
# single scale
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*01.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*02.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*03.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*04.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*05.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*06.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*07.tif -s 1 -m model/pretrained/model_single_test/model_single_test_1_weights.pth -o data/dset/s1_water_test_single

# # #### multiple scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*01.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*02.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*03.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*04.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*05.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*06.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales
# python model_infer_tmp.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*07.tif -s 1 -m model/pretrained/model_scales_test/model_scales_test_4_weights.pth -o data/dset/s1_water_test_scales

# ### gated multiple scales
## best: as_des -> fine -> 95.7.
# python swatnet_infer.py -as data/dset/s1_ascend/*01.tif -des data/dset/s1_descend/*01.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*02.tif -des data/dset/s1_descend/*02.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*03.tif -des data/dset/s1_descend/*03.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*04.tif -des data/dset/s1_descend/*04.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*05.tif -des data/dset/s1_descend/*05.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*06.tif -des data/dset/s1_descend/*06.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py -as data/dset/s1_ascend/*07.tif -des data/dset/s1_descend/*07.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_${i_model_as_des}_weights.pth -o data/dset/s1_water_test_gscales

# ## ascending image-based
# ## best: as -> 6 -> 87.3; 
# i_model_as=6
# python swatnet_infer.py -as data/dset/s1_ascend/*01.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*02.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*03.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*04.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*05.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*06.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*07.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_${i_model_as}_weights.pth -o data/dset/s1_water_test_as

# descending image-based
## des -> fine -> 93.4
i_model_des=fine
python swatnet_infer.py -des data/dset/s1_descend/*01.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*02.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*03.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*04.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*05.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*06.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -des data/dset/s1_descend/*07.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_${i_model_des}_weights.pth -o data/dset/s1_water_test_des


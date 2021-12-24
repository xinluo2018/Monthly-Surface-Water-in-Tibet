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

# # ### gated multiple scales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*01.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*02.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*03.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*04.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*05.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*06.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales
# python swatnet_infer.py data/dset/s1_ascend/*.tif -des data/dset/s1_descend/*07.tif -s 1 -m model/pretrained/model_gscales_test/model_gscales_test_9_weights.pth -o data/dset/s1_water_test_gscales

# # # # ascending image-based
# python swatnet_infer.py -as data/dset/s1_ascend/*01.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*02.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*03.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*04.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*05.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*06.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as
# python swatnet_infer.py -as data/dset/s1_ascend/*07.tif -s 1 -m_as model/pretrained/gscales_as_test/gscales_as_test_6_weights.pth -o data/dset/s1_water_test_as

# descending image-based
python swatnet_infer.py -as data/dset/s1_descend/*01.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*02.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*03.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*04.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*05.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*06.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des
python swatnet_infer.py -as data/dset/s1_descend/*07.tif -s 1 -m_des model/pretrained/gscales_des_test/gscales_des_test_6_weights.pth -o data/dset/s1_water_test_des

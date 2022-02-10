#!/bin/bash

## remove buffer region
path_imgs=/WD-myBook/tibet-water/tibet-202005/s1_water/* 
path_tiles=data/tibet/tibet_tiles_vec/tibet_tiles.gpkg
python utils/buffer_remove.py -imgs $path_imgs -tiles $path_tiles

## mosaic
paths_in=/WD-myBook/tibet-water/tibet-202005/s1_water/*_water_debuf*
path_out=/myDrive/tibet-water/tibet-result/tibet_water_202005_debuf_mosaic.tif
gdal_merge.py -init 0 -co COMPRESS=LZW -o $path_out $paths_in


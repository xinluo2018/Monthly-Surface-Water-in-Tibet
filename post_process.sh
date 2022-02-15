#!/bin/bash
#### des: 1. remove buffer region; 2. tiles mosaic.

date=202003
## remove buffer region
path_imgs=/WD-myBook/tibet-water/tibet-${date}/s1_water/* 
path_tiles=data/tibet/tibet_tiles_vec/tibet_tiles.gpkg
python utils/buffer_remove.py -imgs $path_imgs -tiles $path_tiles

## mosaic
paths_in=/WD-myBook/tibet-water/tibet-${date}/s1_water/*_water_debuf*
path_out=/WD-myBook/tibet-water/tibet-result/tibet_water_${date}_debuf_mosaic.tif
gdal_merge.py -init 0 -co COMPRESS=LZW -o $path_out $paths_in


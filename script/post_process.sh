#!/bin/bash
## author: xin luo
## create: 2022.5.14
## des: 1. remove buffer region; 2. tiles mosaic.

cd /home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet

dates='202001 202002 202003 202004 202005 202006 202007 202008 202009 202010 202011 202012'
# date=202001
for date in $dates
do
  # ## remove buffer region
  path_imgs=/WD-myBook/tibet-water/tibet-${date}/s1-water/* 
  path_tiles=data/tibet/tibet_tiles_vec/tibet_tiles.gpkg
  python utils/buffer_remove.py -imgs $path_imgs -tiles $path_tiles

  ## mosaic
  paths_in=/WD-myBook/tibet-water/tibet-${date}/s1-water/*_water_debuf*
  path_out=/WD-myBook/tibet-water/tibet-result/tibet_water_${date}_debuf_mosaic.tif
  echo 'Run the images mosaic...'
  gdal_merge.py -init 0 -co COMPRESS=LZW -o $path_out $paths_in

done
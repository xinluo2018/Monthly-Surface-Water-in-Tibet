#!/bin/bash

path_jrc='data/tibet/jrc_water/tibet_water_jrc_100m.tif' 
path_jrc_down='data/tibet/jrc_water/tibet_water_jrc_1000m.tif' 
gdal_translate -outsize 10% 10% -r average $path_jrc $path_jrc_down



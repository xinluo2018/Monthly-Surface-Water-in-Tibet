#! /bin/bash

## parse the input arguments
while getopts 'i:o:h' opt;
do 
  case $opt in
  h)
    echo 'Run the images mosaic'
    echo 'Basic usage: mosaic.sh -i Path_in -o Path_out'
    echo '-i Paths_in   :  paths of the images to be mosaicked, the wildcard is allowed; if 
                           Path_in contains wildcard or multiple files, the multiple files 
                           should be put in " " ' 
    echo '-o Path_out  :  paths of the out mosaicked image'
    exit 0
    ;;
  i) paths_in=$OPTARG
    ;;
  o) path_out=$OPTARG
    ;;
  esac
done

## setting the necessary parameters
in_file_1=$(echo $paths_in | cut -d ' ' -f 1)   # the first file to be mosaicked.
files_dir=$(dirname $in_file_1)
if [[ $path_out = "" ]]; 
then
  path_out=$files_dir'/mosaic.tif'
fi
echo 'in_paths:' $paths_in
echo 'out_path:' $path_out

path_out_tmp=$(echo $path_out | cut -d . -f1)'_tmp.tif' 
i_mosaic=0

## one-by-one images mosaic through a for loop.
for path_in in $paths_in
do 
  if [ $i_mosaic == 0 ];then
    echo 'mosaic image ->:' $path_in
    gdal_merge.py -o $path_out $path_in $path_in -init 0 -co COMPRESS=LZW
  else
    mv $path_out $path_out_tmp    # rename
    echo 'mosaic image ->:' $path_in
    gdal_merge.py -o $path_out $path_out_tmp $path_in -init 0 -co COMPRESS=LZW
  fi
  i_mosaic=$(expr $i_mosaic + 1)
done
rm $path_out_tmp
echo $i_mosaic "images to be mosaicked"


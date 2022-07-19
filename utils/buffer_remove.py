'''
author: xin luo
creat: 2022.1.27
des:
    remove buffer for the images by using tile (.shp/gpkg) file.
example:
    path_imgs=/myDrive-2/tibet-water/tibet-202012/s1_water/* 
    path_tiles=data/tibet/tibet_tiles_vec/tibet_tiles.gpkg
    python utils/buffer_remove.py -imgs $path_imgs -tiles $path_tiles
'''

import os
import numpy as np
import argparse
import geopandas as gpd
from geotif_io import readTiff, writeTiff

### get arguments
def get_args():

    description = 'buffer removing for 321 tiles image'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-imgs', metavar='path_imgs', dest= 'path_imgs', type=str, 
        nargs='+', default=[None],
        help=('input paths of images (.tiff) that to be removing buffer'))
    parser.add_argument(
        '-tiles', metavar='path_tiles', dest= 'path_tiles', type=str, 
        nargs='+', default=[None],
        help=('input tiles file (.shp)'))
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    path_imgs = args.path_imgs
    path_tiles = args.path_tiles[0]    # .shp/.gpkg file
    print(path_tiles)
    tiles_gdf = gpd.read_file(path_tiles)

    for path_img in path_imgs:
        print('process image - > ', path_img)
        tile_id = path_img.split('.')[0][-9:-6]
        path_img_debuf = path_img.split('.')[0] + '_debuf.tif'
        img, img_info = readTiff(path_in=path_img)
        ## extract tile extent
        idx, = np.where(tiles_gdf['tile_id'].values == tile_id)
        tile_region = tiles_gdf.loc[int(idx)]['geometry'].bounds
        left_up = [tile_region[0],tile_region[3]]
        right_down = [tile_region[2],tile_region[1]]
        extent = str(left_up[0])+' '+str(left_up[1])+' '+str(right_down[0])+' '+str(right_down[1]) 
        ## remove the buffer region
        command = 'gdal_translate -projwin ' + extent +' -co COMPRESS=LZW ' + path_img + ' ' + path_img_debuf
        print(os.popen(command).read())
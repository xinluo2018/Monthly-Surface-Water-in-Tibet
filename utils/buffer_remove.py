## author: xin luo
## create: 2021.9.13
## modify: 2021.10.13

from utils.geotif_io import readTiff, writeTiff
from utils.transform_xy import coor2coor, geo2imagexy


def buffer_remove(path_in, path_out, extent, save=True):
    '''
    des: subset image with the given extent
    arg:
        img: np.array
        extent: list, [left, right, down, up], wgs84 coords
    imgtrans:
        gdal transform of image.
    '''
    img, img_info = readTiff(path_in=path_in)
    imgsrs, imgtrans = img_info['geosrs'], img_info['geotrans']
    if imgsrs != '4326':
        extent[0], extent[2] = coor2coor(srs_from='4326', srs_to=imgsrs, x=extent[0], y=extent[2])
        extent[1], extent[3] = coor2coor(srs_from='4326', srs_to=imgsrs, x=extent[1], y=extent[3])
    row_min, col_min = geo2imagexy(x=extent[0], y=extent[3], gdal_trans=imgtrans)
    row_max, col_max = geo2imagexy(x=extent[1], y=extent[2], gdal_trans=imgtrans)
    img_subs = img[row_min:row_max, col_min:col_max]
    imgtrans_subs = (extent[0], imgtrans[1], imgtrans[2], extent[3], imgtrans[4], imgtrans[5])
    if save:
        writeTiff(im_data=img_subs, im_geotrans=imgtrans_subs, im_geosrs=imgsrs, path_out=path_out)

    return img_subs, imgtrans_subs





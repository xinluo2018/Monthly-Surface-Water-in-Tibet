## author: xin luo
## create: 2021.11.27
## des: crop one image to specific image size. usually used for alignment to 
##      another image.
## usage: 

import numpy as np
from osgeo import gdal

def crop_to_extent(path_img, extent, size_target=None, path_save=None):

    '''
    crop image to given extent/size.
    arg:
        image: the image to be croped; np.array().
        extent: extent to which image should be croped;
                list/tuple,(xmin,xmax,ymin,ymax). 
        size_target: size to which image should be croped 
              list/tuple, (row, col)
    return: 
        img_croped: the croped image, np.array()
    '''

    rs_data=gdal.Open(path_img)
    dtype_id = rs_data.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(dtype_id)
    if 'int8' in dtype_name:
        datatype = gdal.GDT_Byte
    elif 'int16' in dtype_name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    geotrans = rs_data.GetGeoTransform()
    dx, dy = geotrans[1], geotrans[5]
    nbands = rs_data.RasterCount
    proj_wkt = rs_data.GetProjection()
    NDV = rs_data.GetRasterBand(1).GetNoDataValue()
    xmin, xmax, ymin, ymax = extent

    if size_target is None:
        npix_x = int(np.round((xmax - xmin) / float(dx)))  # new col
        npix_y = int(np.round((ymin - ymax) / float(dy)))  # new row
        dx = (xmax - xmin) / float(npix_x)
        dy = (ymin - ymax) / float(npix_y)
    else:
        npix_x = size_target[1]
        npix_y = size_target[0]
        dx = (xmax - xmin) / float(size_target[1])  # new resolution
        dy = (ymin - ymax) / float(size_target[0])

    if path_save is None:
        drv = gdal.GetDriverByName('MEM')
        dest = drv.Create('', npix_x, npix_y, nbands, datatype)
    else: 
        driver = gdal.GetDriverByName("GTiff")
        dest = driver.Create(path_save, npix_x, npix_y, nbands, datatype)
        
    dest.SetProjection(proj_wkt)
    newgeotrans = (xmin, dx, 0.0, ymax, 0.0, dy)
    dest.SetGeoTransform(newgeotrans)
    if NDV is not None:
        for i in range(nbands):
            dest.GetRasterBand(i+1).SetNoDataValue(NDV)
            dest.GetRasterBand(i+1).Fill(NDV)
    else:
        for i in range(nbands):
            dest.GetRasterBand(i+1).Fill(0)

    gdal.ReprojectImage(rs_data, dest, proj_wkt, proj_wkt, gdal.GRA_Bilinear)
    out_array = dest.ReadAsArray(0, 0,  npix_x,  npix_y)

    if NDV is not None:
        out_array = np.ma.masked_where(out_array == NDV, out_array)

    if nbands > 1:
        return np.transpose(out_array, (1, 2, 0))  # 
    else:
        return out_array
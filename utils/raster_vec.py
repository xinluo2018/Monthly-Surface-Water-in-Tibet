## author: xin luo, 
## creat: 2021.7.15
## modify: 2021.11.27

import fiona
import rasterio
import numpy as np
import rasterio.features
from osgeo import ogr, gdal
from shapely.geometry import shape, mapping
from shapely.geometry.multipolygon import MultiPolygon

def raster2vec(raster_path, output_path, dn_values):
    ''' des: Read input band with Rasterio
        input:
            raster_path, output_path: raster and ouput vector path
            dn_values: list, consist of the raster value to be vectorization
        return:
            vector (gpkg format) written to the given path.
    '''
    # Read input band with Rasterio
    with rasterio.open(raster_path) as src:
        crs = src.crs
        src_band = src.read(1)
        shapes = list(rasterio.features.shapes(src_band, transform=src.transform))
    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'pixelvalue': 'int'}
        }
    ## write out vector
    with fiona.open(output_path, 'w', 'GPKG', shp_schema, crs) as shp:
        for pixel_value in dn_values:
            polygons = [shape(geom) for geom, value in shapes
                        if value == pixel_value]
            multipolygon = MultiPolygon(polygons)
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {'pixelvalue': int(pixel_value)}
            })

def vec2mask(path_vec, path_raster, path_save=None):
    """
    des: generate/save mask file using the vector file(e.g.,.shp,.gpkg).
    author: jinhua zhang, create: 2021.3.13, modify by luo: 2021.11.27
    input: 
        path_vec, path_raster, path_save: str
    retrun: 
        mask, np.array.
        a .tiff file written to the given path
    """
    raster, vec = gdal.Open(path_raster, gdal.GA_ReadOnly), ogr.Open(path_vec)
    x_res = raster.RasterXSize
    y_res = raster.RasterYSize
    layer = vec.GetLayer()
    if path_save is None:
        drv = gdal.GetDriverByName('MEM')
        targetData = drv.Create('', x_res, y_res, 1, gdal.GDT_Byte)
    else: 
        driver = gdal.GetDriverByName("GTiff")
        targetData = driver.Create(path_save, x_res, y_res, 1, gdal.GDT_Byte)

    targetData.SetGeoTransform(raster.GetGeoTransform())
    targetData.SetProjection(raster.GetProjection())
    band = targetData.GetRasterBand(1)
    NoData_value = -9999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(targetData, [1], layer, burn_values=[1])
    mask = targetData.ReadAsArray(0, 0, x_res, y_res)
    mask = np.where(mask>0, 1, 0)
    return mask

    
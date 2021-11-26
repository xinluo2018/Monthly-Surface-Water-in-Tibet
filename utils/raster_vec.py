## author: xin luo, creat: 2021.7.15

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
    ## writh out vector
    with fiona.open(output_path, 'w', 'GPKG', shp_schema, crs) as shp:
        for pixel_value in dn_values:
            polygons = [shape(geom) for geom, value in shapes
                        if value == pixel_value]
            multipolygon = MultiPolygon(polygons)
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {'pixelvalue': int(pixel_value)}
            })

def vec2mask(vec_path, raster_path, output_path):
    """
    des: generate/save mask file using the vector file(e.g.,.shp,.gpkg).
    author: jinhua zhang, create: 2021.3.13, modify: 2021.7.28
    input: 
        vec_path, raster_path, output_path: str
    retrun: 
        mask, np.array.
        a .tiff file written to the given path
    """
    raster, shp = gdal.Open(raster_path, gdal.GA_ReadOnly), ogr.Open(vec_path)
    x_res = raster.RasterXSize
    y_res = raster.RasterYSize
    layer = shp.GetLayer()
    targetDataset = gdal.GetDriverByName('GTiff').Create(output_path, x_res, y_res, 1, gdal.GDT_Byte)
    targetDataset.SetGeoTransform(raster.GetGeoTransform())
    targetDataset.SetProjection(raster.GetProjection())
    band = targetDataset.GetRasterBand(1)
    NoData_value = -9999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1], layer, burn_values=[1])
    mask = targetDataset.ReadAsArray(0, 0, x_res, y_res)
    mask = np.where(mask>0, 1, 0)
    return mask
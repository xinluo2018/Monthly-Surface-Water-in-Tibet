///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.18
// Description: show the sentinel-2/landsat-8 image in the tibetan region
///////////////////////////////////////////////////////


// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb_bound = area_tb.geometry().bounds();
// date of the collected data
var start_time = '2020-01-01'
var end_time = '2020-12-31'


/// Landsat 8 images
var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
var image_collection_l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
  .filter(ee.Filter.lt('CLOUD_COVER_LAND', 15))
  .filterBounds(area_tb)
  .filterDate(start_time, end_time);
print('image_collection_l8 --> ', image_collection_l8)
var image_l8 = image_collection_l8.median()

/// Sentinel-2 image
var sen2Coll = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(area_tb)
                  .filterDate(start_time, end_time)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',15));
print('sen2Coll --> ', sen2Coll)
var sen2 = sen2Coll.median()

var empty = ee.Image().byte();
var tb_outline = empty.paint({
      featureCollection: area_tb, color: 1, width: 3});
Map.setCenter(86.0, 32.0, 4);
Map.addLayer(image_l8, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'Landsat 8')
Map.addLayer(sen2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2');
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline');


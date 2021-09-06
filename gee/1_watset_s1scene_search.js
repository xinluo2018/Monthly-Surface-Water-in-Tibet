//////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.19
// Description: This code assist us to select the training scenes (sen1 ascending and descending
// pair and sen2 image),that the sen1 image and sen2 image should be located in the 
// same region, and are collected at the similar date.
//////////////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb_bound = area_tb.geometry().bounds();

//// scene regions
// dates:
// scene01_sen2:2019.9.6, scene01_sen1:2019.9.8&2019.9.3
// scene02_sen2:2020.9.2, scene01_sen1:2020.8.29&2020.8.29
// scene03_sen2:2020.10.11, scene01_sen1:2020.10.12&2020.10.7
// scene04_sen2:2020.6.24, scene01_sen1:2020.6.19&2020.6.19
// scene05_sen2:2020.9.1, scene05_sen1:2020.8.25&2020.8.26
// scene06_sen2:2020.9.14, scene06_sen1:2020.9.16&2020.9.12
// scene07_sen2:2020.8.31, scene07_sen1:2020.8.26&2020.9.3
// scene08_sen2:2020.10.25, scene08_sen1:2020.10.16&2020.10.21
// scene09_sen2:2020.8.18, scene09_sen1:2020.8.11&2020.8.18
// scene10_sen2:2019.9.27, scene10_sen1:2019.9.24&2019.9.25
// scene11_sen2:2020.9.23, scene11_sen1:2020.9.19&2020.9.20
// scene12_sen2:2019.11.14, scene12_sen1:2019.11.15&2019.11.10
// scene13_sen2:2019.8.11, scene13_sen1:2019.8.18&2019.8.18
// scene14_sen2:2019.8.13, scene14_sen1:2019.8.7&2019.8.2
// scene15_sen2:2019.8.16, scene15_sen1:2019.8.13&2019.8.13
// scene16_sen2:2020.8.31, scene16_sen1:2020.9.3&2020.9.3
// scene17_sen2:2020.2.13, scene17_sen1:2020.2.17&2020.2.12
// scene18_sen2:2020.9.7, scene18_sen1:2020.9.5&2020.9.11
// scene19_sen2:2019.9.26, scene19_sen1:2019.9.27&2019.10.4
// scene20_sen2:2019.7.7, scene20_sen1:2019.7.2&2019.7.8

// var scene01_region = ee.Geometry.Rectangle(82.26, 33.82, 82.68, 34.14)
// var scene02_region = ee.Geometry.Rectangle(97.93, 34.62, 98.35, 34.93) 
// var scene03_region = ee.Geometry.Rectangle(90.61, 28.84, 91.01, 29.15)
// var scene04_region = ee.Geometry.Rectangle(86.67, 35.48, 87.09, 35.78) 
// var scene05_region = ee.Geometry.Rectangle(89.78, 33.82, 90.14, 34.11) 
// var scene06_region = ee.Geometry.Rectangle(87.33, 31.77, 87.76, 32.09)
// var scene07_region = ee.Geometry.Rectangle(81.14,30.43, 81.51, 30.73)
// var scene08_region = ee.Geometry.Rectangle(97.43, 38.13, 97.82, 38.44)
// var scene09_region = ee.Geometry.Rectangle(86.35,33.77,86.77,34.10)
// var scene10_region = ee.Geometry.Rectangle(90.95, 35.48, 91.35, 35.78)
// var scene11_region = ee.Geometry.Rectangle(79.47,33.36,79.86,33.66)
// var scene12_region = ee.Geometry.Rectangle(98.70,29.38,99.08,29.68)
// var scene13_region = ee.Geometry.Rectangle(100.45,35.82,100.85,36.12)
// var scene14_region = ee.Geometry.Rectangle(91.27,31.82,91.67,32.13)
// var scene15_region = ee.Geometry.Rectangle(102.07,33.65,102.46,33.95)
// var scene16_region = ee.Geometry.Rectangle(96.08,32.95,96.47,33.26)
// var scene17_region = ee.Geometry.Rectangle(94.42,29.29,94.80,29.58)
// var scene18_region = ee.Geometry.Rectangle(74.69,38.55,75.09,38.84)
// var scene19_region = ee.Geometry.Rectangle(83.85,31.29,84.22,31.58)
var scene20_region = ee.Geometry.Rectangle(88.94,37.39,89.36,37.69)

var scene_region = scene20_region
print('scene region area:', scene_region.area())

var start_time = '2020-9-18';
var end_time = '2020-10-2';

/// Sentinel-2 image for reference
var Bands_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
// Note: before 2019, sen2 image should be searched with the TOA data.
var sen2Coll = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(scene_region)
                  .filterDate('2020-9-2', '2020-10-22')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
                  .sort('CLOUDY_PIXEL_PERCENTAGE');

// Ensure the sen2 image fully contain the training region.
var condition = function(image){
    var footprint = ee.Geometry(image.get('system:footprint'))
    var condition = ee.Geometry.Polygon(footprint.coordinates()).contains(scene_region)
    return ee.Algorithms.If(condition, 
                            image.set('data', 'true'), 
                            ee.Image([]).set('data', 'false'))};

var sen2Coll_Sel = sen2Coll.map(condition).filterMetadata('data', 'equals','true')

////Sentinel-1 images (ascending and descending obit).
var Bands_S1 = ['VV', 'VH']

var ascendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filterBounds(scene_region)
    .filterDate(start_time, end_time);
var ascendCol_Sel = ascendCol.map(condition).filterMetadata('data', 'equals','true')

var descendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filterBounds(scene_region)
    .filterDate(start_time, end_time);
var descendCol_Sel = descendCol.map(condition).filterMetadata('data', 'equals','true')


var ascendImg = ascendCol_Sel.first().select(Bands_S1).float()
var descendImg = descendCol_Sel.first().select(Bands_S1).float()

var sen2Img = sen2Coll_Sel.first().select(Bands_S2)

print('sen2Coll:', sen2Coll, sen2Coll.size().getInfo())
print('sen2Coll_Sel:', sen2Coll_Sel, sen2Coll_Sel.size().getInfo())
print('sen2Img_Sel:', sen2Img, sen2Img.date())
print('ascendCol_Sel:', ascendCol_Sel, ascendCol_Sel.size().getInfo())
print('descendCol_Sel', descendCol_Sel, descendCol_Sel.size().getInfo())
print('acendImg_Sel:', ascendImg, ascendImg.date())
print('descendImg_Sel:', descendImg, descendImg.date())

var empty = ee.Image().byte();
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});

Map.addLayer(ascendImg.select('VV'), {min:-30, max:0}, 'ascend image')
Map.addLayer(descendImg.select('VV'), {min:-30, max:0}, 'descend image')
Map.addLayer(sen2Img, {bands: ['B4', 'B3', 'B2'], min:0, max:5000}, 'sen2 image')
Map.addLayer(scene_region, {color:'green'}, 'training region 2')
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')



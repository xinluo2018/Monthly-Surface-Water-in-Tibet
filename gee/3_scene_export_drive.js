/////////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.20
// Description: This code export the selected image to the google drive.
/////////////////////////////////////////////////////////////////

var Bands_S1 = ['VV', 'VH']
var Bands_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

// region and source images
// tra1
var scene01_region = ee.Geometry.Rectangle(82.26, 33.82, 82.68, 34.14)
var scene01_s2_source = 'COPERNICUS/S2_SR/20190906T050659_20190906T051806_T44SPC'
var scene01_s1_as_source = 'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190908T123207_20190908T123232_028930_0347AE_1771'
var scene01_s1_des_source = 'COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20190903T003331_20190903T003356_017866_0219F6_E5B3'

////////////////////////////////////////////////////////////////////////////////
//// training regions loading
var region = scene01_region
var s2_img = ee.Image(scene01_s2_source).clip(region).select(Bands_S2)
var s1_as = ee.Image(scene01_s1_as_source).clip(region).select(Bands_S1)
var s1_des = ee.Image(scene01_s1_des_source).clip(region).select(Bands_S1)
print(s2_img, s1_as, s1_des)

Map.addLayer(s2_img, {bands:['B4','B3','B2'], max:3000, min:0}, 'sen2 image')
Map.addLayer(s1_as, {bands:['VV','VH','VV'], max:0, min:-30}, 'sen1 ascend')
Map.addLayer(s1_des, {bands:['VV','VH','VV'], max:0, min:-30}, 'sen1 descend')

// // Export to Google Drive
// //// Note: the dimensions should be specified to the same among the three images.
// Export.image.toDrive({
//   image: s2_img,
//   description: 'scene01_s2_img',
//   folder: 'Sar_WaterExt_Data',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: region
//   });

// Export.image.toDrive({
//   image: s1_as,
//   description: 'scene01_s1_ascend',
//   folder: 'Sar_WaterExt_Data',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: region
//   });

// Export.image.toDrive({
//   image: s1_des,
//   description: 'scene01_s1_descend',
//   folder: 'Sar_WaterExt_Data',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: region
//   });

///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.11.13, modify: 2021.08.27
// Description: select the sentinel-1 images that fully cover the tibetan plateau
///////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb_bound = area_tb.geometry().bounds();
// date of the collected data
var start_time = '2020-08-01'
var end_time = '2020-08-30'

////**********************************************************////
// -------- data search ------- ///
////**********************************************************////

//// ---- 1. sentinel-1 images (ascending and descending obit).
var ascendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filterBounds(area_tb)
    .filterDate(start_time, end_time);

var descendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filterBounds(area_tb)
    .filterDate(start_time, end_time)
    .sort('system:time_start');

print('ascend images:', ascendCol.size(), ascendCol)
print('descend images:', descendCol.size(), descendCol)


//// ---- 2. remove the images which contains masked value (null data).
var valid_percent_func = function(image){
    var region_sel = image.geometry()
                          .centroid({'maxError': 1})
                          .buffer({'distance': 15000})
                          .bounds();
    var image_count = image.reduceRegion({
                           // count the non masked values
                          reducer: ee.Reducer.count(),  
                          maxPixels: 1e10, 
                          geometry: region_sel,
                          scale: 10,
                          })
    var img_count = image_count.getNumber('VV')
                          .min(image_count.getNumber('VH'))
                          .divide(ee.Number(7000000))
    return image.set('valid_per', img_count)  
    }


var ascendCol = ascendCol.map(valid_percent_func)
                         .filterMetadata('valid_per', 'greater_than', 0.90)
var descendCol= descendCol.map(valid_percent_func)
                         .filterMetadata('valid_per','greater_than', 0.90)

print('filtered ascend images:', ascendCol.size(), ascendCol)
print('filtered descend images:', descendCol.size(), descendCol)

// ---- 3. remove the redundant image which at the same location
var area_intersec_thre = ee.Number(ascendCol.first().geometry().area()).multiply(0.8);
var region_exclusive_func = function(image, imgCol_list){
    var imgCol_ls = ee.List(imgCol_list);
    var imgCol_ls_geo = ee.ImageCollection(imgCol_ls).union().geometry();
    var image_geo = image.geometry();
    var image_inter_area = imgCol_ls_geo.intersection(image_geo).area();
    return imgCol_ls.add(image.set('area_intersec', image_inter_area))
}

var ascendCol_sel = ee.ImageCollection(
                      ee.List(ascendCol.iterate(region_exclusive_func, [])))
                      .filterMetadata('area_intersec', 'less_than', area_intersec_thre)
                        
var descendCol_sel = ee.ImageCollection(
                      ee.List(descendCol.iterate(region_exclusive_func, [])))
                      .filterMetadata('area_intersec', 'less_than', area_intersec_thre)
              
print('selected ascend images:', ascendCol_sel.size(), ascendCol_sel)
print('selected descend images:', descendCol_sel.size(), descendCol_sel)


// ------ 4. add image footprint and id
function generate_img_id(img){
    var img_geo = img.geometry();
    var img_id = ee.String('COPERNICUS/S1_GRD/').cat(img.id())
    var fea = ee.Feature(img_geo);
    return fea.set('img_id', img_id);}

var as_fp_id = ascendCol_sel.map(generate_img_id);
var des_fp_id = descendCol_sel.map(generate_img_id);
print('as_fp_id:', as_fp_id)


// //**********************************************************////
// -------- image footprint and id export ------- ///
// //**********************************************************////
// Export an ee.FeatureCollection as an Earth Engine asset.
// Export.table.toAsset({
//   collection: as_fp_id,
//   description:'tibet_s1_202008_as',
//   assetId: 'tibet_s1_202008_as',
// });

// Export.table.toAsset({
//   collection: des_fp_id,
//   description:'tibet_s1_202008_des',
//   assetId: 'tibet_s1_202008_des',
// });


////**********************************************************////
// -------- visualization ------- ///
////**********************************************************////
var empty = ee.Image().byte();
//// outline visualization of the study area.
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});
//// outline visualization of the image footprint.
var as_fprint_outline = empty.paint({
    featureCollection: as_fp_id, color: 1, width: 2});
var des_fprint_outline = empty.paint({
    featureCollection: des_fp_id, color: 1, width: 2});


Map.setCenter(86.0, 32.0, 4);
Map.addLayer(ascendCol_sel.select('VH'), {min: -50, max: 1}, 'ascendVH');
Map.addLayer(ascendCol_sel.select('VV'), {min: -50, max: 1}, 'ascendVV');
Map.addLayer(descendCol_sel.select('VH'), {min: -50, max: 1}, 'descendVH');
Map.addLayer(descendCol_sel.select('VV'), {min: -50, max: 1}, 'descendVV');
Map.addLayer(as_fprint_outline, {palette: 'FF00FF'}, 'as_footprint');
Map.addLayer(des_fprint_outline, {palette: '000000'}, 'des_footprint');
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline');



///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2021.12.24
// Description: export jcr surface water map
///////////////////////////////////////////////////////


// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb_bound = area_tb.geometry().bounds();

var tibet_water_jrc = ee.Image('JRC/GSW1_3/GlobalSurfaceWater')
                        .select(['occurrence'])
                        .clip(area_tb);

var visualization = {
    bands: ['occurrence'],
    min: 0.0,
    max: 100.0,
    palette: ['ffffff', 'ffbbbb', '0000ff']
};

var empty = ee.Image().byte();
//// outline visualization of the study area.
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});
    
Map.setCenter(86.0, 32.0, 4);
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline');
Map.addLayer(tibet_water_jrc, visualization, 'Occurrence');

// export jcr water image
// Export.image.toDrive({
//     image: tibet_water_jrc,
//     description: 'tibet_water_jrc',
//     folder: 'tibet_sar_data',
//     scale: 100, 
//     maxPixels: 1e10,
//     fileFormat: 'GeoTIFF',
//     region: area_tb.geometry(),
//     });

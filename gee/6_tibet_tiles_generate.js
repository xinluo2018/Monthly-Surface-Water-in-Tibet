///////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2021.08.30, modify: 2021.09.05
// Description: split the tibet region into multiple tiles.
///////////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb = area_tb.geometry()
var area_tb_bound = area_tb.bounds();

print('area_tb_buf', area_tb_bound)

// *******************************************************************
// ------ 1. generate tiles
// *******************************************************************

// var generate_grid = function(xmin, ymin, xmax, ymax, dx, dy) {
//     // how to use: var tiles = generate_grid(xmin, ymin, xmax, ymax, dx_utm, dy_utm)
//   var xx = ee.List.sequence(xmin, ee.Number(xmax), dx)
//   var yy = ee.List.sequence(ymin, ee.Number(ymax), dy)
//   var cells = xx.map(function(x) {
//     return yy.map(function(y) {
//       var x1 = ee.Number(x)
//       var x2 = ee.Number(x).add(ee.Number(dx))
//       var y1 = ee.Number(y)
//       var y2 = ee.Number(y).add(ee.Number(dy))
//       var coords = ee.List([x1, y1, x2, y2]);
//       var rect = ee.Algorithms.GeometryConstructors.Rectangle(coords, 'EPSG:3857');    //
//       // var rect = ee.Algorithms.GeometryConstructors.Rectangle(coords);     //
//       return ee.Feature(rect)
//     })
//   }).flatten();   //

//   return ee.FeatureCollection(cells);
// }


function generate_grid(lonmin, latmin, lonmax, latmax, dx_utm, dy_utm, buf, tile_num) {

  var xx = ee.List.sequence(1, tile_num, 1)
  function add_xtile(num, ls){
    var tiles_ls = ee.List(ls);
    var coords = ee.Feature(tiles_ls.get(-1)).geometry().coordinates().flatten()
    var point_ = ee.Algorithms.If({
                    condition: ee.Number(num).eq(1), 
                    trueCase: ee.Geometry.Point([lonmin, latmin]), 
                    falseCase: ee.Algorithms.If({
                        condition:  ee.Number(coords.get(0)).gt(lonmax),
                        trueCase: ee.Geometry.Point([lonmin, coords.get(7)]), 
                        falseCase: ee.Geometry.Point([coords.get(2), coords.get(3)])})
                                  })
    var point = ee.Geometry(point_)
    var lon = ee.Number(point.coordinates().get(0))
    var utm_zone = lon.divide(6).add(31).floor().byte()
    var epsg_utm = ee.String('326').cat(utm_zone)
    var proj_utm = ee.String('EPSG: ').cat(epsg_utm)
    var point_utm = point.transform(proj_utm).coordinates()
    var point_ur_utm = [ee.Number(dx_utm).add(point_utm.get(0)), 
                        ee.Number(dy_utm).add(point_utm.get(1))]
    var point_ur = ee.Geometry.Point(point_ur_utm, proj_utm).transform('EPSG:4326')
    var rect = ee.Algorithms.GeometryConstructors.Rectangle([point, point_ur])
    var rect_ = ee.Feature(rect).set('proj', proj_utm)
    return tiles_ls.add(rect_);
    
  }
  var tiles_geo = xx.iterate(add_xtile, [])

  //---- region buffer ----
  var tiles_geo_buf = ee.List(tiles_geo).map(function add_buf(fea) {
      var fea_ = ee.Feature(fea);
      var proj = ee.String(fea_.get('proj'));
      return fea_.buffer({'distance': 10000, 'proj': proj}).bounds(10)}
      )

  return ee.FeatureCollection(tiles_geo_buf.flatten())
}

//// parameters
var region_coord = area_tb_bound.coordinates()
var region_lon_min = region_coord.flatten().get(0)
var region_lat_min = region_coord.flatten().get(1)
var region_lon_max = region_coord.flatten().get(4)
var region_lat_max = region_coord.flatten().get(5)

print('region:', region_lon_min,region_lon_max,region_lat_min,region_lat_max)
var tile_num = 500
var buf = 10000
var dx_utm = 100000        // 100 km
var dy_utm = 100000
var lonmin = region_lon_min
var lonmax = region_lon_max
var latmin = region_lat_min
var latmax = region_lat_max

// ----- generate tiles
var tiles_tb = generate_grid(lonmin, latmin, lonmax, latmax, dx_utm, dy_utm, buf, tile_num)

// *******************************************************************
// ----- 2. remove empty grid and add region buffer
// *******************************************************************
function remove_tile(fea){
    var inter_ = area_tb.intersects({'right': fea.geometry(), 'maxError': 10});
    var fea_1 = fea.set('intersection', inter_);   //  intersection
    var fea_2 = fea_1.set('area', fea_1.area({'maxError':10}).divide(1000000));  // set area 
    return fea_2
}

var tiles_tb = tiles_tb.map(remove_tile).filterMetadata('intersection','equals', true)

// *******************************************************************
// ----- 3. add tile id: 1,2...
// *******************************************************************
function add_tiles_id(fea, tiles_tb){
    var tiles_ls = ee.List(tiles_tb);
    var id = tiles_ls.size();
    return tiles_ls.add(fea.set('tile_id', id));
}
var tiles_final = tiles_tb.iterate(add_tiles_id, []);

var tiles_final = ee.FeatureCollection(ee.List(tiles_final).flatten())  // convert ee.List to ee.FeatureCollection
print('tile final:',tiles_final)

// //// outline visualization of the study area.
var empty = ee.Image().byte();
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});
var tb_bound = empty.paint({
    featureCollection: area_tb_bound, color: 1, width: 3});

Map.setCenter(86.0, 32.0, 4);
Map.addLayer(tiles_final, {}, 'tiles_final')
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline');
Map.addLayer(tb_bound, {palette: '0000FF'}, 'Tibet_bound');


// // Export an ee.FeatureCollection as an Earth Engine asset.
// Export.table.toAsset({
//   collection: tiles_final,
//   description:'tibet_tiles',
//   assetId: 'tibet_tiles',
// });







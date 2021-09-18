//////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2021.08.31
// Description: export the tiled images in tibet.
//////////////////////////////////////////////////////////////


var bands_s1 = ['VV', 'VH']


// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = area_tb.geometry().bounds();

// ********************************************************
///////////////////// tiles and image_id load /////////////
// ********************************************************
var tb_tiles_buf = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_tiles_buf');
var s1_imgid_as = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_s1_202008_as')
var s1_imgid_des = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_s1_202008_des')
var tb_tiles_buf = tb_tiles_buf.sort('tile_id')
print('tb_tiles_buf:', tb_tiles_buf)
print('s1_imgid_as:', s1_imgid_as)
print('s1_imgid_des:', s1_imgid_des)

// ********************************************************
//////////////////////// Data export //////////////////////
/////////////  note: should set as/des manunally //////////
// ********************************************************
// --- 1. obtain s1 images
function get_id(fea, ls){
    var id_ls = ee.List(ls);
    return id_ls.add(fea.get('img_id'))
    }

// collection s1 images: ascending and descending
var s1_id_as = s1_imgid_as.iterate(get_id, [])
var s1_imgs_as = ee.ImageCollection(s1_id_as.getInfo()).select(bands_s1)
var s1_id_des = s1_imgid_des.iterate(get_id, [])
var s1_imgs_des = ee.ImageCollection(s1_id_des.getInfo()).select(bands_s1)
print('s1_imgs_as -- >', s1_imgs_as)

// // remove the edge pixles
// function remove_edge(img){
//     var inner_geo = img.geometry().buffer(-100);  // about 10 pixels.
//     return img.clip(inner_geo)
//     }
// var s1_imgs_as = s1_imgs_as.map(remove_edge)

var s1_img_as = s1_imgs_as.mosaic().float()   // mosaic imgs into img
var s1_img_des = s1_imgs_des.mosaic().float()   
print('s1_img_as_mosaic -- >', s1_img_as)
Map.addLayer(s1_img_as.select('VH'), {min: -50, max: 1}, 'ascendVH')
Map.addLayer(tb_tiles_buf, {}, 'tb_tiles_buf')

// ---2 export tiled images
for (var i = 0; i < 30; i++){
  // define tile name
  var tb_tiles_ls = tb_tiles_buf.toList(10000)
  var tile = tb_tiles_ls.get(i)
  var tile_fea = ee.Feature(tile)
  var tile_id = tile_fea.get('tile_id')
  var tile_name_as = 'tibet_s1as_202008_tile_' + tile_id.getInfo()
  var tile_name_des = 'tibet_s1des_202008_tile_' + tile_id.getInfo()
  print('output tile -->', tile_name_as, tile_name_des)
  // // export image: ascending and descending
  // Export.image.toDrive({
  //     image: s1_img_as,
  //     description: tile_name_as,
  //     folder: 'tibet_sar_data',
  //     scale: 10,
  //     maxPixels: 1e9,
  //     fileFormat: 'GeoTIFF',
  //     region: tile_fea.geometry(),
  //     });

  // Export.image.toDrive({
  //     image: s1_img_des,
  //     description: tile_name_des,
  //     folder: 'tibet_sar_data',
  //     scale: 10,
  //     maxPixels: 1e9,
  //     fileFormat: 'GeoTIFF',
  //     region: tile_fea.geometry(),
  //     });
  }



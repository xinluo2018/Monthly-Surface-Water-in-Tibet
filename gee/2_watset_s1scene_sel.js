//////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.20
// Description: This code checks the selected pair-wise sentinel-1 scenes, 
// and the matched sentinel-2 scenes.
//////////////////////////////////////////////////////////////

// tibet region
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var empty = ee.Image().byte();
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});

// satellite image (base map)
var sen2Coll = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(area_tb)
                  .filterDate('2020-07-15', '2020-10-15')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
                  .sort('CLOUDY_PIXEL_PERCENTAGE');

var sen2Img = sen2Coll.median()

// gsw dataset
var gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater');
var occurrence = gsw.select('occurrence').clip(area_tb)
var water_mask = occurrence.updateMask(occurrence.gt(50));

// training regions loading
var scene01_region = ee.Geometry.Rectangle(82.26, 33.82, 82.68, 34.14)
var scene01_sen2 = ee.Image('COPERNICUS/S2_SR/20190906T050659_20190906T051806_T44SPC').clip(scene01_region)
var scene01_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190908T123207_20190908T123232_028930_0347AE_1771').clip(scene01_region)
var scene01_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20190903T003331_20190903T003356_017866_0219F6_E5B3').clip(scene01_region)
// print(scene01_sen2,scene01_sen1_ascend,scene01_sen1_descend)

var scene02_region = ee.Geometry.Rectangle(97.93, 34.62, 98.35, 34.93) 
var scene02_sen2 = ee.Image('COPERNICUS/S2_SR/20200902T040549_20200902T040615_T47SMU').clip(scene02_region)
var scene02_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200829T112638_20200829T112703_034121_03F666_B1BF').clip(scene02_region)
var scene02_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200829T232821_20200829T232846_034128_03F6AD_DCC0').clip(scene02_region)
// print(tra2_sen2,tra2_sen1_ascend,tra2_sen1_descend)

var scene03_region = ee.Geometry.Rectangle(90.61, 28.84, 91.01, 29.15)
var scene03_sen2 = ee.Image('COPERNICUS/S2_SR/20201011T043719_20201011T043837_T46RBT').clip(scene03_region)
var scene03_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201012T115754_20201012T115819_034763_040CF1_EFA7').clip(scene03_region)
var scene03_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201007T235439_20201007T235504_034697_040A9F_19E9').clip(scene03_region)
// print(scene3_sen2,scene3_sen1_ascend,scene3_sen1_descend)

var scene04_region = ee.Geometry.Rectangle(86.67, 35.48, 87.09, 35.78) 
var scene04_sen2 = ee.Image('COPERNICUS/S2_SR/20200624T045701_20200624T050509_T45SVV').clip(scene04_region)
var scene04_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200619T120745_20200619T120810_033086_03D533_F8DC').clip(scene04_region)
var scene04_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200619T001644_20200619T001709_022095_029EE2_204F').clip(scene04_region)

var scene05_region = ee.Geometry.Rectangle(89.78, 33.82, 90.14, 34.11)
var scene05_sen2 = ee.Image('COPERNICUS/S2_SR/20200901T043709_20200901T043834_T45SYT').clip(scene05_region)
var scene05_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200825T115907_20200825T115932_034063_03F45D_8D93').clip(scene05_region)
var scene05_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200826T000130_20200826T000155_034070_03F49B_6CAB').clip(scene05_region)

var scene06_region = ee.Geometry.Rectangle(87.33, 31.77, 87.76, 32.09)
var scene06_sen2 = ee.Image('COPERNICUS/S2_SR/20200914T044659_20200914T045805_T45SWR').clip(scene06_region)
var scene06_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200916T121507_20200916T121532_034384_03FF9D_86C7').clip(scene06_region)
var scene06_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200912T001020_20200912T001045_034318_03FD5A_3E94').clip(scene06_region)

var scene07_region = ee.Geometry.Rectangle(81.14,30.43, 81.51, 30.73)
var scene07_sen2 = ee.Image('COPERNICUS/S2_SR/20200831T050659_20200831T051456_T44RNU').clip(scene07_region)
var scene07_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200826T123916_20200826T123941_034078_03F4DA_7766').clip(scene07_region)
var scene07_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T003512_20200903T003537_034187_03F8BE_B4F9').clip(scene07_region)

var scene08_region = ee.Geometry.Rectangle(97.43, 38.13, 97.82, 38.44)
var scene08_sen2 = ee.Image('COPERNICUS/S2_SR/20201025T041849_20201025T041843_T47SLC').clip(scene08_region)
var scene08_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201016T112729_20201016T112754_034821_040F00_0DBC').clip(scene08_region)
var scene08_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201021T233542_20201021T233607_034901_0411BF_67A8').clip(scene08_region)

var scene09_region = ee.Geometry.Rectangle(86.35,33.77,86.77,34.10)
var scene09_sen2 = ee.Image('COPERNICUS/S2_SR/20200818T045659_20200818T050858_T45SVT').clip(scene09_region)
var scene09_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200811T121530_20200811T121555_033859_03ED25_81E3').clip(scene09_region)
var scene09_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200818T001713_20200818T001738_022970_02B99E_0C3B').clip(scene09_region)

var scene10_region = ee.Geometry.Rectangle(90.95, 35.48, 91.35, 35.78)
var scene10_sen2 = ee.Image('COPERNICUS/S2_SR/20190927T043659_20190927T044051_T46SCE').clip(scene10_region)
var scene10_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190924T115927_20190924T115952_029163_034FB0_6868').clip(scene10_region)
var scene10_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190925T000059_20190925T000124_029170_034FEC_BBB6').clip(scene10_region)

var scene11_region = ee.Geometry.Rectangle(79.47,33.36,79.86,33.66)
var scene11_sen2 = ee.Image('COPERNICUS/S2_SR/20200923T051649_20200923T052705_T44SLC').clip(scene11_region)
var scene11_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200919T124007_20200919T124032_034428_040137_60A4').clip(scene11_region)
var scene11_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200920T004251_20200920T004320_034435_04017D_A03E').clip(scene11_region)

var scene12_region = ee.Geometry.Rectangle(98.70,29.38,99.08,29.68)
var scene12_sen2 = ee.Image('COPERNICUS/S2_SR/20191114T040019_20191114T040017_T47RMN').clip(scene12_region)
var scene12_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20191115T112452_20191115T112517_029921_036A01_CC09').clip(scene12_region)
var scene12_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20191110T232122_20191110T232147_029855_0367B3_CB0E').clip(scene12_region)


var scene13_region = ee.Geometry.Rectangle(100.45,35.82,100.85,36.12)
var scene13_sen2 = ee.Image('COPERNICUS/S2_SR/20190811T035541_20190811T040503_T47SPV').clip(scene13_region)
var scene13_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190818T111834_20190818T111859_028623_033D00_3E27').clip(scene13_region)
var scene13_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190818T231940_20190818T232005_028630_033D3D_FFD9').clip(scene13_region)

var scene14_region = ee.Geometry.Rectangle(91.27,31.82,91.67,32.13)
var scene14_sen2 = ee.Image('COPERNICUS/S2_SR/20190813T043701_20190813T043704_T46SCA').clip(scene14_region)
var scene14_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190807T115834_20190807T115859_028463_033779_A571').clip(scene14_region)
var scene14_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190802T235339_20190802T235404_028397_03357D_E806').clip(scene14_region)

var scene15_region = ee.Geometry.Rectangle(102.07,33.65,102.46,33.95)
var scene15_sen2 = ee.Image('COPERNICUS/S2_SR/20190816T035549_20190816T040759_T48STC').clip(scene15_region)
var scene15_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190813T110933_20190813T110958_028550_033A8E_47C3').clip(scene15_region)
var scene15_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190813T231155_20190813T231220_028557_033AC7_DD17').clip(scene15_region)


var scene16_region = ee.Geometry.Rectangle(96.08,32.95,96.47,33.26)
var scene16_sen2 = ee.Image('COPERNICUS/S2_SR/20200831T041551_20200831T042007_T47SKS').clip(scene16_region)
var scene16_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T113413_20200903T113438_034194_03F8ED_E4E4').clip(scene16_region)
var scene16_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T233656_20200903T233721_034201_03F92F_8084').clip(scene16_region)


var scene17_region = ee.Geometry.Rectangle(94.42,29.29,94.80,29.58)
var scene17_sen2 = ee.Image('COPERNICUS/S2_SR/20200213T041901_20200213T041911_T46RFT').clip(scene17_region)
var scene17_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200217T114124_20200217T114149_031292_039993_6FF2').clip(scene17_region)
var scene17_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200212T233803_20200212T233828_031226_039745_94C5').clip(scene17_region)


var scene18_region = ee.Geometry.Rectangle(74.69,38.55,75.09,38.84)
var scene18_sen2 = ee.Image('COPERNICUS/S2_SR/20200907T054641_20200907T054642_T43SDC').clip(scene18_region)
var scene18_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200905T125805_20200905T125830_034224_03FA09_2DBA').clip(scene18_region)
var scene18_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200911T010600_20200911T010625_034304_03FCD1_0AA8').clip(scene18_region)


var scene_region_19 = ee.Geometry.Rectangle(83.85,31.29,84.22,31.58)
var scene19_sen2 = ee.Image('COPERNICUS/S2_SR/20190926T050659_20190926T051645_T45RTQ').clip(scene_region_19)
var scene19_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190927T122307_20190927T122332_029207_03512E_DCB2').clip(scene_region_19)
var scene19_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20191004T002612_20191004T002641_018318_02281E_3F05').clip(scene_region_19)


var scene20_region = ee.Geometry.Rectangle(88.94,37.39,89.36,37.69)
var scene20_sen2 = ee.Image('COPERNICUS/S2_SR/20190707T044711_20190707T045327_T45SXB').clip(scene20_region)
var scene20_sen1_ascend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190702T120012_20190702T120037_027938_032783_FE5F').clip(scene20_region)
var scene20_sen1_descend = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190708T000826_20190708T000855_028018_032A07_B240').clip(scene20_region)



// // Map.centerObject(area_tb, 5);
// // satellite image and tibet region
// Map.addLayer(sen2Img, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'sen2 image')
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet region')
// // scene region 01
Map.addLayer(scene01_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene01 sen2 image')
// Map.addLayer(scene01_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene01 sen1 ascend')
// Map.addLayer(scene01_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene01 sen1 descend')

// // scene region 02
Map.addLayer(scene02_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene02 sen2 image')
// Map.addLayer(scene02_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene02 sen1 ascend')
// Map.addLayer(scene02_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene02 sen1 descend')

// // scene region 03
Map.addLayer(scene03_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene03 sen2 image')
// Map.addLayer(scene03_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene03 sen1 ascend')
// Map.addLayer(scene03_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, ‘scene03 sen1 descend')

// scene region 04
Map.addLayer(scene04_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene04 sen2 image')
// Map.addLayer(scene04_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene04 sen1 ascend')
// Map.addLayer(scene04_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene04 sen1 descend')

// scene region 05
Map.addLayer(scene05_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene05 sen2 image')
// Map.addLayer(scene05_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene05 sen1 ascend')
// Map.addLayer(scene05_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene05 sen1 descend')

// // scene region 06
Map.addLayer(scene06_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene06 sen2 image')
// Map.addLayer(scene06_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene06 sen1 ascend')
// Map.addLayer(scene06_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene06 sen1 descend')

// scene region 07
Map.addLayer(scene07_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene07 sen2 image')
// Map.addLayer(scene07_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene07 sen1 ascend')
// Map.addLayer(scene07_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene07 sen1 descend')

// //scene region 08
Map.addLayer(scene08_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene08 sen2 image')
// Map.addLayer(scene08_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene08 sen1 ascend')
// Map.addLayer(scene08_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene08 sen1 descend')

// // scene region 09
Map.addLayer(scene09_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene09 sen2 image')
// Map.addLayer(scene09_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene09 sen1 ascend')
// Map.addLayer(scene09_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene09 sen1 descend')


// // scene region 10
Map.addLayer(scene10_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene10 sen2 image')
// Map.addLayer(scene10_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene10 sen1 ascend')
// Map.addLayer(scene10_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene10 sen1 descend')


// // scene region 11
Map.addLayer(scene11_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene11 sen2 image')
// Map.addLayer(scene11_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene11 sen1 ascend')
// Map.addLayer(scene11_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene11 sen1 descend')


// // scene region 12
Map.addLayer(scene12_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene12 sen2 image')
// Map.addLayer(scene12_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene12 sen1 ascend')
// Map.addLayer(scene12_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene12 sen1 descend')

// // scene region 13
Map.addLayer(scene13_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene13 sen2 image')
// Map.addLayer(scene13_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene13 sen1 ascend')
// Map.addLayer(scene13_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene13 sen1 descend')

// // scene region 14
Map.addLayer(scene14_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene14 sen2 image')
// Map.addLayer(scene14_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene14 sen1 ascend')
// Map.addLayer(scene14_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene14 sen1 descend')

// // scene region 15
Map.addLayer(scene15_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene15 sen2 image')
// Map.addLayer(scene15_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene15 sen1 ascend')
// Map.addLayer(scene15_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene15 sen1 descend')

// // scene region 16
Map.addLayer(scene16_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene16 sen2 image')
// Map.addLayer(scene16_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene16 sen1 ascend')
// Map.addLayer(scene16_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene16 sen1 descend')

// // scene region 17
Map.addLayer(scene17_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene17 sen2 image');
// Map.addLayer(scene17_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene17 sen1 ascend')
// Map.addLayer(scene17_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene17 sen1 descend')

// // scene region 18
Map.addLayer(scene18_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene18 sen2 image');
// Map.addLayer(scene18_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene18 sen1 ascend')
// Map.addLayer(scene18_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene18 sen1 descend')

// // scene region 19
Map.addLayer(scene19_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene19 sen2 image');
// Map.addLayer(scene19_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene19 sen1 ascend')
// Map.addLayer(scene19_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene19 sen1 descend')

// // scene region 20
Map.addLayer(scene20_sen2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene20 sen2 image');
// Map.addLayer(scene20_sen1_ascend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene20 sen1 ascend')
// Map.addLayer(scene20_sen1_descend, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene20 sen1 descend')


// Map.addLayer(water_mask, {palette: ['blue']}, '50% occurrence water mask');
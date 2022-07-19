import sys
sys.path.append("/home/yons/Desktop/developer-luo/Monthly-Surface-Water-in-Tibet")

from notebooks import config
from dataloader.img_aug import missing_band_p, rotate, flip, torch_noise, missing_region, numpy2tensor
from model.seg_model.deeplabv3_plus import deeplabv3plus
from model.seg_model.model_scales_in import unet_scales
from model.seg_model.model_scales_gate import unet_scales_gate

# model = unet_scales_gate(num_bands=4, num_classes=2)
model = deeplabv3plus(num_bands=4, num_classes=2)
if model.name == 'deeplabv3plus':
  print(model.name)



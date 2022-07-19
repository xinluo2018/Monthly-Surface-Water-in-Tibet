
import torch.nn as nn
from model.base_model.xception65 import Xception65
from model.base_model.mobilenet import MobileNetV2

class Xception65_feat(nn.Module):
    '''original encoder for deeplabv3_plus moel
      retrun:
        high-level + low-level features

    '''
    def __init__(self, num_bands=4):
        super(Xception65_feat, self).__init__()
        self.backbone = Xception65(num_bands)
    def forward(self,x):
        '''output -> low feature + high feature'''
        ## output_stride = 32
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.relu(x)
        x = self.backbone.block1(x)
        x = self.backbone.relu(x)    
        fea_low = x      # size -> 1/4, num_channel -> 128
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        # Middle flow
        x = self.backbone.midflow(x)
        # fea_mid = x      # size -> 1/16, num_channel -> 728
        # Exit flow
        x = self.backbone.block20(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv3(x)
        x = self.backbone.bn3(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv4(x)
        x = self.backbone.bn4(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv5(x)
        x = self.backbone.bn5(x)
        x = self.backbone.relu(x) 
        fea_high = x      # size -> 1/32, num_channels -> 2048
        return fea_low, fea_high



class Mobilev2_feat(nn.Module):
    '''return:
        low-level + middle-level + high-level features'''
    def __init__(self, num_bands=4):
        super(Mobilev2_feat, self).__init__()
        self.backbone = MobileNetV2(num_bands=num_bands)
    def forward(self,x):
        x = self.backbone.head(x)
        x = self.backbone.body.inverted_0(x)
        fea_low = x    # size -> 1/2, num_channel -> 16
        x = self.backbone.body.inverted_1(x) 
        fea_mid = x    # size -> 1/4, num_channel -> 24
        x = self.backbone.body.inverted_2(x)
        x = self.backbone.body.inverted_3(x)
        fea_high = x   # size -> 1/16, num_channel -> 64
        return fea_low, fea_mid, fea_high

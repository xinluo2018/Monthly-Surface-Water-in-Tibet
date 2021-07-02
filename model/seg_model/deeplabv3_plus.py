import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .helper import aspp
from .encoder import Xception65_feat
from .encoder import Mobilev2_feat
from .helper import conv1x1_bn_relu, conv3x3_bn_relu,deconv4x4_bn_relu


class deeplabv3plus(nn.Module):
    '''des: original deeplabv3_plus model'''
    def __init__(self, num_bands=4, num_classes=2, backbone=Xception65_feat):
        super(deeplabv3plus, self).__init__()        
        self.backbone = backbone(num_bands=num_bands)
        self.aspp = aspp(in_channels=2048, atrous_rates=[12, 24, 36])
        self.low_block = conv1x1_bn_relu(128, 48)
        self.high_low_block = nn.Sequential(      
                            conv3x3_bn_relu(304, 256),
                            nn.Dropout(0.5),
                            conv3x3_bn_relu(256, 256),
                            nn.Dropout(0.1),
                            )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256,out_channels=1, kernel_size=1),
                    nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1),
                    nn.Softmax(dim=1))


    def forward(self,x):
        fea_low, fea_high = self.backbone(x) 
        ## -------high-level features------- ##
        x_fea_high = self.aspp(fea_high)    # channels: -> 256
        x_fea_high = F.interpolate(x_fea_high, fea_low.size()[-2:], \
                                    mode='bilinear', align_corners=True)

        ## -------low-level features------- ##
        x_fea_low = self.low_block(fea_low)   # channels: ->48

        ## -------features concat-------  ##
        x_fea_concat = torch.cat([x_fea_high, x_fea_low], dim=1) # 
        x_fea_concat = self.high_low_block(x_fea_concat)
        x_out_prob = self.outp_layer(x_fea_concat)
        x_out_prob = F.interpolate(x_out_prob, x.size()[-2:], \
                                        mode='bilinear', align_corners=True)
        return x_out_prob

class deeplabv3plus_imp(nn.Module): 
    def __init__(self, num_bands, num_classes, channels_fea=[16, 24, 64]):
        ''' 
        Improvement: 1. use mobilenetv2; 2. use a mid-level feature.
        channels_fea (list) -> [chan_low,chan_mid,chan_high] '''
        super(deeplabv3plus_imp, self).__init__()
        self.aspp_channels = 256
        self.backbone = Mobilev2_feat(num_bands=num_bands)
        self.aspp = aspp(in_channels=channels_fea[2], atrous_rates=[12, 24, 36])
        self.mid_layer = conv1x1_bn_relu(channels_fea[1], 128)
        self.high_mid_layer = nn.Sequential(
                        conv1x1_bn_relu(128+self.aspp_channels, 128),
                        conv3x3_bn_relu(128, 128)
                        )
        self.low_layer = conv1x1_bn_relu(channels_fea[0], 128)
        self.high_mid_low_layer = nn.Sequential(
                        deconv4x4_bn_relu(128+128, 256),
                        nn.Dropout(0.5),
                        conv1x1_bn_relu(256, 128),
                        conv3x3_bn_relu(128, 128),
                        nn.Dropout(0.1),
                        )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                    nn.Sigmoid())
        else: 
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128,out_channels=num_classes, kernel_size=1),
                    nn.Softmax(dim=1))

    def forward(self,x):
        ### 
        fea_low, fea_mid, fea_high = self.backbone(x)

        ### ------high level feature
        x_fea_high = self.aspp(fea_high)        # channels:256
        x_fea_high = F.interpolate(x_fea_high, \
                        fea_mid.size()[-2:], mode='bilinear', align_corners=True)
        ### ------mid-level feature, and concat
        x_fea_mid = self.mid_layer(fea_mid)
        x_fea_high_mid = torch.cat([x_fea_high, x_fea_mid], dim=1)
        x_fea_high_mid = self.high_mid_layer(x_fea_high_mid)
        x_fea_high_mid = F.interpolate(x_fea_high_mid, \
                        fea_low.size()[-2:], mode='bilinear', align_corners=True)
        ### ------low-level feature, and concat
        x_fea_low = self.low_layer(fea_low)
        x_fea_high_mid_low = torch.cat([x_fea_high_mid, x_fea_low], dim=1)
        x_fea_high_mid_low = self.high_mid_low_layer(x_fea_high_mid_low)
        out_prob = self.outp_layer(x_fea_high_mid_low)
        return out_prob


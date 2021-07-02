'''
author: xin luo
date: 2021.6.29
'''

import torch
import torch.nn as nn
from model.seg_model.helper import dsample, upsample
import torch.nn.functional as F
from .helper import convert_g_l, dwconv3x3_bn_relu,conv3x3_bn_relu


class unet_triple(nn.Module):
    ''' 
    description: unet model for single-scale image processing
    '''
    def __init__(self, num_bands, num_classes, scale_high=2048, scale_mid=512, scale_low=256):
        super(unet_triple, self).__init__()
        self.num_classes = num_classes
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.high2low_ratio = scale_high//scale_low
        self.mid2low_ratio = scale_mid//scale_low
        self.encoder = nn.ModuleList([
            dsample(in_channels=num_bands, ex_channels=32, out_channels=16, scale=2), # 1/2
            dsample(in_channels=16, ex_channels=64, out_channels=16, scale=2),   # 1/4
            dsample(in_channels=16, ex_channels=128, out_channels=32, scale=2),  # 1/8
            dsample(in_channels=32, ex_channels=128, out_channels=32, scale=4),  # 1/32
            dsample(in_channels=32, ex_channels=256, out_channels=64, scale=4),  # 1/128
        ])
        self.decoder = nn.ModuleList([
            upsample(in_channels=64*3, out_channels=64, scale=4),         # 1/32
            upsample(in_channels=64+32*3, out_channels=64, scale=4),      # 1/8
            upsample(in_channels=64+32*3, out_channels=64, scale=2),      # 1/4
            upsample(in_channels=64+16*3, out_channels=32, scale=2),      # 1/2
        ])
        self.up_last = upsample(in_channels=32+16*3, out_channels=32, scale=2)
        self.dropout = nn.Dropout(p=0.5)
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                        nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                        nn.Softmax(dim=1))

    def forward(self, input):
        x_encode_high, x_encode_mid, x_encode_low = input[0], input[1], input[2]
        '''--------feature encoding---------'''
        skips_high, skips_mid, skips_low= [],[],[]

        '''---low-level feature learning'''
        for encode in self.encoder:
            x_encode_low = encode(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])

        '''---mid-level feature learning'''
        for encode in self.encoder:
            x_encode_mid = encode(x_encode_mid)
            x_encode_mid2low = convert_g_l(img_g=x_encode_mid, \
                                            scale_ratio=self.mid2low_ratio)
            skips_mid.append(x_encode_mid2low)
        skips_mid = reversed(skips_mid[:-1])

        '''---high-level feature learning'''
        for encode in self.encoder:
            x_encode_high = encode(x_encode_high)
            x_encode_high2low = convert_g_l(img_g=x_encode_high, \
                                            scale_ratio=self.mid2low_ratio)
            skips_high.append(x_encode_high2low)
        skips_high = reversed(skips_high[:-1])

        '''-------- feature decoding-------'''
        '''--- feature fusion'''
        x_decode = x_encode_high
        x_decode = torch.cat((x_encode_low, x_encode_mid2low, x_encode_high2low), 1)

        for i, (decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.decoder, skips_high, skips_mid, skips_low)):
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip_high, skip_mid, skip_low], dim=1)

        output = self.up_last(x_decode)
        out_prob = self.outp_layer(output)
        return out_prob


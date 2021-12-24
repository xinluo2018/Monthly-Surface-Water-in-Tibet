'''
author: xin luo
date: 2021.6.29
'''

import torch
import torch.nn as nn
from model.seg_model.helper import dsample, upsample
import torch.nn.functional as F
from .helper import convert_g_l


class unet_scales_gate(nn.Module):
    ''' 
    description: unet model for single-scale image processing
    '''
    def __init__(self, num_bands, num_classes, scale_high=2048, scale_mid=512, scale_low=256):
        super(unet_scales_gate, self).__init__()
        self.name = 'unet_scales_gate'
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

        self.encoder_gate = nn.ModuleList([
            dsample(in_channels=num_bands, ex_channels=32, out_channels=16, scale=2), # 1/2
            dsample(in_channels=16, ex_channels=64, out_channels=16, scale=2),   # 1/4
            dsample(in_channels=16, ex_channels=128, out_channels=32, scale=2),  # 1/8
            dsample(in_channels=32, ex_channels=128, out_channels=32, scale=4),  # 1/32
            dsample(in_channels=32, ex_channels=256, out_channels=64, scale=4),  # 1/128
        ])

        self.decoder_gate = nn.ModuleList([
            upsample(in_channels=64, out_channels=64, scale=4),         # 1/32
            upsample(in_channels=64+32*1, out_channels=64, scale=4),      # 1/8
            upsample(in_channels=64+32*1, out_channels=64, scale=2),      # 1/4
            upsample(in_channels=64+16*1, out_channels=32, scale=2),      # 1/2
        ])

        self.gate_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
                        #   nn.Tanh()),
                          nn.Sigmoid()),  

            nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=1, kernel_size=1),
                        #   nn.Tanh()),
                          nn.Sigmoid()),

            nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=1, kernel_size=1),
                        #   nn.Tanh()),
                          nn.Sigmoid()),

            nn.Sequential(nn.Conv2d(in_channels=64+16, out_channels=1, kernel_size=1),
                        #   nn.Tanh()),
                          nn.Sigmoid()),

            nn.Sequential(nn.Conv2d(in_channels=32+16, out_channels=1, kernel_size=1),
                        #   nn.Tanh()),
                          nn.Sigmoid()),
                          ])

        self.up_last = upsample(in_channels=32+16*3, out_channels=32, scale=2)
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
        '''
        ---------------------------------------------------
        --------------- 1. feature encoding ---------------
        ---------------------------------------------------
        '''
        skips_high2low, skips_mid2low, skips_low= [],[],[]

        '''--- 1.1 low-level feature learning'''
        for encode in self.encoder:
            x_encode_low = encode(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])

        '''--- 1.2 mid-level feature learning'''
        for encode in self.encoder:
            x_encode_mid = encode(x_encode_mid)
            x_encode_mid2low = convert_g_l(img_g=x_encode_mid, \
                                            scale_ratio=self.mid2low_ratio)
            skips_mid2low.append(x_encode_mid2low)
        skips_mid2low = reversed(skips_mid2low[:-1])

        '''--- 1.3 high-level feature learning'''
        for encode in self.encoder:
            x_encode_high = encode(x_encode_high)
            x_encode_high2low = convert_g_l(img_g=x_encode_high, \
                                            scale_ratio=self.mid2low_ratio)
            skips_high2low.append(x_encode_high2low)
        skips_high2low = reversed(skips_high2low[:-1])

        '''
        ---------------------------------------------------
        ----------------- 2. gate learning ----------------
        ---------------------------------------------------
        '''                
        '''--- 2.1 mid-scale gate: determined by low-scale feature
               a) low-scale feature encoding '''
        x_encode_mid_gate = input[2]    #  low-scale patch: to determine mid-feature gate
        skips_mid_gate = []
        for encode in self.encoder_gate:
            x_encode_mid_gate = encode(x_encode_mid_gate)
            skips_mid_gate.append(x_encode_mid_gate)
        skips_mid_gate = list(reversed(skips_mid_gate))

        ''' b) low feature decoding'''
        gates_mid = []
        gate = self.gate_layers[0](x_encode_mid_gate)
        gates_mid.append(gate)
        x_decode = x_encode_mid_gate
        for i, (decode, skip_mid_gate) in enumerate(zip(self.decoder_gate, skips_mid_gate[1:])):            
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip_mid_gate], dim=1)
            gate = self.gate_layers[i+1](x_decode)        # obtain mid-level gate 
            gates_mid.append(gate)

        '''--- 2.2 high-scale gate: determined by low/mid-scale feature
               a) mid-feature encoding and b) encoded low feature + encoded mid feature '''
        x_encode_high_gate = input[1]    #  mid-scale patch: to determine high-feature gate
        skips_high_gate = []
        for i, encode in enumerate(self.encoder_gate):
            x_encode_high_gate = encode(x_encode_high_gate)
            x_encode_high_gate_down = convert_g_l(img_g=x_encode_high_gate, \
                                                scale_ratio=self.mid2low_ratio)
            x_encode_high_gate_concat = x_encode_high_gate_down+skips_mid_gate[-(i+1)]  # mid-feature + low-feature
            skips_high_gate.append(x_encode_high_gate_concat)  
        skips_high_gate = list(reversed(skips_high_gate))

        '''--- c) low+mid feature decoding and obtain high-level gate ----'''
        gates_high = []
        gate = self.gate_layers[0](skips_high_gate[0])
        gates_high.append(gate)

        x_decode = skips_high_gate[0]
        for i, (decode, skip_high_gate) in enumerate(zip(self.decoder_gate, skips_high_gate[1:])):            
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip_high_gate], dim=1)    # decoded mid feature + (encoded low fea + encoded mid fea)
            gate = self.gate_layers[i+1](x_decode)        # obtain high-level gate 
            gates_high.append(gate)

        '''
        ---------------------------------------------------
        ---------------- 3. feature decoding --------------
        ---------------------------------------------------
        '''
        '''--- feature fusion'''
        # x_decode = torch.cat((x_encode_low, x_encode_mid2low*gates_mid[0]), 1)
        x_decode = torch.cat((x_encode_low, x_encode_mid2low*gates_mid[0], x_encode_high2low*gates_high[0]), 1)
        for i, (decode, skip_low, skip_mid2low, skip_high2low) in enumerate(zip(self.decoder, skips_low, skips_mid2low, skips_high2low)):
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip_low, skip_mid2low*gates_mid[i+1], \
                                                            skip_high2low*gates_high[i+1]], dim=1)
        output = self.up_last(x_decode)
        out_prob = self.outp_layer(output)
        return out_prob, gates_mid, gates_high

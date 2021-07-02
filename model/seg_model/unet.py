import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.seg_model.helper import dsample, upsample

class unet(nn.Module):
    ''' 
    description: unet model for single-scale image processing
    '''
    def __init__(self, num_bands, num_classes):
        super(unet, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.ModuleList([
            dsample(in_channels=num_bands, ex_channels=32, out_channels=16, scale=2), # 1/2
            dsample(in_channels=16, ex_channels=64, out_channels=16, scale=2),   # 1/4
            dsample(in_channels=16, ex_channels=128, out_channels=32, scale=2),  # 1/8
            dsample(in_channels=32, ex_channels=128, out_channels=32, scale=4),  # 1/32
            dsample(in_channels=32, ex_channels=256, out_channels=64, scale=4),  # 1/128
        ])
        self.decoder = nn.ModuleList([
            upsample(in_channels=64, out_channels=64, scale=4),    # 1/32
            upsample(in_channels=64+32, out_channels=64, scale=4), # 1/8
            upsample(in_channels=64+32, out_channels=64, scale=2), # 1/4
            upsample(in_channels=64+16, out_channels=32, scale=2), # 1/2
        ])
        self.up_last = upsample(in_channels=32+16, out_channels=32, scale=2)
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                        nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                        nn.Softmax(dim=1))


    def forward(self, input):
        x_encode = input
        '''feature encoding'''
        skips = []
        for encode in self.encoder:
            x_encode = encode(x_encode)
            skips.append(x_encode)
        skips = reversed(skips[:-1])

        '''feature decoding'''
        x_decode = x_encode
        for i, (decode, skip) in enumerate(zip(self.decoder, skips)):
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip], dim=1)
        output = self.up_last(x_decode)
        out_prob = self.outp_layer(output)
        return out_prob



import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def dwconv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
            kernel_size=3, stride=1, padding=1, groups=in_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv4x4_bn_relu(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, \
                                    kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def convert_g_l(img_g, scale_ratio):
    '''global_size should be divisible by local_size.
    '''
    size_g = img_g.shape[2]
    size_l = size_g//scale_ratio
    if size_l >= 1:
        ''' crop -> enlarge scale '''
        start_crop = (size_g - size_l)//2
        img_l_crop = img_g[:,:, start_crop:start_crop+size_l, start_crop:start_crop+size_l]
        img_l = F.interpolate(img_l_crop, size=[size_g, size_g], mode='nearest')

    else:
        ''' enlarge scale -> crop '''
        start_crop = (size_g*scale_ratio - size_l*scale_ratio)//2
        img_g_up = F.interpolate(img_g, size=[size_g*scale_ratio, size_g*scale_ratio], mode='nearest')
        img_l = img_g_up[:, :, start_crop:start_crop+size_l*scale_ratio, start_crop:start_crop+size_l*scale_ratio]

    return img_l


####-----------for the unet-----------####
class dsample(nn.Module):
    '''down x2: pooling->conv_bn_relu->dwconv_bn_relu->conv_bn_relu
       down x4: pooling->conv_bn_relu->dwconv_bn_relu->dwconv_bn_relu->conv_bn_relu
    '''
    def __init__(self, in_channels, ex_channels, out_channels, scale = 2, **kwargs):
        super(dsample, self).__init__()
        self.scale = scale
        self.pool = nn.AvgPool2d(kernel_size=scale)
        self.conv_bn_relu_in = conv3x3_bn_relu(in_channels, ex_channels)
        self.dwconv_bn_relu_1 = dwconv3x3_bn_relu(ex_channels, ex_channels)
        self.dwconv_bn_relu_2 = dwconv3x3_bn_relu(ex_channels, ex_channels)
        self.conv_bn_relu_out = conv1x1_bn_relu(ex_channels, out_channels)
    def forward(self, x):
        if self.scale == 2:
            x = self.pool(x)
            x = self.conv_bn_relu_in(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale == 4:
            x = self.pool(x)
            x = self.conv_bn_relu_in(x) 
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x

class upsample(nn.Module):
    '''up x2: up_resize -> dwconv_bn_relu -> conv_bn_relu 
       up x4: up_resize -> dwconv_bn_relu -> dwconv_bn_relu -> conv_bn_relu 
    '''
    def __init__(self, in_channels, out_channels, scale = 2, **kwargs):
        super(upsample, self).__init__()
        self.scale = scale
        self.up2_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_layer = nn.Upsample(scale_factor=4, mode='nearest')
        self.dwconv_bn_relu_1 = dwconv3x3_bn_relu(in_channels, in_channels)
        self.dwconv_bn_relu_2 = dwconv3x3_bn_relu(in_channels, in_channels)
        self.conv_bn_relu_out = conv3x3_bn_relu(in_channels, out_channels)

    def forward(self, x):
        if self.scale == 2:
            x = self.up2_layer(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale == 4:
            x = self.up4_layer(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x

####--------------for the deeplabv3_plus-----------####
def aspp_conv(in_channels, out_channels, dilation):
    '''aspp convolution: atrous_conv + bn + relu'''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, \
                            padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class aspp_pooling(nn.Module):
    '''aspp pooling: pooling + 1x1 conv + bn + relu'''
    def __init__(self, in_channels, out_channels):
        super(aspp_pooling, self).__init__()
        self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),     ## size -> 1x1
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.layers(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

class aspp(nn.Module):
    '''aspp module: consist of aspp conv, aspp pooling...'''
    def __init__(self, in_channels, atrous_rates):
        super(aspp, self).__init__()
        out_channels = 256   
        modules = []
        # branch 1：conv2d+bn+relu
        modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)  ## 
        ## branch 2-4：atrous convolution
        modules.append(aspp_conv(in_channels, out_channels, rate1))
        modules.append(aspp_conv(in_channels, out_channels, rate2))
        modules.append(aspp_conv(in_channels, out_channels, rate3))
        ## branch 5：pooling
        modules.append(aspp_pooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules) 
        ## for merged branches：conv2d+bn+relu+dropout
        self.project = nn.Sequential(
                    nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




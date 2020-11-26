import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import torch.optim as optim
class SPBlock(nn.Module):

    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = int(outplanes // 2)

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(
            3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(
            1, 3), padding=(0, 1), bias=False)

        self.fuse_conv = nn.Conv2d(
            midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        self.conv_final = nn.Conv2d(
            midplanes, outplanes, kernel_size=1, bias=True)

        self.mask_conv_1 = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, padding=1)

    def forward(self, x):
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)

        hx = self.relu(self.fuse_conv((x_1_h + x_1_w)))
        mask_1 = self.conv_final(hx).sigmoid()

        out1 = x * mask_1

        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()
        hx = out1 * mask_2

        return hx

def init_net(net, init_type='orthogonal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class LayerActivation:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.device = device
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.skip1 = LayerActivation(self.vgg.features, 3)
        self.skip2 = LayerActivation(self.vgg.features, 8)
        self.skip3 = LayerActivation(self.vgg.features, 15)
        self.skip4 = LayerActivation(self.vgg.features, 22)
        self.skip5 = LayerActivation(self.vgg.features, 29)

    def forward(self, x):
        self.vgg(x)

        return x, self.skip1.features.to(self.device), self.skip2.features.to(self.device) \
            , self.skip3.features.to(self.device), self.skip4.features.to(self.device), self.skip5.features.to(self.device),


def upsample(x, convT, skip, conv1x1, device):
    x = convT(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0/255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1(x)
    return x


def upsample_last(x, conv1x1_64_3, skip, conv1x1_6_3, device):
    x = conv1x1_64_3(x)
    bn = nn.BatchNorm2d(x.shape[1]).to(device)
    x = bn(x)
    x = F.leaky_relu(x, 0.2)

    skip = torch.log(skip ** 2 + 1.0 / 255.0)
    x = torch.cat([x, skip], dim=1)
    x = conv1x1_6_3(x)
    return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.latent_representation = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.convTranspose_5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_5 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.conv1x1_4 = Conv2d(1024, 512, kernel_size=1)

        self.convTranspose_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv1x1_3 = Conv2d(512, 256, kernel_size=1)

        self.convTranspose_2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv1x1_2 = Conv2d(256, 128, kernel_size=1)

        self.convTranspose_1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1x1_1 = Conv2d(128, 64, kernel_size=1)

        self.conv1x1_64_3 = Conv2d(64, 3, kernel_size=1)
        self.conv1x1_6_3 = Conv2d(6, 3, kernel_size=1)

    def forward(self, skip0, skip1, skip2, skip3, skip4, skip5):
        x = self.latent_representation(skip5)
        x = upsample(x, self.convTranspose_5, skip5, self.conv1x1_5, self.device)
        x = upsample(x, self.convTranspose_4, skip4, self.conv1x1_4, self.device)
        x = upsample(x, self.convTranspose_3, skip3, self.conv1x1_3, self.device)
        x = upsample(x, self.convTranspose_2, skip2, self.conv1x1_2, self.device)
        x = upsample(x, self.convTranspose_1, skip1, self.conv1x1_1, self.device)
        x = upsample_last(x, self.conv1x1_64_3, skip0, self.conv1x1_6_3, self.device)
        return x


# class Model(nn.Module):
#     def __init__(self, device):
#         super(Model, self).__init__()
#         self.encoder = Encoder(device)
#         self.decoder = Decoder(device)

#     def forward(self, x):
#         x = x.float()
#         skip0, skip1, skip2, skip3, skip4, skip5 = self.encoder(x)
#         x = self.decoder(skip0, skip1, skip2, skip3, skip4, skip5)
#         return x



#se-net
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#strip pooling
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, up_kwargs = {'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()

        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool5 = nn.AdaptiveAvgPool2d((3, None))
        self.pool6 = nn.AdaptiveAvgPool2d((None, 3))
        self.pool7 = nn.AdaptiveAvgPool2d((5, None))
        self.pool8 = nn.AdaptiveAvgPool2d((None, 5))
        self.pool9 = nn.AdaptiveAvgPool2d((7, None))
        self.pool10 = nn.AdaptiveAvgPool2d((None, 7))
        self.sigmoid = nn.Sigmoid()
#         inter_channels = int(in_channels/2)
        inter_channels = in_channels

        self.conv1_2 = nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
                               #kernel size?
        self.conv2_3 = nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False)
        self.conv2_4 = nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False)
        self.conv2_5 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (0, 1), bias=False)
        self.conv2_6 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 0), bias=False)
        self.conv2_7 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (0, 1), bias=False)
        self.conv2_8 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 0), bias=False)
        self.conv2_9 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (0, 1), bias=False)
        self.conv2_10 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 0), bias=False)


        self.conv2_11 = nn.Conv2d(inter_channels*4, inter_channels, 3, 1, 1, bias=False)

        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                        nn.ReLU(True))
        self.conv3_1 = nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False)

        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x2 = self.conv1_2(x)
        x_conv = x2
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), (h, w), **self._up_kwargs)
        x2_7 = F.interpolate(self.conv2_6(self.pool6(x2)), (h, w), **self._up_kwargs)
        x2_8 = F.interpolate(self.conv2_7(self.pool7(x2)), (h, w), **self._up_kwargs)
        x2_9 = F.interpolate(self.conv2_8(self.pool8(x2)), (h, w), **self._up_kwargs)
        x2_10 = F.interpolate(self.conv2_7(self.pool9(x2)), (h, w), **self._up_kwargs)
        x2_11 = F.interpolate(self.conv2_8(self.pool10(x2)), (h, w), **self._up_kwargs)
        one = x2_5 + x2_4
        three = x2_6+ x2_7
        five = x2_8+ x2_9
        seven = x2_10 + x2_11
        x2 = self.conv2_11(torch.cat([one, three, five, seven], dim=1) )
        x2 = self.sigmoid(x2)
#         print(x2.shape)
#         print(x_conv.shape)
        sp_out = x2 * x_conv
        sp_out_weight = self.sigmoid(self.conv3_1(self.conv3(sp_out)))

        return sp_out_weight * sp_out


#dilated conv
class dilated_conv(nn.Module):
    """
    Reference:
    """
    def __init__(self, inter_channels):
        super(dilated_conv, self).__init__()

#         kernel_size, stride=1, padding=0, dilation=1, groups=1
#need to check shape
        self.conv1 = nn.Conv2d(inter_channels, inter_channels, (1, 1), 1, (0, 0), bias=False)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels, (3, 3), 1, (1, 1), bias=False)
        self.conv5 = nn.Conv2d(inter_channels, inter_channels, (5, 5), 1, (2, 2), bias=False)
        self.conv = nn.Conv2d(inter_channels*3, inter_channels, 3, 1, 1, bias=False)


    def forward(self, x):
        one = self.conv1(x)
        three = self.conv3(x)
        five = self.conv5(x)
#         print(one.shape)
#         print(three.shape)
#         print(five.shape)
        x2 = self.conv(torch.cat([one, three, five], dim=1) )
        return x2

class attention_block(nn.Module):
    def __init__(self, in_channels):
        super(attention_block, self).__init__()
        self.sp_layer = StripPooling(in_channels)
        self.de_layer = dilated_conv(in_channels)
        #self.se_layer = SELayer(in_channels)
        self.conv1 = nn.Conv2d((in_channels*2), in_channels*2, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d((in_channels*2), in_channels, 3, 1, 1, bias=False)
    def forward(self, x):
        sp_out = self.sp_layer(x)
        de_out = self.de_layer(sp_out)
        #se_out = self.se_layer(sp_out)
#         print(x.shape)
#         print(sp_out.shape)
#         print(se_out.shape)
        conv_out1 = self.conv1(torch.cat([sp_out, de_out], dim=1))
        conv_out2 = self.conv2(conv_out1)
        return conv_out2 + x

class attention_block_Encoder(nn.Module):
    def __init__(self, in_channels,downsample=True):
        super(attention_block_Encoder, self).__init__()
        self.sp_layer = StripPooling(in_channels)
        self.de_layer = dilated_conv(in_channels)
#         self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.down_conv = nn.Conv2d((in_channels*2), in_channels*2, 3, 2, 1, bias=False)
        self.downsample = downsample
    def forward(self, x):
        sp_out = self.sp_layer(x)
        de_out = self.de_layer(sp_out)
        down_out =self.down_conv(torch.cat([sp_out, de_out], dim=1))
#         print(torch.cat([sp_out, de_out], dim=1).shape)
#         print(down_out.shape)
        if self.downsample:
            return down_out
        else:
            return torch.cat([sp_out, de_out], dim=1)

class Encoder_att(nn.Module):
    def __init__(self, in_channels):
        super(Encoder_att, self).__init__()

        self.att1 = attention_block_Encoder(in_channels,False)
        self.att2 = attention_block_Encoder(in_channels*2)
        self.att3 = attention_block_Encoder(in_channels*4)
        self.att4 = attention_block_Encoder(in_channels*8)
        self.in_conv = nn.Conv2d(1, in_channels, 3, 1, 1, bias=False)
        self.in_conv2 = nn.Conv2d(in_channels, in_channels*2, 3, 1, 1, bias=False)

        self.sk_conv1 = SPBlock(1, in_channels * 4)
        self.sk_conv2 = SPBlock(1, in_channels * 8)
        self.sk_conv3 = SPBlock(1, in_channels * 16)

        self.sk_conv5 = SPBlock(1, in_channels * 4)
        self.sk_conv6 = SPBlock(1, in_channels * 8)
        self.sk_conv7 = SPBlock(1, in_channels * 16)

    def forward(self, x, skips):
        in_out = self.in_conv(x)
        in_out2 = self.in_conv2(in_out)
        skips['fir_pool1'][0] = self.sk_conv1(skips['fir_pool1'][0])
        skips['fir_pool2'][0] = self.sk_conv2(skips['fir_pool2'][0])
        skips['fir_pool3'][0] = self.sk_conv3(skips['fir_pool3'][0])

        skips['sec_pool1'][0] = self.sk_conv5(skips['sec_pool1'][0])
        skips['sec_pool2'][0] = self.sk_conv6(skips['sec_pool2'][0])
        skips['sec_pool3'][0] = self.sk_conv7(skips['sec_pool3'][0])

        # print(skips['sec_pool1'][0].shape)
        # print(skips['sec_pool2'][0].shape)
        # print(skips['sec_pool3'][0].shape)


        att1_out = self.att1(in_out)
        att2_out = self.att2(att1_out)
        att3_out = self.att3(att2_out+skips['fir_pool1'][0]+skips['sec_pool1'][0])
        att4_out = self.att4(att3_out+skips['fir_pool2'][0]+skips['sec_pool2'][0])
        att4_out += skips['fir_pool3'][0]+skips['sec_pool3'][0]
#         print('in',in_out.shape)
#         print(att1_out.shape)
#         print(att2_out.shape)
#         print(att3_out.shape)
#         print(att4_out.shape)
# in torch.Size([1, 32, 512, 512])
# torch.Size([1, 64, 256, 256])
# torch.Size([1, 128, 128, 128])
# torch.Size([1, 256, 64, 64])
# torch.Size([1, 512, 32, 32])
        return in_out2, att1_out, att2_out, att3_out, att4_out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):

    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Model(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(Model,self).__init__()
        self.encoder_att = Encoder_att(32)
        # resnet = models.resnet34(pretrained=False)

        ## -------------Encoder--------------

        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=False)

        # #stage 1
        # self.encoder1 = resnet.layer1 #224
        # #stage 2
        # self.encoder2 = resnet.layer2 #112
        # #stage 3
        # self.encoder3 = resnet.layer3 #56
        # #stage 4
        # self.encoder4 = resnet.layer4 #28

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #14

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #7

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=False)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=False)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=False)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=False)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=False)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=False)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=False)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=False)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=False)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(512,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=False)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=False)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=False)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(256,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=False)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=False)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=False)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(128,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=False)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=False)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=False)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=False)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=False)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=False)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear', align_corners=True)###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear', align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)


        self.cc1 = nn.Conv2d(134,128,3,padding=1)
        self.cc2= nn.Conv2d(262,256,3,padding=1)
        self.cc3 = nn.Conv2d(518,512,3,padding=1)


        #change to fusion =cf
        self.cf1 =  nn.Conv2d(512,512,3,padding=1)
        self.cf2 =  nn.Conv2d(256,256,3,padding=1)
        self.cf3 =  nn.Conv2d(128,128,3,padding=1)


        self.out1 = nn.Conv2d(64,64,3,padding=1)
        self.out2 = nn.Conv2d(64,1,3,padding=1)
        self.bn_out_1 = nn.BatchNorm2d(3)
        self.bn_out_2 = nn.BatchNorm2d(3)

        self.tanh = nn.Tanh()

        #for wavelet conv add
        self.wav_fir1 = nn.Conv2d(1,512,3,padding=1)
        self.wav_fir2 = nn.Conv2d(1,512,3,padding=1)
        self.wav_fir3 = nn.Conv2d(1,512,3,padding=1)

        self.wav_sec1 = nn.Conv2d(1,512,3,padding=1)
        self.wav_sec2 = nn.Conv2d(1,512,3,padding=1)
        self.wav_sec3 = nn.Conv2d(1,512,3,padding=1)

        self.wav_fir4 = nn.Conv2d(1,256,3,padding=1)
        self.wav_fir5 = nn.Conv2d(1,256,3,padding=1)
        self.wav_fir6 = nn.Conv2d(1,256,3,padding=1)

        self.wav_sec4 = nn.Conv2d(1,256,3,padding=1)
        self.wav_sec5 = nn.Conv2d(1,256,3,padding=1)
        self.wav_sec6 = nn.Conv2d(1,256,3,padding=1)

        self.wav_fir7 = nn.Conv2d(1,128,3,padding=1)
        self.wav_fir8 = nn.Conv2d(1,128,3,padding=1)
        self.wav_fir9= nn.Conv2d(1,128,3,padding=1)

        self.wav_sec7 = nn.Conv2d(1,128,3,padding=1)
        self.wav_sec8 = nn.Conv2d(1,128,3,padding=1)
        self.wav_sec9 = nn.Conv2d(1,128,3,padding=1)

    def forward(self,x,skips):
        self.use_att_en = True
        self.use_attention = False
        hx = x
        residual = x
#         print(self.att_encode)
        # if self.use_att_en:
        hx, h1, h2, h3, h4 = self.encoder_att(hx, skips)
        #     print('\n ===========>encoder_att')
        # else:
            ## -------------Encoder-------------
#         hx = self.inconv(hx)
#         hx = self.inbn(hx)
#         hx = self.inrelu(hx)
#         h1 = self.encoder1(hx) # 256
#         print('\n ===========>resnet_encoder')


# #     #         print(skips['fir_pool1'][0].shape)
# #     #         print(skips['fir_pool2'][0].shape)
# #     #         print(skips['fir_pool3'][0].shape)

# #             #Change Channel = cc
# #     #         cc1 = cc1(skips['fir_pool1'][0])
# #     #         cc2 = cc2(skips['sec_pool1'][0])

#         h2 = self.encoder2(h1) # 128


# #             #128+3+3
#         fusion1 = torch.cat([h2,skips['fir_pool1'][0],skips['sec_pool1'][0]],1)

# #             print('self.use_attention= ',self.use_attention)
#         if self.use_attention:
#             fusion1 = self.att_cc1(fusion1)
#             fusion1 = self.cc1(fusion1)
#             print('use attention')
#         else:
#             fusion1 = self.cc1(fusion1)
#             print('nooooo attention')


#         h3 = self.encoder3(fusion1) # 64

# #              #256+3+3
#         fusion2 = torch.cat([h3,skips['fir_pool2'][0],skips['sec_pool2'][0]],1)
#         if self.use_attention:
#             fusion2 = self.att_cc2(fusion2)
#             fusion2 = self.cc2(fusion2)
#         else:
#             fusion2 = self.cc2(fusion2)




#         h4 = self.encoder4(fusion2) # 32 512channel

# #             #512+3+3
#         fusion3 = torch.cat([h4,skips['fir_pool3'][0],skips['sec_pool3'][0]],1)
#         if self.use_attention:
#             fusion3 = self.att_cc3(fusion3)
#             fusion3 = self.cc3(fusion3)
#         else:
#             fusion3 = self.cc3(fusion3)
#         h4 += fusion3


### ori encoder end


#         print(hx.shape)
#         print(h1.shape)
#         print(h2.shape)
#         print(h3.shape)
#         print(h4.shape)

# torch.Size([17, 64, 224, 224])
# torch.Size([17, 64, 224, 224])

# torch.Size([17, 128, 112, 112])
# torch.Size([17, 256, 56, 56])
# torch.Size([17, 512, 28, 28])

# torch.Size([17, 128, 112, 112])
# torch.Size([17, 256, 56, 56])
# torch.Size([17, 512, 28, 28])
# skips[fir_pool3] 4
        # h2 = self.encoder2(h1+skips['fir_pool1'][0]+skips['sec_pool1'][0]) # 128
        # h3 = self.encoder3(h2+skips['fir_pool2'][0]+skips['sec_pool2'][0]) # 64
        # h4 = self.encoder4(h3+skips['fir_pool3'][0]+skips['sec_pool3'][0]) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32
    #         print('skips[fir_pool3]',len(skips['fir_pool3']))



        skips['fir_pool3'][1] = self.wav_fir1(skips['fir_pool3'][1])
        skips['fir_pool3'][2] = self.wav_fir2(skips['fir_pool3'][2])
        skips['fir_pool3'][3] = self.wav_fir3(skips['fir_pool3'][3])

        skips['sec_pool3'][1] = self.wav_sec1(skips['sec_pool3'][1])
        skips['sec_pool3'][2] = self.wav_sec2(skips['sec_pool3'][2])
        skips['sec_pool3'][3] = self.wav_sec3(skips['sec_pool3'][3])
        # high_level1 = skips['fir_pool3'][1]*0.15+skips['fir_pool3'][2]*0.15+skips['fir_pool3'][3]*0.2+skips['sec_pool3'][1]*0.15+skips['sec_pool3'][2]*0.15+skips['sec_pool3'][3]*0.2
        high_level1 = skips['fir_pool3'][1]+skips['fir_pool3'][2]+skips['fir_pool3'][3]+skips['sec_pool3'][1]+skips['sec_pool3'][2]+skips['sec_pool3'][3]


    #         print('pool3\n')
    #         print(hx.shape)
    #         print(h4.shape)
    #         print(high_level1.shape)
    #         print(high_level2.shape)

        high_fusion = high_level1#+high_level2
    #         print('h4',hx.shape)
    #         print(h4.shape)
        feature_fusion = hx + h4
    #         print(high_fusion.shape)
    #         print('hx + h4',feature_fusion.shape)
        adaptive_feature = self.cf1(feature_fusion+high_fusion)

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(adaptive_feature)))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        skips['fir_pool2'][1] = self.wav_fir4(skips['fir_pool2'][1])
        skips['fir_pool2'][2] = self.wav_fir5(skips['fir_pool2'][2])
        skips['fir_pool2'][3] = self.wav_fir6(skips['fir_pool2'][3])

        skips['sec_pool2'][1] = self.wav_sec4(skips['sec_pool2'][1])
        skips['sec_pool2'][2] = self.wav_sec5(skips['sec_pool2'][2])
        skips['sec_pool2'][3] = self.wav_sec6(skips['sec_pool2'][3])

    #         print('weight fusion')
        # high_level1 = skips['fir_pool2'][1]*0.15+skips['fir_pool2'][2]*0.15+skips['fir_pool2'][3]*0.2+skips['sec_pool2'][1]*0.15+skips['sec_pool2'][2]*0.15+skips['sec_pool2'][3]*0.2
        high_level1 = skips['fir_pool2'][1]+skips['fir_pool2'][2]+skips['fir_pool2'][3]+skips['sec_pool2'][1]+skips['sec_pool2'][2]+skips['sec_pool2'][3]
    #         high_level2 = sum(skips['sec_pool2'][1]+skips['sec_pool2'][2]+skips['sec_pool2'][3])

    #         print('pool2\n')
    #         print(hx.shape)
    #         print(h3.shape)
    #         print(high_level1.shape)
    #         print(high_level2.shape)

        high_fusion = high_level1
    #         print(hx.shape)
    #         print(h3.shape)
        feature_fusion = hx + h3

    #         print('hx + h3',feature_fusion.shape)
        adaptive_feature = self.cf2(feature_fusion+high_fusion)

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(adaptive_feature)))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128

        skips['fir_pool1'][1] = self.wav_fir7(skips['fir_pool1'][1])
        skips['fir_pool1'][2] = self.wav_fir8(skips['fir_pool1'][2])
        skips['fir_pool1'][3] = self.wav_fir9(skips['fir_pool1'][3])

        skips['sec_pool1'][1] = self.wav_sec7(skips['sec_pool1'][1])
        skips['sec_pool1'][2] = self.wav_sec8(skips['sec_pool1'][2])
        skips['sec_pool1'][3] = self.wav_sec9(skips['sec_pool1'][3])

        # high_level1 = skips['fir_pool1'][1]*0.15+skips['fir_pool1'][2]*0.15+skips['fir_pool1'][3]*0.2+skips['sec_pool1'][1]*0.15+skips['sec_pool1'][2]*0.15+skips['sec_pool1'][3]*0.2
        high_level1 = skips['fir_pool1'][1]+skips['fir_pool1'][2]+skips['fir_pool1'][3]+skips['sec_pool1'][1]+skips['sec_pool1'][2]+skips['sec_pool1'][3]
    #         high_level2 = sum(skips['sec_pool1'][1]+skips['sec_pool1'][2]+skips['sec_pool1'][3])
    #         torch.cat()
    #         print('pool1\n')
    #         print(hx.shape)
    #         print(h2.shape)
    #         print(high_level1.shape)
    #         print(high_level2.shape)

        high_fusion = high_level1#+high_level2
        feature_fusion = hx + h2
    #         print(high_fusion.shape)
    #         print('hx + h2',feature_fusion.shape)
        adaptive_feature = self.cf3(feature_fusion+high_fusion)

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(adaptive_feature)))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2) # 128->256

    #         d1 = self.outconv1(hd1) # 256

    #         d1 = self.out1(d1)
        d1 = self.tanh(self.out2(hd1)) + residual
        return (d1), (d2), (d3), (d4), (d5), (d6), (db)



import torch.nn as nn
class HDRWaveEnDe_simple_decompose(nn.Module):

    def __init__(self, option_unpool='cat5'):
        super(HDRWaveEnDe_simple_decompose, self).__init__()
        self.skips = {}

        self.option_unpool = option_unpool

        self.avgpool = nn.AvgPool2d((2, 2), stride=2)

        self.En_fir_pool1 = WavePool(1)

        self.En_fir_pool2 = WavePool(1)

        self.En_fir_pool3 = WavePool(1)

        self.En_sec_pool1 = WavePool(1)

        self.En_sec_pool2 = WavePool(1)

        self.En_sec_pool3 = WavePool(1)

#fusion then wavelet pooling
    def forward(self, x1, x2):
        # Encoder
        # odd is top branch , evEn is low branch

        res_x11 = self.avgpool(x1)
        res_x12 = self.avgpool(res_x11)
        res_x13 = self.avgpool(res_x12)

        res_x21 = self.avgpool(x2)
        res_x22 = self.avgpool(res_x21)
        res_x23 = self.avgpool(res_x22)

        fir_LL, fir_LH, fir_HL, fir_HH = self.En_fir_pool1(x1)
        sec_LL, sec_LH, sec_HL, sec_HH = self.En_sec_pool1(x2)
        self.skips['fir_pool1'] = [fir_LL, fir_LH, fir_HL, fir_HH]
        self.skips['sec_pool1'] = [sec_LL, sec_LH, sec_HL, sec_HH]


        fir_LL, fir_LH, fir_HL, fir_HH = self.En_fir_pool2(res_x11)
        sec_LL, sec_LH, sec_HL, sec_HH = self.En_sec_pool2(res_x21)
        self.skips['fir_pool2'] = [fir_LL, fir_LH, fir_HL, fir_HH]
        self.skips['sec_pool2'] = [sec_LL, sec_LH, sec_HL, sec_HH]

        fir_LL, fir_LH, fir_HL, fir_HH = self.En_fir_pool3(res_x12)
        self.skips['fir_pool3'] = [fir_LL, fir_LH, fir_HL, fir_HH]

        sec_LL, sec_LH, sec_HL, sec_HH = self.En_sec_pool3(res_x22)
        self.skips['sec_pool3'] = [sec_LL, sec_LH, sec_HL, sec_HH]

        return self.skips
def get_norm_layer(norm_type='none'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                #norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            #norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='instance', use_sigmoid=False, init_type='orthogonal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,3,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x,ori_input):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return ori_input + residual

class WaveUnpool(nn.Module):

    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(
            self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' :
            # print(self.LL(LL).shape)
            # print(self.LH(LL).shape)
            # print(original.shape)
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH)], dim=1)
        else:
            raise NotImplementedError


class Model2(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(Model2,self).__init__()
        self.encoder_att = Encoder_att(32)

        ## -------------Encoder--------------

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #14

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #7

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=False)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=False)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=False)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=False)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=False)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=False)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=False)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=False)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=False)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(512,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=False)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=False)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=False)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(256,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=False)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=False)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=False)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(128,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=False)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=False)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=False)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=False)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=False)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=False)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear', align_corners=True)###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)


        self.cc1 = nn.Conv2d(134,128,3,padding=1)
        self.cc2= nn.Conv2d(262,256,3,padding=1)
        self.cc3 = nn.Conv2d(518,512,3,padding=1)

        self.att_cc1 = attention_block(134)
        self.att_cc2 = attention_block(262)
        self.att_cc3 = attention_block(518)



        #change to fusion =cf
        self.cf1_1 =  nn.Conv2d(2048,1024,3,padding=1)
        self.bn5cf1_1 = nn.BatchNorm2d(1024)
        self.cf1_2 =  nn.Conv2d(1024,512,3,padding=1)
        self.bn5cf1_2 = nn.BatchNorm2d(512)
        self.cf1_3 =  nn.Conv2d(512,512,3,padding=1)
        self.bn5cf1_3 = nn.BatchNorm2d(512)

        self.cf2_1 =  nn.Conv2d(1024,512,3,padding=1)
        self.bn5cf2_1 = nn.BatchNorm2d(512)
        self.cf2_2 =  nn.Conv2d(512,256,3,padding=1)
        self.bn5cf2_2 = nn.BatchNorm2d(256)
        self.cf2_3 =  nn.Conv2d(256,256,3,padding=1)
        self.bn5cf2_3 = nn.BatchNorm2d(256)

        self.cf3_1 =  nn.Conv2d(512,256,3,padding=1)
        self.bn5cf3_1 = nn.BatchNorm2d(256)
        self.cf3_2 =  nn.Conv2d(256,128,3,padding=1)
        self.bn5cf3_2 = nn.BatchNorm2d(128)
        self.cf3_3 =  nn.Conv2d(128,128,3,padding=1)
        self.bn5cf3_3 = nn.BatchNorm2d(128)

        self.out1 = nn.Conv2d(64,64,3,padding=1)
        self.out2 = nn.Conv2d(64,1,3,padding=1)
        self.bn_out_1 = nn.BatchNorm2d(3)
        self.bn_out_2 = nn.BatchNorm2d(3)

        self.tanh = nn.Tanh()

        #for wavelet conv add
        self.wav_fir1 = SPBlock(1, 512)
        self.wav_fir2 = SPBlock(1, 512)
        self.wav_fir3 = SPBlock(1, 512)

        self.wav_sec1 = SPBlock(1, 512)
        self.wav_sec2 = SPBlock(1, 512)
        self.wav_sec3 = SPBlock(1, 512)

        self.wav_fir4 = SPBlock(1, 256)
        self.wav_fir5 = SPBlock(1, 256)
        self.wav_fir6 = SPBlock(1, 256)

        self.wav_sec4 = SPBlock(1, 256)
        self.wav_sec5 = SPBlock(1, 256)
        self.wav_sec6 = SPBlock(1, 256)

        self.wav_fir7 = SPBlock(1, 128)
        self.wav_fir8 = SPBlock(1, 128)
        self.wav_fir9 = SPBlock(1, 128)

        self.wav_sec7 = SPBlock(1, 128)
        self.wav_sec8 = SPBlock(1, 128)
        self.wav_sec9 = SPBlock(1, 128)
        option_unpool = 'cat5'
        self.unpool1 = WaveUnpool(512, option_unpool)
        self.unpool2 = WaveUnpool(512, option_unpool)
        self.unpool3 = WaveUnpool(256, option_unpool)
        self.unpool4 = WaveUnpool(256, option_unpool)
        self.unpool5 = WaveUnpool(128, option_unpool)
        self.unpool6 = WaveUnpool(128, option_unpool)

        self.refunet = RefUnet(1, 64)
    def forward(self,x,skips):
        # print('\n it is attention_unpool file')
        self.use_att_en = True
        self.use_attention = False
        hx = x
        residual = x
#         print(self.att_encode)
        # if self.use_att_en:
        hx, h1, h2, h3, h4 = self.encoder_att(hx, skips)
        #     print('\n ===========>encoder_att')
        # else:
            ## -------------Encoder-------------
#         hx = self.inconv(hx)
#         hx = self.inbn(hx)
#         hx = self.inrelu(hx)
#         h1 = self.encoder1(hx) # 256
#         print('\n ===========>resnet_encoder')


# #     #         print(skips['fir_pool1'][0].shape)
# #     #         print(skips['fir_pool2'][0].shape)
# #     #         print(skips['fir_pool3'][0].shape)

# #             #Change Channel = cc
# #     #         cc1 = cc1(skips['fir_pool1'][0])
# #     #         cc2 = cc2(skips['sec_pool1'][0])

#         h2 = self.encoder2(h1) # 128


# #             #128+3+3
#         fusion1 = torch.cat([h2,skips['fir_pool1'][0],skips['sec_pool1'][0]],1)

# #             print('self.use_attention= ',self.use_attention)
#         if self.use_attention:
#             fusion1 = self.att_cc1(fusion1)
#             fusion1 = self.cc1(fusion1)
#             print('use attention')
#         else:
#             fusion1 = self.cc1(fusion1)
#             print('nooooo attention')


#         h3 = self.encoder3(fusion1) # 64

# #              #256+3+3
#         fusion2 = torch.cat([h3,skips['fir_pool2'][0],skips['sec_pool2'][0]],1)
#         if self.use_attention:
#             fusion2 = self.att_cc2(fusion2)
#             fusion2 = self.cc2(fusion2)
#         else:
#             fusion2 = self.cc2(fusion2)




#         h4 = self.encoder4(fusion2) # 32 512channel

# #             #512+3+3
#         fusion3 = torch.cat([h4,skips['fir_pool3'][0],skips['sec_pool3'][0]],1)
#         if self.use_attention:
#             fusion3 = self.att_cc3(fusion3)
#             fusion3 = self.cc3(fusion3)
#         else:
#             fusion3 = self.cc3(fusion3)
#         h4 += fusion3


### ori encoder end


#         print(hx.shape)
#         print(h1.shape)
#         print(h2.shape)
#         print(h3.shape)
#         print(h4.shape)

# torch.Size([17, 64, 224, 224])
# torch.Size([17, 64, 224, 224])

# torch.Size([17, 128, 112, 112])
# torch.Size([17, 256, 56, 56])
# torch.Size([17, 512, 28, 28])

# torch.Size([17, 128, 112, 112])
# torch.Size([17, 256, 56, 56])
# torch.Size([17, 512, 28, 28])
# skips[fir_pool3] 4
        # h2 = self.encoder2(h1+skips['fir_pool1'][0]+skips['sec_pool1'][0]) # 128
        # h3 = self.encoder3(h2+skips['fir_pool2'][0]+skips['sec_pool2'][0]) # 64
        # h4 = self.encoder4(h3+skips['fir_pool3'][0]+skips['sec_pool3'][0]) # 32

        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32
    #         print('skips[fir_pool3]',len(skips['fir_pool3']))



        skips['fir_pool3'][1] = self.wav_fir1(skips['fir_pool3'][1])
        skips['fir_pool3'][2] = self.wav_fir2(skips['fir_pool3'][2])
        skips['fir_pool3'][3] = self.wav_fir3(skips['fir_pool3'][3])

        skips['sec_pool3'][1] = self.wav_sec1(skips['sec_pool3'][1])
        skips['sec_pool3'][2] = self.wav_sec2(skips['sec_pool3'][2])
        skips['sec_pool3'][3] = self.wav_sec3(skips['sec_pool3'][3])

        # print((hx + h4).shape)
        # print(skips['fir_pool3'][1].shape)
        # print(skips['fir_pool3'][0].shape)
        # print(skips['fir_pool3'][2].shape)
        # print(skips['fir_pool3'][3].shape)

        ###change
        high = self.unpool1( hx + h4, skips['fir_pool3'][1], skips['fir_pool3'][2], skips['fir_pool3'][3])
        low = self.unpool2( hx + h4, skips['sec_pool3'][1], skips['sec_pool3'][2], skips['sec_pool3'][3])
        adaptive_feature = self.relu5d_2(self.bn5cf1_1(self.cf1_1(high+low)))
        adaptive_feature = self.relu5d_2(self.bn5cf1_2(self.cf1_2(adaptive_feature)))
        adaptive_feature = self.relu5d_2(self.bn5cf1_3(self.cf1_3(adaptive_feature)))
        # print(high.shape)
        # print(low.shape)




        # high_level1 = skips['fir_pool3'][1]*0+skips['fir_pool3'][2]*0+skips['fir_pool3'][3]*0+skips['sec_pool3'][1]*0+skips['sec_pool3'][2]*0+skips['sec_pool3'][3]*0
        # high_level1 = skips['fir_pool3'][1]+skips['fir_pool3'][2]+skips['fir_pool3'][3]+skips['sec_pool3'][1]+skips['sec_pool3'][2]+skips['sec_pool3'][3]


    ### original
        # high_level1 = skips['fir_pool3'][1]+skips['fir_pool3'][2]+skips['fir_pool3'][3]+skips['sec_pool3'][1]+skips['sec_pool3'][2]+skips['sec_pool3'][3]
        # high_fusion = high_level1#+high_level2
        # feature_fusion = hx + h4
        # adaptive_feature = self.cf1(feature_fusion+high_fusion)

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(adaptive_feature)))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        skips['fir_pool2'][1] = self.wav_fir4(skips['fir_pool2'][1])
        skips['fir_pool2'][2] = self.wav_fir5(skips['fir_pool2'][2])
        skips['fir_pool2'][3] = self.wav_fir6(skips['fir_pool2'][3])

        skips['sec_pool2'][1] = self.wav_sec4(skips['sec_pool2'][1])
        skips['sec_pool2'][2] = self.wav_sec5(skips['sec_pool2'][2])
        skips['sec_pool2'][3] = self.wav_sec6(skips['sec_pool2'][3])


        # high_level1 = skips['fir_pool2'][1]+skips['fir_pool2'][2]+skips['fir_pool2'][3]+skips['sec_pool2'][1]+skips['sec_pool2'][2]+skips['sec_pool2'][3]
    #         high_level2 = sum(skips['sec_pool2'][1]+skips['sec_pool2'][2]+skips['sec_pool2'][3])


###ori
        # high_level1 = skips['fir_pool2'][1]*0+skips['fir_pool2'][2]*0.0+skips['fir_pool2'][3]*0.0+skips['sec_pool2'][1]*0.0+skips['sec_pool2'][2]*0.0+skips['sec_pool2'][3]*0
        # high_fusion = high_level1
        # feature_fusion = hx + h3
        # adaptive_feature = self.cf2(feature_fusion+high_fusion)
###change
        # print((hd4 + h3).shape)

        high = self.unpool3( hd4 + h3, skips['fir_pool2'][1], skips['fir_pool2'][2], skips['fir_pool2'][3])
        low = self.unpool4( hd4 + h3, skips['sec_pool2'][1], skips['sec_pool2'][2], skips['sec_pool2'][3])
        # adaptive_feature = self.cf2(high+low)
        adaptive_feature = self.relu5d_2(self.bn5cf2_1(self.cf2_1(high+low)))
        adaptive_feature = self.relu5d_2(self.bn5cf2_2(self.cf2_2(adaptive_feature)))
        adaptive_feature = self.relu5d_2(self.bn5cf2_3(self.cf2_3(adaptive_feature)))

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(adaptive_feature)))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        # hx = self.upscore2(hd3) # 64 -> 128

        skips['fir_pool1'][1] = self.wav_fir7(skips['fir_pool1'][1])
        skips['fir_pool1'][2] = self.wav_fir8(skips['fir_pool1'][2])
        skips['fir_pool1'][3] = self.wav_fir9(skips['fir_pool1'][3])

        skips['sec_pool1'][1] = self.wav_sec7(skips['sec_pool1'][1])
        skips['sec_pool1'][2] = self.wav_sec8(skips['sec_pool1'][2])
        skips['sec_pool1'][3] = self.wav_sec9(skips['sec_pool1'][3])

        # high_level1 = skips['fir_pool1'][1]*0.0+skips['fir_pool1'][2]*0.0+skips['fir_pool1'][3]*0.0 + skips['sec_pool1'][1]*0.0+skips['sec_pool1'][2]*0.0+skips['sec_pool1'][3]*0.0
        #
    #         high_level2 = sum(skips['sec_pool1'][1]+skips['sec_pool1'][2]+skips['sec_pool1'][3])
    #         torch.cat()
    #         print('pool1\n')
    #         print(hx.shape)
    #         print(h2.shape)
    #         print(high_level1.shape)
    #         print(high_level2.shape)


    #         print(high_fusion.shape)
    #         print('hx + h2',feature_fusion.shape)

###ori
        # high_level1 = skips['fir_pool1'][1]+skips['fir_pool1'][2]+skips['fir_pool1'][3]+skips['sec_pool1'][1]+skips['sec_pool1'][2]+skips['sec_pool1'][3]
        # high_fusion = high_level1#+high_level2
        # feature_fusion = hx + h2
        # adaptive_feature = self.cf3(feature_fusion+high_fusion)

    ###change
        high = self.unpool5( hd3 + h2, skips['fir_pool1'][1], skips['fir_pool1'][2], skips['fir_pool1'][3])
        low = self.unpool6( hd3 + h2, skips['sec_pool1'][1], skips['sec_pool1'][2], skips['sec_pool1'][3])
        # adaptive_feature = self.cf3(high+low)
        adaptive_feature = self.relu5d_2(self.bn5cf3_1(self.cf3_1(high+low)))
        adaptive_feature = self.relu5d_2(self.bn5cf3_2(self.cf3_2(adaptive_feature)))
        adaptive_feature = self.relu5d_2(self.bn5cf3_3(self.cf3_3(adaptive_feature)))




        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(adaptive_feature)))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        ###change
        hx = hd2
        #ori
        # hx = self.upscore2(hd2) # 128 -> 256


        # print(hx.shape)
        # print(h1.shape)
        # print(self.conv1d_1)
        # print(skips['fir_pool1'][2].shape)
        # print(skips['fir_pool2'][2].shape)
        # print(skips['fir_pool3'][2].shape)

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        # d2 = self.upscore2(d2) # 128->256

    #         d1 = self.outconv1(hd1) # 256

    #         d1 = self.out1(d1)
        d1 = self.tanh(self.out2(hd1)) + residual
        # ref_out = self.refunet(d1, residual)
        return (d1), (d2), (d3), (d4), (d5), (d6), (db)


# if __name__ =='__main__':
#     in1 = torch.ones(1,3,512,512).cuda()
#     in2 = torch.ones(1,3,512,512).cuda()
#     bas = BASNet(3,3).cuda()
# #     print(bas)
# #     print(bas(in1))
#     net = HDRWaveEnDe_simple_decompose().cuda()
#     skip = net(in1,in2)
# #     print(skip['fir_pool1'])
#     print(bas(in1,skip).shape)
# #     summary(bas(in1,skip), (3, 512, 512))
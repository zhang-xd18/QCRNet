r""" The proposed CRNet
"""

import torch
import torch.nn as nn
import sys
from collections import OrderedDict
#import quantization
from .quantization import Quantization
sys.path.append("..")
from utils import logger

import scipy.io as io
import numpy as np
import os

__all__ = ["crnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class CRNet(nn.Module):
    def __init__(self, reduction=4, mode='mu', nbit=8):
        super(CRNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        logger.info(f'reduction={reduction}')
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.sigmoid = nn.Sigmoid()

        self.quant = Quantization(nbit=nbit,mode=mode)
        self.quant_sigmoid = nn.Sigmoid()

        self.identity = nn.Identity()
        self.quant_sigmoid_out = None
        self.quant_out = None

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, save_data=False, path='./data'):
        n, c, h, w = x.detach().size()
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        #quantization
        out = self.quant_sigmoid(out)

        if save_data:
            if self.quant_sigmoid_out == None:
                self.quant_sigmoid_out = out
            else:
                self.quant_sigmoid_out = torch.cat((self.quant_sigmoid_out, out))
        
        identity = self.identity(out)
        out, SNR = self.quant(out)

        if save_data:
            if self.quant_out == None:
                self.quant_out = out
            else:
                self.quant_out = torch.cat((self.quant_out, out))

        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        if save_data:
            self._save(path)

        return out, identity, SNR

    def _save(self, path='./'):
        io.savemat(os.path.join(path, 'sigmoid_out.mat'),{'result':np.array(self.quant_sigmoid_out.cpu())})
        io.savemat(os.path.join(path, 'quant_out.mat'),{'result':np.array(self.quant_out.cpu())})


def crnet(reduction=4,mode='mu',nbit=8):
    model = CRNet(reduction=reduction,mode=mode,nbit=nbit)
    return model


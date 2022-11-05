r""" The proposed CRNet
"""

import torch
import torch.nn as nn
import sys
from collections import OrderedDict
from .quantization import Quantization
sys.path.append("..")
from utils import logger

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


class Adaptor(nn.Module):
    def __init__(self, reduction=4, bottle='b'):
        super(Adaptor, self).__init__()
        total_size = 2048
        self.bottle = bottle
        if bottle == 'b':
            self.fc = nn.Sequential(OrderedDict([
                ("fc_in", nn.Linear(total_size // reduction, total_size // (8 * reduction))),
                ("fc_out",nn.Linear(total_size // (8 * reduction), total_size // reduction))
            ]))
        elif bottle == 'pb':
            self.fc1 = nn.Sequential(OrderedDict([
                ("fc_in", nn.Linear(total_size // reduction, total_size // (8 * reduction))),
                ("fc_out",nn.Linear(total_size // (8 * reduction), total_size // reduction))
            ]))
            self.fc2 = nn.Sequential(OrderedDict([
                ("fc_in", nn.Linear(total_size // reduction, total_size // (16 * reduction))),
                ("fc_out",nn.Linear(total_size // (16 * reduction), total_size // reduction))
            ]))
        self.identity = nn.Identity()
    def forward(self, x):
        if self.bottle == 'b':
            path1 = self.fc(x)
        elif self.bottle == 'pb':
            path11 = self.fc1(x)
            path12 = self.fc2(x)
            path1 = path11 + path12
        path2 = self.identity(x)
        out = path1 + path2
        return out


class CRNet(nn.Module):
    def __init__(self, reduction=4, nbit=6, bottle='b'):
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

        self.quant = Quantization(nbit=nbit)
        self.quant_sigmoid = nn.Sigmoid()

        self.identity_codewordQ = nn.Identity()
        self.identity_codeword = nn.Identity()
 
        self.adaptor = Adaptor(reduction=reduction,bottle=bottle)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))
        out = self.quant_sigmoid(out)

        codeword = self.identity_codeword(out)
        out, SNR = self.quant(out) 
        out = self.adaptor(out)    
        codewordQ = self.identity_codewordQ(out)            
        SNR = torch.mean(torch.norm(codeword,p=2,dim=1)/torch.norm(codeword-codewordQ,p=2,dim=1))
        
        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out) 
        
        return out, codeword, codewordQ, SNR



def crnet(reduction=4, nbit=6, bottle='b'):
    model = CRNet(reduction=reduction,
                nbit=nbit,
                bottle=bottle)
    return model

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.archs.base_network import BaseNetwork
from models.archs.normalization import get_nonspade_norm_layer
from models.archs.arch_util import LayerNorm2d
import torch
import pdb


class ResnetBlock(nn.Module):
    def __init__(self, dim, act, kernel_size=3):
        super().__init__()
        self.act = act
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            act,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return self.act(out)


class ConvEncoder(BaseNetwork):
    # Same architecture as the image discriminator

    def __init__(self, embedding_dim):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        self.D = embedding_dim
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)

        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 16, ndf * 16, kw, stride=2, padding=pw))
       
        self.res_0 = ResnetBlock(ndf*16, nn.LeakyReLU(0.2, False))
        self.so = s0 = 4
        activation=nn.LeakyReLU(0.2,False)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ndf * 16, ndf * 16, kernel_size=kw)),
            activation
        )

        self.down = nn.AvgPool2d(2,2)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

        self.fc1 = nn.Linear(embedding_dim*8, embedding_dim)


    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.layer1(x)  
   
        x = self.conv_7x7(self.pad_3(self.actvn(x))) 
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x)) 
        x = self.layer4(self.actvn(x)) 
        x = self.layer5(self.actvn(x)) 
        x = self.layer6(self.actvn(x)) 

        x = self.res_0(x) 
        mu = self.out(x) 

        mu = mu.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        mu_shape = mu.shape
        flat_mu = mu.view(mu.size(0),-1) 
        flat_mu = self.fc1(flat_mu) 
        flat_mu = flat_mu.view(-1, self.D) 


        return flat_mu, mu_shape

class ConvEncoderLoss(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.out = norm_layer(nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=0))
        self.down = nn.AvgPool2d(2,2)
        # self.global_avg = nn.AdaptiveAvgPool2d((6,6))
        
        

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x):

        x1 = self.layer1(x) # 128
        x2 = self.conv_7x7(self.pad_3(self.actvn(x1)))
        x3 = self.layer2(self.actvn(x2)) # 64
        x4 = self.layer3(self.actvn(x3)) # 32
        x5 = self.layer4(self.actvn(x4)) # 16 
        return [x1, x2, x3, x4, x5]
        
class EncodeMap(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.layer_final = nn.Conv2d(ndf * 8, ndf * 16, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #     x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        return self.layer_final(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
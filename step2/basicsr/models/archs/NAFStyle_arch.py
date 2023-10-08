import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from copy import deepcopy
from collections import OrderedDict
from models.archs.arch_util import LayerNorm2d
from models.archs.local_arch import Local_Base
from models.archs.encoder import ConvEncoder
from models.archs.architecture import STYLEResnetBlock as STYLEResnetBlock
from models.archs.loss import InfoNCE
import pdb

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

"""
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
"""

class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2,False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        '''self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation)'''
        # norm_layer = 
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation
        )
        

    def forward(self, x):
        # conv1 = self.conv1(x)
        y = self.conv_block(x)
        return y

"""
# vq原始版本
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K) # [16,256]

    def forward(self, latents):
        # latents[1,2048,8,8]
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [1,8,8,2048][B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # 展平特征，[64,2048], [BHW x D]

        # Compute L2 distance between latents and embedding weights 计算距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [64,4] [z_e(x)]^2+e^2-2*z_e(x)*e [BHW x K] 平方差公式
        # self.embedding.weight[4,2048]

        # Get the encoding that has the min distance 得到最小距离 
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [64,1] [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device) # [64,4]
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [64,4] [BHW x K]

        # Quantize the latents
        # 和独热编码相乘
        #!!!量化后的latent
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [64,2048][BHW, D] 得到量化的latent
        quantized_latents = quantized_latents.view(latents_shape)  # [1,8,8,2048] [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents) 
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach() # [1,8,8,2048]

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [1,2048,8,8] [B x D x H x W]

"""


# 方案一
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta, temperature):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.temperature = temperature

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K) # [4,2048]
        
        self.infoNCE = InfoNCE(temperature, 'mean')

    def forward(self, flat_latents, label, state):
        # latents[4,1024,4,4]

        if state=='train':
            # Convert to one-hot encodings
            device = flat_latents.device
            encoding_one_hot = torch.zeros(label.size(0), self.K, device=device) # [2,4]
            #temp = torch.ones(self.K, self.D, device=device)
            #codebook = torch.mul(temp, self.embedding.weight)
            labels = label.unsqueeze(1) # [3,2]->[[3],[2]] [2,1]
            encoding_one_hot.scatter_(1, labels, 1)  # [2,4] [BHW x K] 

            # Quantize the latents
            # 和独热编码相乘
            #!!!量化后的latent
            positive_key = torch.matmul(encoding_one_hot, self.embedding.weight)  # [2,2048][BHW, D] 得到量化的latent
            if label.size(0)!=0:
                negative_key = torch.cat((self.embedding.weight[:label[0]],self.embedding.weight[label[0]+1:]))
                negative_key = negative_key.unsqueeze(0)
            for i in range(1, label.size(0)):
                #current_negative_key = self.embedding.weight[torch.arange(self.embedding.weight.size(0)).to(device)!=label[i]]
                current_negative_key = torch.cat((self.embedding.weight[:label[i], :],self.embedding.weight[label[i]+1:, :]))
                current_negative_key = current_negative_key.unsqueeze(0)
                negative_key = torch.cat([negative_key, current_negative_key])
            #quantized_latents = positive_key.view(positive_key.size(0), -1) # [2,2048]
            # Compute the VQ Losses
            commitment_loss = F.mse_loss(positive_key.detach(), flat_latents) 
            embedding_loss = F.mse_loss(positive_key, flat_latents.detach())

            quant_loss = commitment_loss * self.beta + embedding_loss
            #混合光传输条件
            #flat_latents = flat_latents.unsqueeze(1)
            #positive_key = positive_key.unsqueeze(1)
            infoNCE_loss = self.infoNCE(flat_latents, positive_key, negative_key, self.temperature, 'mean', 'paired')
            # 单光传输条件
            #infoNCE_loss = 0

            vq_loss = quant_loss + infoNCE_loss 

            
        else:
            # 验证和测试阶段vq_loss# 测试阶段vq_loss记录分类错误的码字个数
            vq_loss = 0
            # Compute L2 distance between latents and embedding weights 计算距离
            dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [64,4] [z_e(x)]^2+e^2-2*z_e(x)*e [BHW x K] 平方差公式
            # self.embedding.weight[4,2048]

            # Get the encoding that has the min distance 得到最小距离 
            encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [64,1] [BHW, 1]
            if encoding_inds!= label:
                vq_loss += encoding_inds.size(0) # 加上label的个数

            # Convert to one-hot encodings
            device = flat_latents.device
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device) # [64,4]
            encoding_one_hot.scatter_(1, encoding_inds, 1)  # [64,4] [BHW x K]

            # Quantize the latents
            # 和独热编码相乘
            #!!!量化后的latent
            positive_key = torch.matmul(encoding_one_hot, self.embedding.weight)  # [2,2048][BHW, D] 得到量化的latent
        quantized_latents = positive_key.view(positive_key.size(0), -1) # [2,2048]


        return quantized_latents, vq_loss  # [1,1024,4,4] [B x D x H x W]


class prior_upsampling(nn.Module):
    def __init__(self, wf=64, embedding_dim=2048):
        super(prior_upsampling, self).__init__()
        # self.conv_latent_init = Up_ConvBlock(4 * wf, 32 * wf)
        self.fc4 = nn.Linear(embedding_dim, embedding_dim*8)

        self.conv_latent_up2 = Up_ConvBlock(16 * wf, 8 * wf) 
        self.conv_latent_up3 = Up_ConvBlock(8 * wf, 8 * wf)
        self.conv_latent_up4 = Up_ConvBlock(8 * wf, 8 * wf)
        self.conv_latent_up5 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up6 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up7 = Up_ConvBlock(2 * wf, wf)

    def forward(self, quantized_latents, z_shape):
        # latent_1 = self.conv_latent_init(z) # 8, 8
        # z[1024,4,4]
        quantized_latents = self.fc4(quantized_latents) # [2,32768]
        z = quantized_latents.view(z_shape)  # [2,8,8,512] [B x H x W x D]
        z = z.permute(0, 3, 1, 2).contiguous()

        latent_2 = self.conv_latent_up2(z) # 16 [1,512,8,8]
        latent_3 = self.conv_latent_up3(latent_2) # 32 [1,512,16,16]
        latent_4 = self.conv_latent_up4(latent_3) # 64 [1,512,32,32]
        latent_5 = self.conv_latent_up5(latent_4) # 128 [1,256,64,64]
        latent_6 = self.conv_latent_up6(latent_5) # 256 [1,128,128,128]
        latent_7 = self.conv_latent_up7(latent_6) # 256 [1,64,256,256]
        latent_list = [latent_7, latent_6,latent_5,latent_4,latent_3,latent_2]
        return latent_list # latent_6,latent_5,latent_4,latent_3,latent_2,latent_1
"""
class NAFBlock_de(nn.Module):
    def __init__(self, inc=64, outc=64, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
    
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(outc, momentum=0.5)
        # self.norm1 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm2 = nn.BatchNorm2d(outc,momentum=0.5)        
        #self.norm2 = nn.BatchNorm2d(outc)
        
        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=outc, out_channels=outc // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=outc // 2, out_channels=outc, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=False),
            nn.Tanh()
        )

        # GELU
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm3 = nn.BatchNorm2d(outc, momentum=0.5)
        #self.norm3 = nn.BatchNorm2d(outc)       
        self.conv5 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm4 = nn.BatchNorm2d(outc, momentum=0.5)   
        #self.norm4 = nn.BatchNorm2d(outc)       

        #self.norm1 = LayerNorm2d(c)
        #self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):

        if self.conv_expand is not None:
          identity_data = self.conv_expand(inp)
        else:
          identity_data = inp

        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        y1 = identity_data + x

        x1 = self.conv4(y1)
        x1 =self.norm3(x1)
        x1 = self.leaky_relu(x1)
        x1 = self.conv5(x1)
        x1 =self.norm4(x1)
        x1 = self.leaky_relu(x1)

        y = x1+y1

        return y

"""
# v2，保留了一些用不到的模块
class NAFBlock(nn.Module):
    def __init__(self, inc=64, outc=64, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
    
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(outc, momentum=0.5)
        self.conv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm2 = nn.BatchNorm2d(outc, momentum=0.5)
        # Channel Attention
        """
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=outc, out_channels=outc // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=outc // 2, out_channels=outc, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=False),
            nn.Tanh()
        )
        """
        self.leaky_relu = nn.LeakyReLU(0.2)
        """
        self.conv4 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm3 = nn.BatchNorm2d(outc, momentum=0.5)       
        self.conv5 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.norm4 = nn.BatchNorm2d(outc, momentum=0.5)  
        """

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):

        if self.conv_expand is not None:
          identity_data = self.conv_expand(inp)
        else:
          identity_data = inp

        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        y1 = identity_data + x
        y1 = self.norm2(y1)
        y1 = self.leaky_relu(y1)
        return y1
"""
# v1,去掉了一些用不到的模块
class NAFBlock(nn.Module):
    def __init__(self, inc=64, outc=64, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
          self.conv_expand = None
    
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(outc,momentum=0.5)
        #self.norm1 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        #self.norm2 = nn.BatchNorm2d(outc)
        self.norm2 = nn.BatchNorm2d(outc,momentum=0.5)
        
        # GELU
        self.leaky_relu = nn.LeakyReLU(0.2)


        #self.norm1 = LayerNorm2d(c)
        #self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):

        if self.conv_expand is not None:
          identity_data = self.conv_expand(inp)
        else:
          identity_data = inp

        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        y1 = identity_data + x
        y1 = self.norm2(y1)
        y1 = self.leaky_relu(y1)

        return y1

"""
class NAFStyleRe(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64,  enc_blk_re=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512], norm_G='spectralspadesyncbatch3x3'):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width), # [16,64,256,256]
            nn.LeakyReLU(0.2),  # [16,64,256,256]             
            #nn.AvgPool2d(2), # [16,64,128,128]
        )

        self.ending = nn.Sequential(
            #NAFBlock(width, width),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ad1_list = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # enc_blk_nums和dec_blk_nums控制卷积block
        enc_cc = enc_blk_re[0]
        for enc_ch in enc_blk_re[0:]: 
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(enc_cc, enc_ch)]
                )
            )
            self.downs.append(
                nn.Conv2d(enc_ch, enc_ch, kernel_size=4, stride=2, padding=1, bias=False)) # 图像尺寸变为原来的一半

            enc_cc = enc_ch
        
        #self.encoders_end = NAFBlock(enc_cc, enc_cc)

        cnt = -1
        dec_cc = dec_blk[-1]
        for dec_ch in dec_blk[::-1]:
            # 卷积上采样
            cnt+=1
            if(cnt<=2):
                self.ups.append(
                    nn.ConvTranspose2d(dec_ch, dec_ch, kernel_size=2, stride=2, bias=False)) # 图像尺寸变为原来的两倍

                self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(dec_ch*2, dec_ch)]
                    )
                )

            else:
                self.ups.append(
                    nn.ConvTranspose2d(dec_cc, dec_ch, kernel_size=2, stride=2, bias=False)) # 图像尺寸变为原来的两倍

                self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(dec_cc, dec_ch)]
                    )
                )

            self.ad1_list.append(STYLEResnetBlock(dec_ch, dec_ch, norm_G))
            dec_cc = dec_ch
            
        

        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x, latent_list):
        # inp是真实图像
        # 第一次调用的结果是随机初始化的结果
        # 初始化net_G调用LocalBase，所以先随机初始化inp和z，走一遍forward流程
        image = x

        # stage 1
        x1 = self.intro(image) # [1,64,256,256]
        encs = []
        decs = []
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) # x1[0]=[1,64,256,256] x1[1]=[1,128,128,128] x1[2]=[1,256,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8] 
            encs.append(x1) # encs[0]=[1,64,256,256] encs[1]=[1,128,128,128] encs[2]=[1,256,64,64] encs[3]=[1,512,32,32] encs[4]=[1,512,16,16] encs[5]=[1,512,8,8]
            x1 = down(x1) # x1[0]=[1,128,128,128] x1[1]=[1,256,64,64] x1[2]=[1,512,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8] x1[5]=[1,512,4,4]
        #x1 = self.encoders_end(x1)

        for decoder, up, ad1, enc_skip, latent in zip(self.decoders, self.ups, self.ad1_list, encs[::-1], latent_list[::-1]):
            temps2 = ad1(enc_skip, latent) # temps2[0]=[1,512,8,8] temps2[1]=[1,512,16,16] temps2[2]=[1,512,32,32] temps2[3]=[1,256,64,64] temps2[4]=[1,128,128,128] temps2[5]=[1,64,256,256]
            up1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,16,16] x1[2]=[1,512,32,32] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[1,64,256,256]
            x1 = torch.cat([up1, temps2], 1) # [1024,8,8] [1024,16,16] [1024,32,32] [512,64,64] [256,128,128] [128,256,256]
            x1 = decoder(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,16,16] x1[2]=[1,512,32,32] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[1,64,256,256]
        out = self.ending(x1)
        # 此处可以直接输出out
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

"""                
# v1
class NAFStyleDe(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk_de=[128, 256, 512, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512], norm_G='spectralspadesyncbatch3x3'):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width), # [16,64,256,256]

            nn.LeakyReLU(0.2),  # [16,64,256,256]             
            #nn.AvgPool2d(2), # [16,64,128,128]
        )

        self.ending = nn.Sequential(
            #NAFBlock(width, width),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ad1_list = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # enc_blk_nums和dec_blk_nums控制卷积block
        enc_cc = width
        cnt = -1
        for enc_ch in enc_blk_de[0:]: 
            cnt += 1
            if(cnt<=0):
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock_de(enc_cc, enc_ch)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock_de(enc_cc*2, enc_ch)]
                    )
                )

            self.downs.append(
                nn.AvgPool2d(2)
            )
            if(cnt<=4):
                self.ad1_list.append(STYLEResnetBlock(enc_ch, enc_ch, norm_G))
            enc_cc = enc_ch
        
        self.encoders_end = NAFBlock_de(enc_cc, enc_cc)

        self.fc = nn.Linear(enc_cc*16, enc_cc*2)
        self.fc2 = nn.Linear(enc_cc*2, enc_cc)
        
        dec_cc = dec_blk[-1]
        self.fc3 = nn.Sequential(
                      nn.Linear(dec_cc, dec_cc*4*4),
                      nn.ReLU(True),
                  )
        dec_cc = dec_blk[-1]
        for dec_ch in dec_blk[::-1]:
            self.decoders.append(
            nn.Sequential(
                *[NAFBlock_de(dec_cc, dec_ch)]
                )
            )

            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            dec_cc = dec_ch
        

        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x, latent_list):
        # inp是真实图像
        # 第一次调用的结果是随机初始化的结果
        # 初始化net_G调用LocalBase，所以先随机初始化inp和z，走一遍forward流程

        image = x

        # stage 1
        x1 = self.intro(image) # [1,64,256,256]
        encs = []
        decs = []
        x1 = self.encoders[0](x1) # [128,256,256]
        x1 = self.downs[0](x1) # [128,128,128]
        for encoder, ad1, latent, down in zip(self.encoders[1:], self.ad1_list[:], latent_list[1:], self.downs[1:]):
            # latent: [1,512,8,8] [1,512,16,16] [1,512,32,32] [1,256,64,64] [1,128,128,128] [1,64,256,256]
            temps2 = ad1(x1, latent) # x1[0]=[1,128,128,128] x1[1]=[1,256,64,64] x1[2]=[1,512,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8]            
            x1 = torch.cat([x1, temps2], 1)  # [256,128,128] [512,64,64] [1024,32,32] [1024,16,16] [1024,8,8]      
            x1 = encoder(x1) # x1[0]=[1,256,128,128] x1[1]=[1,512,64,64] x1[2]=[1,512,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8]
            x1 = down(x1) # [256,64,64] [256,64,64] [512,32,32] [512,16,16] [512,4,4]
        # temps2.shape x1.shape
        #x1 = self.encoders_end(x1) # [1,512,4,4]

        latent = x1.view(x.size(0), -1) # [1,8192]
        latent = self.fc(latent) # [1,1024]
        latent = self.fc2(latent) # [1,512]

        z = self.fc3(latent) # [1,8192]
        x1 = z.view(latent.size(0), -1, 4, 4) # [1,512,4,4]

        for decoder, up in zip(self.decoders, self.ups):   
            x1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,8,8] x1[2]=[1,512,16,16] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[64,256,256]              
            x1 = decoder(x1) 
        
        out = self.ending(x1)
        # 此处可以直接输出out
        return out, latent

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


"""
# v2
class NAFStyleDe(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk_de=[128, 256, 512, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512], norm_G='spectralspadesyncbatch3x3'):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2),  # [16,64,256,256]             
            #nn.AvgPool2d(2), # [16,64,128,128]
        )

        self.ending = nn.Sequential(
            #NAFBlock(width, width),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ad1_list = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # enc_blk_nums和dec_blk_nums控制卷积block
        enc_cc = width
        cnt = -1
        for enc_ch in enc_blk_de[0:]: 
            cnt += 1
            if(cnt<=0):
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(enc_cc, enc_ch)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(enc_cc*2, enc_ch)]
                    )
                )
            """
            self.downs.append(
                nn.AvgPool2d(2)
            )
            """
            
            
            self.downs.append(
                nn.Conv2d(enc_ch, enc_ch, kernel_size=4, stride=2, padding=1, bias=False)) # 图像尺寸变为原来的一半
            

            if(cnt<=4):
                self.ad1_list.append(STYLEResnetBlock(enc_ch, enc_ch, norm_G))
            enc_cc = enc_ch
        
        #self.encoders_end = NAFBlock(enc_cc, enc_cc)

        self.fc = nn.Linear(enc_cc*16, enc_cc*2)
        self.fc2 = nn.Linear(enc_cc*2, enc_cc)
        
        dec_cc = dec_blk[-1]
        self.fc3 = nn.Sequential(
                      nn.Linear(dec_cc, dec_cc*4*4),
                      nn.ReLU(True),
                  )
        dec_cc = dec_blk[-1]
        for dec_ch in dec_blk[::-1]:
            """
            self.ups.append(
                nn.ConvTranspose2d(dec_cc, dec_cc, kernel_size=2, stride=2, bias=False)) # 图像尺寸变为原来的两倍
            """
            self.decoders.append(
            nn.Sequential(
                *[NAFBlock(dec_cc, dec_ch)]
                )
            )
            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )


            dec_cc = dec_ch
        

        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x, latent_list):
        # inp是真实图像
        # 第一次调用的结果是随机初始化的结果
        # 初始化net_G调用LocalBase，所以先随机初始化inp和z，走一遍forward流程

        image = x

        # stage 1
        x1 = self.intro(image) # [1,64,256,256]
        encs = []
        decs = []
        x1 = self.encoders[0](x1) # [128,256,256]
        x1 = self.downs[0](x1) # [128,128,128]
        for encoder, ad1, latent, down in zip(self.encoders[1:], self.ad1_list[:], latent_list[1:], self.downs[1:]):
            # latent: [1,512,8,8] [1,512,16,16] [1,512,32,32] [1,256,64,64] [1,128,128,128] [1,64,256,256]
            temps2 = ad1(x1, latent) # x1[0]=[1,128,128,128] x1[1]=[1,256,64,64] x1[2]=[1,512,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8]            
            x1 = torch.cat([x1, temps2], 1)  # [256,128,128] [512,64,64] [1024,32,32] [1024,16,16] [1024,8,8]      
            x1 = encoder(x1) # x1[0]=[1,256,128,128] x1[1]=[1,512,64,64] x1[2]=[1,512,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8]
            x1 = down(x1) # [256,64,64] [512,32,32] [512,16,16] [512,8,8] [512,4,4]
        # temps2.shape x1.shape
        #x1 = self.encoders_end(x1)
        latent = x1.view(x.size(0), -1) # [1,8192]
        latent = self.fc(latent) # [1,1024]
        latent = self.fc2(latent) # [1,512]

        z = self.fc3(latent) # [1,8192]
        x1 = z.view(latent.size(0), -1, 4, 4) # [1,512,4,4]

        for decoder, up in zip(self.decoders, self.ups):   
            x1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,8,8] x1[2]=[1,512,16,16] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[64,256,256]              
            x1 = decoder(x1) 
        
        out = self.ending(x1)
        # 此处可以直接输出out
        return out, latent

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class NAFStyleLocal(nn.Module):
    def __init__(self, img_channel=3, wf=64, width=64, num_embeddings=8, embedding_dim=2048, beta=0.25, temperature=0.1, enc_blk_re=[], enc_blk_de=[], dec_blk=[], norm_G='spectralspadesyncbatch3x3'):
        super(NAFStyleLocal, self).__init__()
        state_dict = torch.load("/share1/home/zhangjiarui/Projects/nafnet-OT/step1/experiments/NAFNet-STL10-OT-step1-1-0819/models/net_g_64970.pth")['params']
        #print(type(state_dict))
        dir_name = "./checkpoints/NAFNet-STL10-OT-step1-1-0819/"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        self.prior_upsampling = prior_upsampling(embedding_dim=embedding_dim)

        self.net_prior = ConvEncoder(embedding_dim=embedding_dim)

        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta, temperature=temperature)
                    
        self.inverse_generator = NAFStyleRe(img_channel=3, wf=wf, width=width, enc_blk_re=enc_blk_re, dec_blk=dec_blk, norm_G=norm_G)
        self.generator = NAFStyleDe(img_channel=3, wf=wf, width=width, enc_blk_de=enc_blk_de, dec_blk=dec_blk, norm_G=norm_G)
        generator_fc3_dict = OrderedDict()
        generator_decoders_dict = OrderedDict()
        generator_ending_dict = OrderedDict()

        for k, v in deepcopy(state_dict).items():
            if k.startswith('generator.fc3.'):
                generator_fc3_dict[k[len('generator.fc3.'):]] = v
            # 使用Upsample的话上采样层没有参数要学
            # 使用卷积上采样的话有参数要学
            if k.startswith('generator.decoders.'):
                generator_decoders_dict[k[len('generator.decoders.'):]] = v
            if k.startswith('generator.ending.'):
                generator_ending_dict[k[len('generator.ending.'):]] = v
        self.generator.fc3.load_state_dict(generator_fc3_dict, strict=True)
        self.generator.decoders.load_state_dict(generator_decoders_dict, strict=False)    
        self.generator.ending.load_state_dict(generator_ending_dict, strict=False)    
        
        torch.save(self.generator.fc3.state_dict(), dir_name+"generator_fc3.pth")
        torch.save(self.generator.decoders.state_dict(), dir_name+"generator_decoders.pth")
        torch.save(self.generator.ending.state_dict(), dir_name+"generator_ending.pth")

        # 固定prior_upsampling和net_prior
        for k,v in self.generator.fc3.named_parameters():
            v.requires_grad=False
        for k,v in self.generator.decoders.named_parameters():
            v.requires_grad=False
        for k,v in self.generator.ending.named_parameters():
            v.requires_grad=False


        # del state_dict
        del generator_fc3_dict
        del generator_decoders_dict
        del generator_ending_dict

        #del generator_dict 
        torch.cuda.empty_cache()
       


    def forward(self, x, y, label, state):
        prior_z, z_shape = self.net_prior(x) # 谱归一化, 通过全连接层降维到[1,2048] [1024,4,4]
        quantized_inputs, vq_loss = self.vq_layer(prior_z, label, state) # quantized_inputs[1024,4,4]
        latent_list_inverse = self.prior_upsampling(quantized_inputs, z_shape) # 谱归一化
        # 线性映射
        if (y!=None):
            out_inverse = self.inverse_generator(y, latent_list_inverse)
        else: 
            out_inverse = None
        out, latent = self.generator(x, latent_list_inverse)
        
        # 卷积
        #out_inverse = self.inverse_generator(y, latent_list_inverse)
        #out = self.generator(x, latent_list_inverse)
        return out, out_inverse, latent, vq_loss


    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


"""
# v1

class NAFStyle(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512]):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2),  # [16,64,256,256]             
            #nn.AvgPool2d(2), # [16,64,128,128]
        )

        self.ending = nn.Sequential(
            #NAFBlock_de(width, width),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # enc_blk_nums和dec_blk_nums控制卷积block
        enc_cc = enc_blk[0]
        for enc_ch in enc_blk[0:]: 
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock_de(enc_cc, enc_ch)]
                )
            )
            self.downs.append(
                nn.AvgPool2d(2)
            )

            enc_cc = enc_ch
        
        self.encoders_end = NAFBlock_de(enc_cc, enc_cc)

        
        self.fc = nn.Linear(enc_cc*16, enc_cc*2)
        self.fc2 = nn.Linear(enc_cc*2, enc_cc)
        
        dec_cc = dec_blk[-1]
        self.fc3 = nn.Sequential(
                      nn.Linear(dec_cc, dec_cc*4*4),
                      nn.ReLU(True),
                  )

        for dec_ch in dec_blk[::-1]:
            self.decoders.append(
            nn.Sequential(
                *[NAFBlock_de(dec_cc, dec_ch)]
                )
            )

            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )

            dec_cc = dec_ch
        


        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x):
        image = x

        x1 = self.intro(image) # [1,64,128,128]
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) # x1[0]=[1,64,256,256] x1[1]=[1,128,128,128] x1[2]=[1,256,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8]
            x1 = down(x1) # x1[0]=[1,128,128,128] x1[1]=[1,256,64,64] x1[2]=[1,512,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8] x1[5]=[1,512,4,4] 
        
        #x1 = self.encoders_end(x1)

        latent = x1.view(x.size(0), -1) # [1,8192]
        latent = self.fc(latent) # [1,1024]
        latent = self.fc2(latent) # [1,512]

        z = self.fc3(latent) # [1,8192]
        x1 = z.view(latent.size(0), -1, 4, 4) # [1,512,4,4]

        for decoder, up in zip(self.decoders, self.ups):
            x1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,8,8] x1[2]=[1,512,16,16] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[64,256,256]              
            x1 = decoder(x1) 

        out = self.ending(x1)
        
        return out, latent

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
"""
# v2
class NAFStyle(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512]):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width), # [16,64,256,256]
            nn.LeakyReLU(0.2),  # [16,64,256,256]             
            #nn.AvgPool2d(2), # [16,64,128,128]
        )

        self.ending = nn.Sequential(
            #NAFBlock(width, width),
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # enc_blk_nums和dec_blk_nums控制卷积block
        enc_cc = enc_blk[0]
        for enc_ch in enc_blk[0:]: 
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(enc_cc, enc_ch)]
                )
            )
            self.downs.append(
                nn.AvgPool2d(2)
            )
            """
            self.downs.append(
                nn.Conv2d(enc_ch, enc_ch, kernel_size=4, stride=2, padding=1, bias=False)) # 图像尺寸变为原来的一半
            """
            enc_cc = enc_ch
        
        #self.encoders_end = NAFBlock(enc_cc, enc_cc)

        
        self.fc = nn.Linear(enc_cc*16, enc_cc*2)
        self.fc2 = nn.Linear(enc_cc*2, enc_cc)
        
        dec_cc = dec_blk[-1]
        self.fc3 = nn.Sequential(
                      nn.Linear(dec_cc, dec_cc*4*4),
                      nn.ReLU(True),
                  )

        for dec_ch in dec_blk[::-1]:
            """
            self.ups.append(
                nn.ConvTranspose2d(dec_cc, dec_cc, kernel_size=2, stride=2, bias=False)) # 图像尺寸变为原来的两倍
            """

            self.decoders.append(
            nn.Sequential(
                *[NAFBlock(dec_cc, dec_ch)]
                )
            )
            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )
            dec_cc = dec_ch
        


        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x):
        image = x

        x1 = self.intro(image) # [1,64,128,128]
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) # x1[0]=[1,64,256,256] x1[1]=[1,128,128,128] x1[2]=[1,256,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8]
            x1 = down(x1) # x1[0]=[1,128,128,128] x1[1]=[1,256,64,64] x1[2]=[1,512,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8] x1[5]=[1,512,4,4] 
        
        #x1 = self.encoders_end(x1)
        latent = x1.view(x.size(0), -1) # [1,8192]
        latent = self.fc(latent) # [1,1024]
        latent = self.fc2(latent) # [1,512]

        z = self.fc3(latent) # [1,8192]
        x1 = z.view(latent.size(0), -1, 4, 4) # [1,512,4,4]

        for decoder, up in zip(self.decoders, self.ups):
            x1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,8,8] x1[2]=[1,512,16,16] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[64,256,256]              
            x1 = decoder(x1) 

        out = self.ending(x1)
        
        return out, latent

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class NAFLocal(nn.Module):
    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[], dec_blk=[]):
        super(NAFLocal, self).__init__()

                    
        self.generator = NAFStyle(img_channel=3, wf=wf, width=width, enc_blk=enc_blk, dec_blk=dec_blk)
        
        
    def forward(self, x):
        out, latent = self.generator(x)
        
        return out, latent

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)



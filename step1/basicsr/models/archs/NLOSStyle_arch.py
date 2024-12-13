import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from models.archs.arch_util import LayerNorm2d
from models.archs.local_arch import Local_Base
from models.archs.encoder import ConvEncoder
from models.archs.architecture import STYLEResnetBlock as STYLEResnetBlock
import pdb

class NLOSBlock(nn.Module):
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
        
        self.leaky_relu = nn.LeakyReLU(0.2)


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

class NLOSStyle(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512]):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), # [16,64,256,256]
            nn.BatchNorm2d(width, momentum=0.5), # [16,64,256,256]
            nn.LeakyReLU(0.2),  # [16,64,256,256]             
        )

        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        enc_cc = width
        for enc_ch in enc_blk[0:]: 
            self.encoders.append(
                nn.Sequential(
                    *[NLOSBlock(enc_cc, enc_ch)]
                )
            )
            self.downs.append(
                nn.AvgPool2d(2)
            )


            enc_cc = enc_ch
        

        
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
                *[NLOSBlock(dec_cc, dec_ch)]
                )
            )
            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )


            dec_cc = dec_ch
        


        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x):
        image = x

        x1 = self.intro(image) # [1,64,256,256]
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) # x1[0]=[1,128,256,256] x1[1]=[1,128,128,128] x1[2]=[1,256,64,64] x1[3]=[1,512,32,32] x1[4]=[1,512,16,16] x1[5]=[1,512,8,8]
            x1 = down(x1) # x1[0]=[1,64,128,128] x1[1]=[1,128,64,64] x1[2]=[1,256,32,32] x1[3]=[1,512,16,16] x1[4]=[1,512,8,8] x1[5]=[1,512,4,4] 
            #print("x1 shape", x1.shape)  # [1, 64, 256, 256] [128,128,128] [256,64,64] [512,32,32] [512,16,16] [512,8,8]
        

        latent = x1.view(x.size(0), -1) # [1,8192]
        latent = self.fc(latent) # [1,1024]
        latent = self.fc2(latent) # [1,512]

        z = self.fc3(latent) # [1,8192]
        x1 = z.view(latent.size(0), -1, 4, 4) # [1,512,4,4]

        for decoder, up in zip(self.decoders, self.ups):
            x1 = up(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,16,16] x1[2]=[1,512,32,32] x1[3]=[1,512,64,64] x1[4]=[1,256,128,128] x1[5]=[128,256,256]  
            x1 = decoder(x1) # x1[0]=[1,512,8,8] x1[1]=[1,512,16,16] x1[2]=[1,512,32,32] x1[3]=[1,256,64,64] x1[4]=[1,128,128,128] x1[5]=[64,256,256]              

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


class NLOSStyleLocal(nn.Module):
    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[], dec_blk=[]):
        super(NLOSStyleLocal, self).__init__()

                    
        self.generator = NLOSStyle(img_channel=3, wf=wf, width=width, enc_blk=enc_blk, dec_blk=dec_blk)
        
        
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



if __name__ == '__main__':
    img_channel = 3
    width = 32


    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NLOSStyle(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

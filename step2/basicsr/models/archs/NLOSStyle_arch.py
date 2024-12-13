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


class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2,False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
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
        y = self.conv_block(x)
        return y


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

        if state=='train':
            # Convert to one-hot encodings
            device = flat_latents.device
            encoding_one_hot = torch.zeros(label.size(0), self.K, device=device) # [2,4]
            labels = label.unsqueeze(1) # [3,2]->[[3],[2]] [2,1]
            encoding_one_hot.scatter_(1, labels, 1)  # [2,4] [BHW x K] 

            # Quantize the latents
            positive_key = torch.matmul(encoding_one_hot, self.embedding.weight)  
            if label.size(0)!=0:
                negative_key = torch.cat((self.embedding.weight[:label[0]],self.embedding.weight[label[0]+1:]))
                negative_key = negative_key.unsqueeze(0)
            for i in range(1, label.size(0)):
                current_negative_key = torch.cat((self.embedding.weight[:label[i], :],self.embedding.weight[label[i]+1:, :]))
                current_negative_key = current_negative_key.unsqueeze(0)
                negative_key = torch.cat([negative_key, current_negative_key])
            # Compute the VQ Losses
            commitment_loss = F.mse_loss(positive_key.detach(), flat_latents) 
            embedding_loss = F.mse_loss(positive_key, flat_latents.detach())

            quant_loss = commitment_loss * self.beta + embedding_loss
            infoNCE_loss = self.infoNCE(flat_latents, positive_key, negative_key, self.temperature, 'mean', 'paired')

            vq_loss = quant_loss + infoNCE_loss 

            
        else:
            # During the validation and testing phase, vq_loss records the number of codebook entries (codewords) that are misclassified.
            vq_loss = 0
            # Compute L2 distance between latents and embedding weights 
            dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())  #  [z_e(x)]^2+e^2-2*z_e(x)*e [BHW x K] 

            # Get the encoding that has the min distance 
            encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1) 
            if encoding_inds!= label:
                vq_loss += encoding_inds.size(0) 

            # Convert to one-hot encodings
            device = flat_latents.device
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device) 
            encoding_one_hot.scatter_(1, encoding_inds, 1)  

            # Quantize the latents
            positive_key = torch.matmul(encoding_one_hot, self.embedding.weight)  
        quantized_latents = positive_key.view(positive_key.size(0), -1)


        return quantized_latents, vq_loss  


class prior_upsampling(nn.Module):
    def __init__(self, wf=64, embedding_dim=2048):
        super(prior_upsampling, self).__init__()
        self.fc4 = nn.Linear(embedding_dim, embedding_dim*8)

        self.conv_latent_up2 = Up_ConvBlock(16 * wf, 8 * wf) 
        self.conv_latent_up3 = Up_ConvBlock(8 * wf, 8 * wf)
        self.conv_latent_up4 = Up_ConvBlock(8 * wf, 8 * wf)
        self.conv_latent_up5 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up6 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up7 = Up_ConvBlock(2 * wf, wf)

    def forward(self, quantized_latents, z_shape):
        quantized_latents = self.fc4(quantized_latents) 
        z = quantized_latents.view(z_shape)  # [B x H x W x D]
        z = z.permute(0, 3, 1, 2).contiguous()

        latent_2 = self.conv_latent_up2(z) 
        latent_3 = self.conv_latent_up3(latent_2) 
        latent_4 = self.conv_latent_up4(latent_3)
        latent_5 = self.conv_latent_up5(latent_4) 
        latent_6 = self.conv_latent_up6(latent_5) 
        latent_7 = self.conv_latent_up7(latent_6) 
        latent_list = [latent_7, latent_6,latent_5,latent_4,latent_3,latent_2]
        return latent_list 

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

class NLOSStyleRe(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64,  enc_blk_re=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512], norm_G='spectralspadesyncbatch3x3'):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), 
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2),          
        )

        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ad1_list = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        enc_cc = enc_blk_re[0]
        for enc_ch in enc_blk_re[0:]: 
            self.encoders.append(
                nn.Sequential(
                    *[NLOSBlock(enc_cc, enc_ch)]
                )
            )
            self.downs.append(
                nn.Conv2d(enc_ch, enc_ch, kernel_size=4, stride=2, padding=1, bias=False)) 

            enc_cc = enc_ch
        

        cnt = -1
        dec_cc = dec_blk[-1]
        for dec_ch in dec_blk[::-1]:
            cnt+=1
            if(cnt<=2):
                self.ups.append(
                    nn.ConvTranspose2d(dec_ch, dec_ch, kernel_size=2, stride=2, bias=False)) 

                self.decoders.append(
                nn.Sequential(
                    *[NLOSBlock(dec_ch*2, dec_ch)]
                    )
                )

            else:
                self.ups.append(
                    nn.ConvTranspose2d(dec_cc, dec_ch, kernel_size=2, stride=2, bias=False)) 

                self.decoders.append(
                nn.Sequential(
                    *[NLOSBlock(dec_cc, dec_ch)]
                    )
                )

            self.ad1_list.append(STYLEResnetBlock(dec_ch, dec_ch, norm_G))
            dec_cc = dec_ch
            
        

        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x, latent_list):
        image = x

        # stage 1
        x1 = self.intro(image) # [1,64,256,256]
        encs = []
        decs = []
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) 
            encs.append(x1) 
            x1 = down(x1) 

        for decoder, up, ad1, enc_skip, latent in zip(self.decoders, self.ups, self.ad1_list, encs[::-1], latent_list[::-1]):
            temps2 = ad1(enc_skip, latent) 
            up1 = up(x1) 
            x1 = torch.cat([up1, temps2], 1) 
            x1 = decoder(x1) 
        out = self.ending(x1)
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

class NLOSStyleDe(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk_de=[128, 256, 512, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512], norm_G='spectralspadesyncbatch3x3'):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False), 
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2),          
        )

        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=False),
            nn.Tanh()
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ad1_list = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        enc_cc = width
        cnt = -1
        for enc_ch in enc_blk_de[0:]: 
            cnt += 1
            if(cnt<=0):
                self.encoders.append(
                    nn.Sequential(
                        *[NLOSBlock(enc_cc, enc_ch)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NLOSBlock(enc_cc*2, enc_ch)]
                    )
                )
            
            
            self.downs.append(
                nn.Conv2d(enc_ch, enc_ch, kernel_size=4, stride=2, padding=1, bias=False)) 
            

            if(cnt<=4):
                self.ad1_list.append(STYLEResnetBlock(enc_ch, enc_ch, norm_G))
            enc_cc = enc_ch
        

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
                *[NLOSBlock(dec_cc, dec_ch)]
                )
            )
            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest')
            )


            dec_cc = dec_ch
        

        self.padder_size = 2 ** len(self.encoders)



    def forward(self, x, latent_list):

        image = x

        # stage 1
        x1 = self.intro(image) 
        encs = []
        decs = []
        x1 = self.encoders[0](x1) 
        x1 = self.downs[0](x1) 
        for encoder, ad1, latent, down in zip(self.encoders[1:], self.ad1_list[:], latent_list[1:], self.downs[1:]):
            temps2 = ad1(x1, latent)        
            x1 = torch.cat([x1, temps2], 1)     
            x1 = encoder(x1) 
            x1 = down(x1) 
        latent = x1.view(x.size(0), -1) 
        latent = self.fc(latent) 
        latent = self.fc2(latent) 

        z = self.fc3(latent)
        x1 = z.view(latent.size(0), -1, 4, 4) 

        for decoder, up in zip(self.decoders, self.ups):   
            x1 = up(x1)             
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

class NLOSStyleLocal(nn.Module):
    def __init__(self, img_channel=3, wf=64, width=64, num_embeddings=8, embedding_dim=2048, beta=0.25, temperature=0.1, enc_blk_re=[], enc_blk_de=[], dec_blk=[], norm_G='spectralspadesyncbatch3x3'):
        super(NLOSStyleLocal, self).__init__()
        state_dict = torch.load("./NLOS-LTM/step1/experiments/Supermodel/models/net_g.pth")['params']
        dir_name = "./checkpoints/Supermodel/"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        self.prior_upsampling = prior_upsampling(embedding_dim=embedding_dim)

        self.net_prior = ConvEncoder(embedding_dim=embedding_dim)

        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta, temperature=temperature)
                    
        self.inverse_generator = NLOSStyleRe(img_channel=3, wf=wf, width=width, enc_blk_re=enc_blk_re, dec_blk=dec_blk, norm_G=norm_G)
        self.generator = NLOSStyleDe(img_channel=3, wf=wf, width=width, enc_blk_de=enc_blk_de, dec_blk=dec_blk, norm_G=norm_G)
        generator_fc3_dict = OrderedDict()
        generator_decoders_dict = OrderedDict()
        generator_ending_dict = OrderedDict()

        for k, v in deepcopy(state_dict).items():
            if k.startswith('generator.fc3.'):
                generator_fc3_dict[k[len('generator.fc3.'):]] = v
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

        for k,v in self.generator.fc3.named_parameters():
            v.requires_grad=False
        for k,v in self.generator.decoders.named_parameters():
            v.requires_grad=False
        for k,v in self.generator.ending.named_parameters():
            v.requires_grad=False


        del generator_fc3_dict
        del generator_decoders_dict
        del generator_ending_dict

        torch.cuda.empty_cache()
       


    def forward(self, x, y, label, state):
        prior_z, z_shape = self.net_prior(x) 
        quantized_inputs, vq_loss = self.vq_layer(prior_z, label, state) 
        latent_list_inverse = self.prior_upsampling(quantized_inputs, z_shape) 
        if (y!=None):
            out_inverse = self.inverse_generator(y, latent_list_inverse)
        else: 
            out_inverse = None
        out, latent = self.generator(x, latent_list_inverse)
        
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


class NLOSStyle(nn.Module):

    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[64, 128, 256, 512, 512, 512], dec_blk=[64, 128, 256, 512, 512, 512]):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=False),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.2),         
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

        enc_cc = enc_blk[0]
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

        x1 = self.intro(image)
        for encoder, down in zip(self.encoders, self.downs):
            x1 = encoder(x1) 
            x1 = down(x1)
        
        latent = x1.view(x.size(0), -1) 
        latent = self.fc(latent) 
        latent = self.fc2(latent) 

        z = self.fc3(latent) 
        x1 = z.view(latent.size(0), -1, 4, 4) 

        for decoder, up in zip(self.decoders, self.ups):
            x1 = up(x1)      
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

class NLOSLocal(nn.Module):
    def __init__(self, img_channel=3, wf=64, width=64, enc_blk=[], dec_blk=[]):
        super(NLOSLocal, self).__init__()

                    
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



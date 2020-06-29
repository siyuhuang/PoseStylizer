import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
from .model_utils import *
import numpy as np


class UpNaive(nn.Module):
    def __init__(self, 
                 c_fg,
                 c_full, 
                 norm_layer,
                 padding=1,
                 first_layer=False, 
                 last_layer=False):
        super(UpNaive, self).__init__()

        if first_layer:
            layer = [nn.ConvTranspose2d(c_fg, c_fg,
                    4, stride=2, padding=padding)]
        elif last_layer:
            layer = [nn.ReflectionPad2d(3),
                   nn.Conv2d(c_full+c_fg, c_fg, 
                             kernel_size=7, padding=0)]
        else:
            layer = [nn.ConvTranspose2d(c_full+c_fg, c_fg,
                    4, stride=2, padding=padding)]
            
        layer += [norm_layer(c_fg),
                  nn.LeakyReLU(negative_slope=0.2)]
        self.layer = nn.Sequential(*layer)


    def forward(self, x):
        return self.layer(x)
                 
class UpBlock(nn.Module):
    def __init__(self,
                 res,   # resolution level
                 fmap_base,     # Overall multiplier for the number of feature maps.
                 bpt_channels,
                 norm_layer,
                 dlatent_size,   # Disentangled latent (W) dimensionality.
                 use_dropout,
                 padding,
                 dataset,
                 cut,
                 use_wscale=False,
                 use_noise=False,
                 use_pixel_norm=False,
                 use_instance_norm=True,
                 noise_input=None,        # noise
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,       # Maximum number of feature maps in any layer.
                 ):
        super(UpBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), 512)
        
        # res
        self.res = res
        
        # noise
        self.noise_input = noise_input
        
        if res == 0:  # first upsample layer
            self.cat = False
            self.image_up_sample = nn.ConvTranspose2d(
                                    bpt_channels, self.nf(res+1), 4, stride=2, padding=padding)
        else:
            self.cat = True
            self.image_up_sample = nn.ConvTranspose2d(
                        self.nf(res)+bpt_channels, self.nf(res+1), 4, stride=2, padding=padding)

        ### A Composition of AdaPN and Conv2d for foreground generation
        self.patchnorm1 = PatchNorm(self.nf(0), self.nf(res+1), 2**(res+1), dataset=dataset, cut=cut)
        self.conv1  = Conv2d(input_channels=self.nf(res+1), output_channels=self.nf(res+1),
                             kernel_size=3, use_wscale=use_wscale)
        self.patchnorm2 = PatchNorm(self.nf(0), self.nf(res+1), 2**(res+1), dataset=dataset, cut=cut)
        self.att = nn.Conv2d(self.nf(res+1)+bpt_channels, self.nf(res+1), kernel_size=3, padding=1, bias=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, pt, bpt, stat):
        if self.cat:
            x = torch.cat((pt, bpt), 1)
        else:
            x = pt
        x = self.image_up_sample(x)
        residual = x
        y = self.patchnorm1(x, stat, norm="InstanceNorm")
        y = self.conv1(y)
        y = self.patchnorm2(y, stat, norm="InstanceNorm") 
        bpt_upsample = nn.functional.interpolate(bpt, size=(y.shape[2],y.shape[3]), mode='bilinear')
        att = F.sigmoid(self.att(torch.cat((y, bpt_upsample), 1)))
        y = y * att
        y = y + residual
        if self.use_dropout:
            y = self.dropout(y)
        return y

class DownBlock(nn.Module):
    def __init__(self, dim_fg, padding_type, norm_layer, use_bias, cated_stream=1):
        super(DownBlock, self).__init__()
        self.conv_block_down_stream1, self.conv_block_stream1 = self.build_conv_block(dim_fg, padding_type, norm_layer, use_bias, cal_att=False, cated_stream=1)
        self.conv_block_down_stream2, self.conv_block_stream2 = self.build_conv_block(dim_fg, padding_type, norm_layer, use_bias, cal_att=True, cated_stream=1)
        
    def build_conv_block(self, dim, padding_type, norm_layer, use_bias, cated_stream=1, cal_att=False):
        dim_in = min(dim*cated_stream, 512)
        dim_inter = min(dim*2*cated_stream, 512)
        dim_out = min(dim*2, 512)
        conv_block_down = [nn.Conv2d(dim_in, dim_inter, kernel_size=3, stride=2, padding=1, bias=use_bias),
                            norm_layer(dim_inter),
                            nn.ReLU(True)]
        
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        conv_block += [nn.Conv2d(dim_inter, dim_out, kernel_size=3, padding=p, bias=use_bias)]
        if not cal_att:
            conv_block += [norm_layer(dim_out),
                          nn.ReLU(False)]

        return nn.Sequential(*conv_block_down), nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1 = self.conv_block_down_stream1(x1)
        x2 = self.conv_block_down_stream2(x2)
        
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        
        att = F.sigmoid(x2_out)
        x1_out = x1_out * att
        x1_out = x1_out + x1   # residual connection
        
        return x1_out, x2_out


class Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=None, gpu_ids=[], padding_type='reflect', n_downsampling=5, opt=None):
        super(Model, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.input_nc_s3 = input_nc[2]
        self.output_nc = output_nc
        self.ngf_fg = ngf
        self.ngf_bg = ngf // 4
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # in-node
        psf_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s1, self.ngf_fg, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(self.ngf_fg),
                    nn.ReLU(True)]
        bps_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s2, self.ngf_fg, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(self.ngf_fg),
                    nn.ReLU(True)]


        # encoder
        cated_streams = [2 for i in range(n_downsampling)]
        cated_streams[0] = 1
        down_blocks = []
        for i in range(n_downsampling):
            mult = 2**i
            dim = min(self.ngf_fg*mult, 512)
            down_blocks.append(DownBlock(dim, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias, cated_stream=1))
        
        # feats to vector
        mult = mult*2
        if opt.dataset == 'market':
            image_size = [128, 64]
        elif opt.dataset == 'fashion':
            image_size = [256, 176]
        self.dlatent_dim_fg = 512
        
        # foreground image up_sample with adain
        mult = 2**n_downsampling
        up_blocks_fg = []
        for i in range(n_downsampling):
            if opt.dataset == 'fashion' and i == n_downsampling - 4:  
                # handle feat width 6->11->22 in fashion dataset image generation
                padding = (1,2)
            else:
                padding = 1
            if i >= n_downsampling - 4:
                cut = True
            else:
                cut = False
            up_blocks_fg.append(UpBlock(res = i,   # resolution level
                                     fmap_base = self.ngf_fg * mult,      # Overall multiplier for the number of feature maps.
                                     bpt_channels=self.input_nc_s3,
                                     norm_layer = norm_layer,
                                     dlatent_size = self.dlatent_dim_fg,
                                     use_dropout = opt.dataset == 'market',
                                     padding=padding,
                                     dataset=opt.dataset,
                                     cut=cut,
                            ))
     
        
         # full image up_sample with deconv
        up_blocks_full = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            c_fg = int(self.ngf_fg * mult / 2)
            c_fg = min(c_fg, 512)
            c_full = int(self.ngf_fg * mult)
            c_full = min(c_full, 512)
            if opt.dataset == 'fashion' and i == n_downsampling - 5:  
                # handle feat width 6->11->22 in fashion dataset image generation
                padding = (1,2)
            else:
                padding = 1
            if i > 0 and i < n_downsampling - 1:
                up_blocks_full.append(UpNaive(c_fg, c_full, norm_layer, padding=padding, first_layer=False, last_layer=False))
            elif i == 0:
                up_blocks_full.append(UpNaive(c_fg, c_full, norm_layer, padding=padding, first_layer=True, last_layer=False))
            elif i == n_downsampling - 1:   # last layer, no spatial upsample
                up_blocks_full.append(UpNaive(c_fg, c_full, norm_layer, padding=padding, first_layer=False, last_layer=True))
                
        # out-node       
        out_node_full = [nn.ReflectionPad2d(3),
                         nn.Conv2d(self.ngf_fg+c_fg, output_nc, kernel_size=7, padding=0),
                         nn.Tanh()]
        
        # serialization
        self.psf_down = nn.Sequential(*psf_down)
        self.bps_down = nn.Sequential(*bps_down)
        self.down_blocks = nn.Sequential(*down_blocks)
        self.up_blocks_fg = nn.Sequential(*up_blocks_fg)
        self.up_blocks_full = nn.Sequential(*up_blocks_full)
        self.out_node_full = nn.Sequential(*out_node_full)


    def forward(self, input): 
        # here input should be a tuple
        #Person Source, Backbone Person Source, Backbone Person Target, Mask Person Source
        ps, bps, bpt = input
        
        # in-node
        psf = self.psf_down(ps)
        bps = self.bps_down(bps)
        
        # down
        for down_block in self.down_blocks:
            psf, bps = down_block(psf, bps)  
        
        # up
        ptf = nn.functional.interpolate(bpt, size=(psf.shape[2], psf.shape[3]), mode='bilinear')
        flag_first_layer = True
        i = -1
        for up_block_fg, up_block_full in zip(self.up_blocks_fg, self.up_blocks_full):
            i += 1
            bpt_down = nn.functional.interpolate(bpt, size=(ptf.shape[2],ptf.shape[3]), mode='bilinear')
            ptf = up_block_fg(ptf, bpt_down, psf)
            if flag_first_layer:
                pt = ptf
                flag_first_layer = False
            else:
                pt = torch.cat((pt, ptf), 1)
            pt = up_block_full(pt)

        # out_node
        pt = torch.cat((pt, ptf), 1)
        pt = self.out_node_full(pt)
        return pt


class stylegenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=5, opt=None):
        super(stylegenerator, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 3, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = Model(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=n_downsampling, opt=opt)

    def forward(self, input):
        return self.model(input)





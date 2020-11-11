import torch
import torch.nn as nn

from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                 Sub-Nets
# ----------------------------------------
# Global Feature Network
class GlobalFeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(GlobalFeatureExtractor, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(opt.in_channels, 64, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv2_2 = Conv2dLayer(128, 128, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv3_3 = Conv2dLayer(256, 256, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv4_3 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.pool5 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)

    def forward(self, x):
        x = self.conv1_1(x)                                         # out: batch * 64 * 256 * 256
        x = self.conv1_2(x)                                         # out: batch * 64 * 128 * 128
        x = self.conv2_1(x)                                         # out: batch * 128 * 128 * 128
        x = self.conv2_2(x)                                         # out: batch * 128 * 64 * 64
        x = self.conv3_1(x)                                         # out: batch * 256 * 64 * 64
        x = self.conv3_2(x)                                         # out: batch * 256 * 64 * 64
        x = self.conv3_3(x)                                         # out: batch * 256 * 32 * 32
        x = self.conv4_1(x)                                         # out: batch * 512 * 32 * 32
        x = self.conv4_2(x)                                         # out: batch * 512 * 32 * 32
        x = self.conv4_3(x)                                         # out: batch * 512 * 16 * 16
        x = self.conv5_1(x)                                         # out: batch * 512 * 16 * 16
        x = self.conv5_2(x)                                         # out: batch * 512 * 16 * 16
        x = self.conv5_3(x)                                         # out: batch * 512 * 16 * 16
        x = self.pool5(x)                                           # out: batch * 512 * 8 * 8
        return x

# Attention Prediction Network
class AttentionPredictionNet(nn.Module):
    def __init__(self, opt):
        super(AttentionPredictionNet, self).__init__()
        # Fusion & Upsample
        self.conv11 = TransposeConv2dLayer(opt.start_channels * 8, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.conv12 = TransposeConv2dLayer(opt.latent_channels, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Fusion & Upsample
        self.conv2 = TransposeConv2dLayer(opt.start_channels * 4, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Fusion & Upsample
        self.conv3 = Conv2dLayer(opt.start_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Fusion & Upsample
        self.conv4 = TransposeConv2dLayer(opt.latent_channels * 3, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.final = Conv2dLayer(opt.latent_channels, 1, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, x1, x2, x3):
        # in: batch * 512 * 32 * 32, out: batch * 256 * 64 * 64, batch * 128 * 128 * 128
        # initial feature transformation
        x1 = self.conv11(x1)                                        # out: batch * 128 * 64 * 64
        x1 = self.conv12(x1)                                        # out: batch * 128 * 128 * 128
        x2 = self.conv2(x2)                                         # out: batch * 128 * 128 * 128
        x3 = self.conv3(x3)                                         # out: batch * 128 * 128 * 128
        # concatenation
        x = torch.cat((x1, x2, x3), 1)                              # out: batch * 384 * 128 * 128
        # final feature mapping
        x = self.conv4(x)                                           # out: batch * 64 * 256 * 256
        x = self.final(x)                                           # out: batch * 1 * 256 * 256
        return x

# ----------------------------------------
#                Generator
# ----------------------------------------
# SCGAN's generator
class SCGAN(nn.Module):
    def __init__(self, opt):
        super(SCGAN, self).__init__()
        # Global feature extraction part of pre-trained network
        self.global_feature_network = GlobalFeatureExtractor(opt)
        # Attention prediction network
        self.attention_prediction_network = AttentionPredictionNet(opt)
        # Downsample blocks
        self.down1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down5 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down6 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down7 = Conv2dLayer(opt.start_channels * 8 + 512, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down8 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down9 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        # Fusion & Upsample
        self.up1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up5 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up6 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up7 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up8 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up9 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'tanh', norm = 'none')

    def forward(self, x):
        # Global Feature Extraction
        global_feature = self.global_feature_network(x)             # out: batch * 512 * 8 * 8
        # Mainstream Encoder
        down1 = self.down1(x)                                       # out: batch * 64 * 256 * 256
        down2 = self.down2(down1)                                   # out: batch * 128 * 128 * 128
        down3 = self.down3(down2)                                   # out: batch * 256 * 64 * 64
        down4 = self.down4(down3)                                   # out: batch * 512 * 32 * 32
        down5 = self.down5(down4)                                   # out: batch * 512 * 16 * 16
        down6 = self.down6(down5)                                   # out: batch * 512 * 8 * 8
        down6_with_gf = torch.cat((down6, global_feature), 1)       # out: batch * (1024 = 512 + 512) * 8 * 8
        down7 = self.down7(down6_with_gf)                           # out: batch * 512 * 4 * 4
        down8 = self.down8(down7)                                   # out: batch * 512 * 2 * 2
        down9 = self.down9(down8)                                   # out: batch * 512 * 1 * 1
        # Mainstream Decoder
        up1 = self.up1(down9)                                       # out: batch * 512 * 2 * 2
        up1 = torch.cat((down8, up1), 1)                            # out: batch * (1024 = 512 + 512) * 2 * 2
        up2 = self.up2(up1)                                         # out: batch * 512 * 4 * 4
        up2 = torch.cat((down7, up2), 1)                            # out: batch * (1024 = 512 + 512) * 4 * 4
        up3 = self.up3(up2)                                         # out: batch * 512 * 8 * 8
        up3 = torch.cat((down6, up3), 1)                            # out: batch * (1024 = 512 + 512) * 8 * 8
        up4 = self.up4(up3)                                         # out: batch * 512 * 16 * 16
        up4 = torch.cat((down5, up4), 1)                            # out: batch * (1024 = 512 + 512) * 16 * 16
        up5 = self.up5(up4)                                         # out: batch * 512 * 32 * 32
        up5_ = torch.cat((down4, up5), 1)                           # out: batch * (1024 = 512 + 512) * 32 * 32
        up6 = self.up6(up5_)                                        # out: batch * 256 * 64 * 64
        up6_ = torch.cat((down3, up6), 1)                           # out: batch * (512 = 256 + 256) * 64 * 64
        up7 = self.up7(up6_)                                        # out: batch * 128 * 128 * 128
        up7_ = torch.cat((down2, up7), 1)                           # out: batch * (256 = 128 + 128) * 128 * 128
        up8 = self.up8(up7_)                                        # out: batch * 64 * 256 * 256
        up8 = torch.cat((down1, up8), 1)                            # out: batch * (128 = 64 + 64) * 256 * 256
        # Colorization Prediction
        up9 = self.up9(up8)                                         # out: batch * 3 * 256 * 256
        # Saliency Map Prediction
        sal = self.attention_prediction_network(up5, up6, up7)      # out: batch * 1 * 256 * 256
        return up9, sal

class SCGAN_val(nn.Module):
    def __init__(self, opt):
        super(SCGAN_val, self).__init__()
        # Global feature extraction part of pre-trained network
        self.global_feature_network = GlobalFeatureExtractor(opt)
        # Downsample blocks
        self.down1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down5 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down6 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down7 = Conv2dLayer(opt.start_channels * 8 + 512, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down8 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down9 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        # Fusion & Upsample
        self.up1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up5 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up6 = TransposeConv2dLayer(opt.start_channels * 16, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up7 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up8 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up9 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'tanh', norm = 'none')

    def forward(self, x):
        # Global Feature Extraction
        global_feature = self.global_feature_network(x)             # out: batch * 512 * 8 * 8
        # Mainstream Encoder
        down1 = self.down1(x)                                       # out: batch * 64 * 256 * 256
        down2 = self.down2(down1)                                   # out: batch * 128 * 128 * 128
        down3 = self.down3(down2)                                   # out: batch * 256 * 64 * 64
        down4 = self.down4(down3)                                   # out: batch * 512 * 32 * 32
        down5 = self.down5(down4)                                   # out: batch * 512 * 16 * 16
        down6 = self.down6(down5)                                   # out: batch * 512 * 8 * 8
        down6_with_gf = torch.cat((down6, global_feature), 1)       # out: batch * (1024 = 512 + 512) * 8 * 8
        down7 = self.down7(down6_with_gf)                           # out: batch * 512 * 4 * 4
        down8 = self.down8(down7)                                   # out: batch * 512 * 2 * 2
        down9 = self.down9(down8)                                   # out: batch * 512 * 1 * 1
        # Mainstream Decoder
        up1 = self.up1(down9)                                       # out: batch * 512 * 2 * 2
        up1 = torch.cat((down8, up1), 1)                            # out: batch * (1024 = 512 + 512) * 2 * 2
        up2 = self.up2(up1)                                         # out: batch * 512 * 4 * 4
        up2 = torch.cat((down7, up2), 1)                            # out: batch * (1024 = 512 + 512) * 4 * 4
        up3 = self.up3(up2)                                         # out: batch * 512 * 8 * 8
        up3 = torch.cat((down6, up3), 1)                            # out: batch * (1024 = 512 + 512) * 8 * 8
        up4 = self.up4(up3)                                         # out: batch * 512 * 16 * 16
        up4 = torch.cat((down5, up4), 1)                            # out: batch * (1024 = 512 + 512) * 16 * 16
        up5 = self.up5(up4)                                         # out: batch * 512 * 32 * 32
        up5_ = torch.cat((down4, up5), 1)                           # out: batch * (1024 = 512 + 512) * 32 * 32
        up6 = self.up6(up5_)                                        # out: batch * 256 * 64 * 64
        up6_ = torch.cat((down3, up6), 1)                           # out: batch * (512 = 256 + 256) * 64 * 64
        up7 = self.up7(up6_)                                        # out: batch * 128 * 128 * 128
        up7_ = torch.cat((down2, up7), 1)                           # out: batch * (256 = 128 + 128) * 128 * 128
        up8 = self.up8(up7_)                                        # out: batch * 64 * 256 * 256
        up8 = torch.cat((down1, up8), 1)                            # out: batch * (128 = 64 + 64) * 256 * 256
        # Colorization Prediction
        up9 = self.up9(up8)                                         # out: batch * 3 * 256 * 256
        return up9

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 4, 1, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.final2 = Conv2dLayer(opt.start_channels * 8, 1, 4, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)

    def forward(self, img_A, img_B):
        # img_A: input grayscale image
        # img_B: generated color image or ground truth color image; generated weighted image or ground truth weighted image
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 4 * 256 * 256
        # Inference
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 128 * 128 * 128
        x = self.block3(x)                                      # out: batch * 256 * 64 * 64
        x = self.block4(x)                                      # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv3_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channel for U-Net decoder')
    parser.add_argument('--latent_channels', type = int, default = 128, help = 'start channel for APN')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type for generator')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type for discriminator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'intialization type for generator and discriminator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the standard deviation if Gaussian normalization')
    opt = parser.parse_args()

    '''
    net = AttentionPredictionNet(opt).cuda()
    a1 = torch.randn(1, 512, 32, 32).cuda()
    a2 = torch.randn(1, 256, 64, 64).cuda()
    a3 = torch.randn(1, 64, 128, 128).cuda()
    b = net(a1, a2, a3)
    print(b.shape)
    '''
    net = SCGAN(opt).cuda()
    #torch.save(net.state_dict(), 'test.pth')
    a = torch.randn(1, 1, 256, 256).cuda()
    b, c = net(a)
    print(b.shape)
    print(c.shape)
    loss = torch.mean(b)
    loss.backward()

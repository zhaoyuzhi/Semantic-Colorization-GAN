import torch
import torch.nn as nn
import torch.nn.init as init

from network_module import *

def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    print('Initialize network with %s type' % init_type)
    net.apply(init_func)
    
###========================== Diverse types of VGG16 framework ==========================
    # VGG16: original structure with no normalization
    # VGG16_BN: original structure with BN normalization
    # VGG16_IN: original structure with IN normalization
    # VGG16_FC: original structure with no normalization, fully convolutional
    # VGG16_BN_FC: original structure with BN normalization, fully convolutional
    # VGG16_IN_FC: original structure with IN normalization, fully convolutional

class VGG16(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool5 = nn.AdaptiveAvgPool2d((7, 7))
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 224 * 224
        x = self.pool1(x)                                   # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 112 * 112
        x = self.pool2(x)                                   # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 56 * 56
        x = self.pool3(x)                                   # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 28 * 28
        x = self.pool4(x)                                   # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x

class VGG16_BN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16_BN, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool5 = nn.AdaptiveAvgPool2d((7, 7))
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 224 * 224
        x = self.pool1(x)                                   # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 112 * 112
        x = self.pool2(x)                                   # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 56 * 56
        x = self.pool3(x)                                   # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 28 * 28
        x = self.pool4(x)                                   # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x

class VGG16_IN(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16_IN, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool5 = nn.AdaptiveAvgPool2d((7, 7))
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 224 * 224
        x = self.pool1(x)                                   # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 112 * 112
        x = self.pool2(x)                                   # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 56 * 56
        x = self.pool3(x)                                   # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 28 * 28
        x = self.pool4(x)                                   # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x

# Fully convolutional layers
class VGG16_FC(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16_FC, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.pool5 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x

class VGG16_BN_FC(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16_BN_FC, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        self.pool5 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'bn')
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x
        
class VGG16_IN_FC(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(VGG16_IN_FC, self).__init__()
        # feature extraction part
        self.conv1_1 = Conv2dLayer(in_channels, 64, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none')
        self.conv1_2 = Conv2dLayer(64, 64, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv2_1 = Conv2dLayer(64, 128, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv2_2 = Conv2dLayer(128, 128, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv3_1 = Conv2dLayer(128, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv3_2 = Conv2dLayer(256, 256, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv3_3 = Conv2dLayer(256, 256, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv4_1 = Conv2dLayer(256, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv4_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv4_3 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv5_1 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv5_2 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.conv5_3 = Conv2dLayer(512, 512, 3, 1, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        self.pool5 = Conv2dLayer(512, 512, 3, 2, 1, pad_type = 'reflect', activation = 'lrelu', norm = 'in')
        # classification part
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1_1(x)                                 # out: B * 64 * 224 * 224
        x = self.conv1_2(x)                                 # out: B * 64 * 112 * 112
        x = self.conv2_1(x)                                 # out: B * 128 * 112 * 112
        x = self.conv2_2(x)                                 # out: B * 128 * 56 * 56
        x = self.conv3_1(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_2(x)                                 # out: B * 256 * 56 * 56
        x = self.conv3_3(x)                                 # out: B * 256 * 28 * 28
        x = self.conv4_1(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_2(x)                                 # out: B * 512 * 28 * 28
        x = self.conv4_3(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_1(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_2(x)                                 # out: B * 512 * 14 * 14
        x = self.conv5_3(x)                                 # out: B * 512 * 14 * 14
        x = self.pool5(x)                                   # out: B * 512 * 7 * 7
        x = x.view(x.size(0), -1)                           # out: B * (512*7*7)
        x = self.classifier(x)                              # out: B * 1000
        return x

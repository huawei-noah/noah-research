# Obtained from  https://github.com/philipjackson

from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np

class ConvInRelu(nn.Module):
    def __init__(self,channels_in,channels_out,kernel_size,stride=1):
        super(ConvInRelu,self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv = nn.Conv2d(channels_in,channels_out,kernel_size,stride,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # x: B x C_in x H x W

        x = self.reflection_pad(x)
        x = self.conv(x) # B x C_out x H x W
        x = self.instancenorm(x) # B x C_out x H x W
        x = self.relu(x) # B x C_out x H x W
        return x


class UpsampleConvInRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, upsample, stride=1, activation=nn.ReLU):
        super(UpsampleConvInRelu, self).__init__()
        self.n_params = channels_out * 2
        self.upsample = upsample
        self.channels = channels_out

        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.fc_beta = nn.Linear(100,channels_out)
        self.fc_gamma = nn.Linear(100,channels_out)
        if activation:
            self.activation = activation(inplace=False)
        else:
            self.activation = None
        
    def forward(self, x, style):
        # x: B x C_in x H x W
        # style: B x 100

        beta = self.fc_beta(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1

        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x
        x += beta
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    # modelled after that used by Johnson et al. (2016)
    # see https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.n_params = channels * 4
        self.channels = channels

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels)
        self.fc_beta1 = nn.Linear(100,channels)
        self.fc_gamma1 = nn.Linear(100,channels)
        self.fc_beta2 = nn.Linear(100,channels)
        self.fc_gamma2 = nn.Linear(100,channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        
    def forward(self, x, style):
        # x: B x C x H x W  
        # style: B x self.n_params
        
        beta1 = self.fc_beta1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma1 = self.fc_gamma1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        beta2 = self.fc_beta2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma2 = self.fc_gamma2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1

        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        y = gamma1 * y
        y += beta1
        y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        y = gamma2 * y
        y += beta2
        return x + y


class Ghiasi(nn.Module):
    def __init__(self):
        super(Ghiasi,self).__init__()
        self.layers = nn.ModuleList([
            ConvInRelu(3,32,9,stride=1),
            ConvInRelu(32,64,3,stride=2),
            ConvInRelu(64,128,3,stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvInRelu(128,64,3,upsample=2),
            UpsampleConvInRelu(64,32,3,upsample=2),
            UpsampleConvInRelu(32,3,9,upsample=None,activation=None)
        ])

        self.n_params = sum([layer.n_params for layer in self.layers])
    
    def forward(self,x,styles):
        # x: B x 3 x H x W
        # styles: B x 100 batch of style embeddings
        
        for i, layer in enumerate(self.layers):
            if i < 3:
                # first three layers do not perform renormalization (see style_normalization_activations in the original source: https://github.com/tensorflow/magenta/blob/master/magenta/models/arbitrary_image_stylization/nza_model.py)
                x = layer(x)
            else:
                x = layer(x, styles)
        
        return torch.sigmoid(x)
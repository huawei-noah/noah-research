from __future__ import print_function
import abc
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from .nearest_embed_rec import NearestEmbed,SimVQ1D 
from timm.models.layers import to_2tuple

class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return
    
class RecoverLN(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, normalized_shape,expender=1024):
        super().__init__()
        self.expender=expender
        self.ln = nn.LayerNorm(normalized_shape*self.expender)
        # self.ln = nn.LayerNorm(normalized_shape)
        

    def forward(self, x):
        B,L=x.shape
        x = x.reshape(B//self.expender,self.expender*L)
        x=self.ln(x)
        x = x.reshape(B,L)

        return x
    
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, expender=1024,dropout=0.0, **kwargs):#可尝试0.1
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # self.ln = nn.LayerNorm(normalized_shape)
        self.ln = RecoverLN(normalized_shape,expender)
        

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.drop2(x)
        return x
       
class Mlp_VAE(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, \
                 bias=True,norm_layer=nn.LayerNorm, k=10,vq_coef=0.5, commit_coef=0.1,expender=1024,drop=0.): #0.2,0.4 commit_coef=0.01
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features #or in_features
        
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0
        
        self.emb = NearestEmbed(k, out_features)#hidden_features

        self.mlp_encoder0 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        self.mlp_encoder1 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        self.mlp_encoder2 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        
        self.mlp_decoder0 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        self.mlp_decoder1 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        self.mlp_decoder2 = Mlp(in_features=in_features, hidden_features=hidden_features,\
            out_features=out_features,bias=bias,act_layer=act_layer, drop=drop)
        
        self.ennorm1=RecoverLN(out_features,expender)#norm_layer
        self.ennorm2=RecoverLN(out_features,expender)#norm_layer
        self.en_addnorm2 = AddNorm(out_features,expender)#norm_layer
        
        self.denorm1=RecoverLN(out_features,expender)#norm_layer
        self.denorm2=RecoverLN(out_features,expender)#norm_layer
        self.de_addnorm2 = AddNorm(out_features,expender)#norm_layer
        
                                
    def forward(self, x):
        x = self.mlp_encoder0(x)#

        x = x+self.mlp_encoder1(self.ennorm1(x))#
        
        x = self.en_addnorm2(x,self.mlp_encoder2(self.ennorm2(x)))#
        
        y, argmin = self.emb(x, weight_sg=True)#y=[B,emb_size,hidden] index=[B,hidden]
        emb, _ = self.emb(x.detach())
        recon=y
        
        y = y+self.mlp_decoder1(self.denorm1(y))#
        
        y = self.de_addnorm2(y,self.mlp_decoder2(self.denorm2(y)))#
       
        y = self.mlp_decoder0(y)#
        
        return y, x, emb, argmin,recon
    
    
    def loss_function(self, x, recon_x, z_e, emb, argmin,recon):
        self.mse = F.mse_loss(recon_x, x)
        
        self.vq_loss = torch.mean((emb - z_e.detach())**2)
        
        self.commit_loss=torch.norm((recon_x - x)**2, 2, 1).topk(100).values.sum()

        return self.mse+ self.vq_coef*self.vq_loss  #+self.commit_coef*self.commit_loss #
    
    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)
# 2022.06.08-Changed for implementation of TokenFusion
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import torch
import torch.nn as nn
import torch.nn.functional as F 
from . import mix_transformer
from mmcv.cnn import ConvModule
from .modules import num_parallel


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides=None, in_channels=128, embedding_dim=256, num_classes=20, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #decoder_params = kwargs['decoder_params']
        #embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.dropout = nn.Dropout2d(0.1)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = num_parallel
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict = expand_state_dict(self.encoder.state_dict(), state_dict, self.num_parallel)
            self.encoder.load_state_dict(state_dict, strict=True)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, 
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def get_param_groups(self):
        param_groups = [[], [], []]
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.alpha)
        return param_groups

    def forward(self, x):
        x, masks = self.encoder(x)
        x = [self.decoder(x[0]), self.decoder(x[1])]
        ens = 0
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * x[l].detach()
        x.append(ens)
        return x, masks


def expand_state_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            ln = '.ln_%d' % i
            replace = True if ln in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(ln, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict
    

if __name__=="__main__":
    # import torch.distributed as dist
    # dist.init_process_group('gloo', init_method='file:///temp/somefile', rank=0, world_size=1)
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True).cuda()
    wetr.get_param_groupsv()
    dummy_input = torch.rand(2,3,512,512).cuda()
    wetr(dummy_input)

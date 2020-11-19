import pdb
import math
import torch
import numpy as np
from . import builder
import torch.nn as nn
from mmcv import Config
import torch.nn.functional as F

from .kp_utils import make_cnv_layer
import torch.utils.model_zoo as model_zoo
from .bbox import build_assigner, build_sampler, bbox2roi
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss, bbox_overlaps
from .kp_utils import make_tl_layer, make_br_layer, make_region_layer, make_kp_layer, _regr_l1_loss
from .kp_utils import _tranpose_and_gather_feat, _decode, _generate_bboxes, _htbox2roi, _htbox2roi_test, _filter_bboxes

BN_MOMENTUM = 0.1

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='../cache/nnet/DLA34/pretrain/', name='dla34-ba72cf86.pth', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for n in layers:
        for m in n.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class NormConv(nn.Module):
    def __init__(self, chi, cho):
        super(NormConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(chi, cho, kernel_size=(3, 3), stride=1, padding=1, bias=False, dilation=1)
        #DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = NormConv(c, o)
            node = NormConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

class DLASeg(nn.Module):
    def __init__(
        self, db, nstack, base_name,  
        out_dim, head_conv, 
        pretrained, down_ratio, final_kernel, 
        last_level, out_channel=0, 
        make_tl_layer =make_tl_layer, 
        make_br_layer =make_br_layer,
        #make_cnv_layer =make_cnv_layer, 
        make_heat_layer=make_kp_layer,
        make_grouping_layer = make_region_layer, 
        make_regr_layer=make_kp_layer,
        make_region_layer = make_region_layer
    ):
        super(DLASeg, self).__init__()
        
        assert down_ratio in [2, 4, 8, 16]

        self.nstack           = nstack
        self._decode          = _decode
        self._generate_bboxes = _generate_bboxes
        self._db              = db
        self.K                = self._db.configs["top_k"]
        self.input_size       = db.configs["input_size"]
        self.output_size      = db.configs["output_sizes"][0]
        self.kernel           = self._db.configs["nms_kernel"]
        self.gr_threshold     = self._db.configs["gr_threshold"]
        self.categories       = self._db.configs["categories"]
        
        self.grouping_roi_extractor = builder.build_roi_extractor(Config(self._db._model['grouping_roi_extractor']).item)
        self.region_roi_extractor   = builder.build_roi_extractor(Config(self._db._model['region_roi_extractor']).item)
        
        self.roi_out_size   = Config(self._db._model['grouping_roi_extractor']).item.roi_layer.out_size
        self.iou_threshold  = self._db.configs["iou_threshold"]
        self.train_cfg      = Config(self._db._model['train_cfg'])
        self.bbox_head      = builder.build_bbox_head(Config(self._db._model['bbox_head']).item)
        
        self.first_level = int(np.log2(down_ratio))
        self.last_level  = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels  = self.base.channels
        scales    = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])

        #self.cnvs = nn.ModuleList([
        #    make_cnv_layer(64, head_conv) for _ in range(nstack)
        #])
        
        self.tl_cnvs = nn.ModuleList([
           make_tl_layer(64) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
           make_br_layer(64) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(64, head_conv, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(64, head_conv, out_dim) for _ in range(nstack)
        ])

        self.regions = nn.ModuleList([
            make_region_layer(64, 64) for _ in range(nstack)
         ])
        
        self.region_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(64, head_conv, (self.roi_out_size, self.roi_out_size), bias=True),
                              #nn.BatchNorm2d(head_conv),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(head_conv, out_dim, (1, 1))
                          )
                       ])
        
        self.groupings = nn.ModuleList([
            make_grouping_layer(64, 32) for _ in range(nstack)
         ])
        
        self.grouping_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(32, 32, (self.roi_out_size, self.roi_out_size), bias=True),
                              #nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 1, (1, 1))
                          )
                       ])

        for tl_heat, br_heat, region_reduce, grouping_reduce in zip \
           (self.tl_heats, self.br_heats, self.region_reduces, self.grouping_reduces):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            region_reduce[-1].bias.data.fill_(-2.19)
            grouping_reduce[-1].bias.data.fill_(-2.19)


        self.tl_regrs = nn.ModuleList([
            make_regr_layer(64, head_conv, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(64, head_conv, 2) for _ in range(nstack)
        ])
        
        self.init_weights()
        
    def init_weights(self):
        fill_fc_weights(self.tl_regrs)
        fill_fc_weights(self.br_regrs)
        
    def _train(self, *xs):
        image        = xs[0]
        tl_inds      = xs[1]
        br_inds      = xs[2]
        gt_detections= xs[3]
        tag_lens     = xs[4]
        
        num_imgs     = image.size(0)

        outs             = []
        grouping_feats   = []
        region_feats     = []
        decode_inputs    = []
        grouping_list    = []
        gt_list          = []
        gt_labels        = []
        sampling_results = []
        grouping_outs    = []
        region_outs      = []
        
        x = self.base(image)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
            
        self.ida_up(y, 0, len(y))
        
        layers = zip(
            self.tl_cnvs,     self.br_cnvs,
            self.tl_heats,    self.br_heats,    
            self.tl_regrs,    self.br_regrs,    
            self.regions,     self.groupings
        )
        for ind, layer in enumerate(layers):
            tl_cnv_,     br_cnv_     = layer[0:2]
            tl_heat_,    br_heat_    = layer[2:4]
            tl_regr_,    br_regr_    = layer[4:6]
            region_,     grouping_   = layer[6:8]
            
            tl_cnv = tl_cnv_(y[-1])
            br_cnv = br_cnv_(y[-1])

            region_feat    = region_(y[-1])
            grouping_feat  = grouping_(y[-1])
            
            region_feats   += [region_feat]
            grouping_feats += [grouping_feat]

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            
            decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]
            
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            
            outs += [tl_heat, br_heat, tl_regr, br_regr]
            
        ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:])
        for i in range(num_imgs):
            gt_box     = gt_detections[i][:tag_lens[i]][:,:4] 
            ht_box     = ht_boxes[i]
            score_inds = ht_box[:,4] > 0 
            ht_box     = ht_box[score_inds, :4]

            if ht_box.size(0) == 0:
                grouping_list += [gt_box] 
            else:
                grouping_list += [ht_box]

            gt_list  += [gt_box]
            gt_labels += [(gt_detections[i,:tag_lens[i], -1]+ 1).long()]

        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(self.train_cfg.rcnn.sampler, context=self)

        gt_list_ignore = [None for _ in range(num_imgs)]

        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(grouping_list[i], gt_list[i], gt_list_ignore[i], gt_labels[i])
            sampling_result = bbox_sampler.sample(self.categories, assign_result, grouping_list[i], gt_list[i], gt_labels[i])
            sampling_results.append(sampling_result)

        grouping_rois = bbox2roi([res.bboxes for res in sampling_results]) 
        box_targets   = self.bbox_head.get_target(sampling_results, gt_list, gt_labels, self.train_cfg.rcnn)
        roi_labels = box_targets[0]
        gt_label_inds = roi_labels > self.categories
        roi_labels[gt_label_inds] -= self.categories
        grouping_inds = roi_labels > 0
        grouping_labels = grouping_rois.new_full((grouping_rois.size(0), 1, 1, 1), 0, dtype=torch.float).cuda()
        grouping_labels[grouping_inds] = 1
        region_labels = grouping_rois.new_full((grouping_rois.size(0), self.categories+1), 0, dtype=torch.float).cuda()
        region_labels = region_labels.scatter_(1, roi_labels.unsqueeze(-1), 1)
        region_labels = region_labels[:,1:].unsqueeze(-1).unsqueeze(-1)

        grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois)

        for grouping_reduce, grouping_roi_feat in zip(self.grouping_reduces, grouping_roi_feats):
            grouping_outs += [_sigmoid(grouping_reduce(grouping_roi_feat))]

        grouping_scores = grouping_outs[-1][:,0,0,0].clone().detach()
        grouping_scores[gt_label_inds] = 1
        select_inds = grouping_scores >= self.gr_threshold 
        region_rois = grouping_rois[select_inds].contiguous()
        region_labels = region_labels[select_inds]

        region_roi_feats = self.region_roi_extractor(region_feats, region_rois)
        for region_reduce, region_roi_feat in zip(self.region_reduces, region_roi_feats):
             region_outs += [_sigmoid(region_reduce(region_roi_feat))]

        outs += [grouping_outs, grouping_labels, region_outs, region_labels]
                    
        return outs

    def _test(self, *xs, **kwargs):
        image     = xs[0]
        no_flip   = kwargs.pop('no_flip')
        image_idx = kwargs['image_idx'] 
        kwargs.pop('image_idx')
        
        num_imgs = image.size(0)
        
        outs            = []
        region_feats    = []
        grouping_feats  = []
        decode_inputs   = []
        grouping_list   = []
        score_inds_list = []
        
        x = self.base(image)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
            
        self.ida_up(y, 0, len(y))

        layers = zip(
            self.tl_cnvs,     self.br_cnvs,
            self.tl_heats,    self.br_heats,    
            self.tl_regrs,    self.br_regrs,    
            self.regions,     self.groupings
        )
        
        for ind, layer in enumerate(layers):
            tl_cnv_,     br_cnv_     = layer[0:2]
            tl_heat_,    br_heat_    = layer[2:4]
            tl_regr_,    br_regr_    = layer[4:6]
            region_,     grouping_   = layer[6:8]

            tl_cnv = tl_cnv_(y[-1])
            br_cnv = br_cnv_(y[-1])

            region_feat    = region_(y[-1])
            grouping_feat  = grouping_(y[-1])

            region_feats   += [region_feat]
            grouping_feats += [grouping_feat]

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

            decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]

            outs += [tl_regr, br_regr]

            ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:])
            all_groupings = ht_boxes[:,:, -1].new_full(ht_boxes[:,:, -1].size(), 0, dtype=torch.float)

            for i in range(num_imgs):
                ht_box     = ht_boxes[i]
                score_inds = ht_box[:,4] > 0
                ht_box     = ht_box[score_inds, :4]

                grouping_list  += [ht_box]
                score_inds_list+= [score_inds.unsqueeze(0)]

            grouping_rois = _htbox2roi_test(grouping_list)
            grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois.float())

            grouping_scores = self.grouping_reduces[-1](grouping_roi_feats[-1])
            grouping_scores = _sigmoid(grouping_scores)

            grouping_inds = grouping_scores[:,0,0,0] >= self.gr_threshold

            if grouping_inds.float().sum() > 0:
                region_rois = grouping_rois[grouping_inds].contiguous().float()
            else:
                region_rois = grouping_rois

            region_roi_feats = self.region_roi_extractor(region_feats, region_rois)
            region_scores    = self.region_reduces[-1](region_roi_feats[-1])
            region_scores    = _sigmoid(region_scores)

            if grouping_inds.float().sum() > 1:
                 _filter_bboxes(ht_boxes, tl_clses, region_scores, grouping_scores, self.gr_threshold)
            if no_flip:
                all_groupings[score_inds_list[0]] = grouping_scores[:,0,0,0]
            else:
                all_groupings[torch.cat((score_inds_list[0], score_inds_list[1]), 0)] = grouping_scores[:,0,0,0]


            outs += [ht_boxes, all_groupings, tlbr_inds, tlbr_scores, tl_clses, self.gr_threshold]
                
        return self._decode(*outs[-8:], **kwargs)
    
    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight   = pull_weight
        self.push_weight   = push_weight
        self.regr_weight   = regr_weight
        self.focal_loss    = focal_loss
        self.ae_loss       = _ae_loss
        self.regr_loss     = _regr_loss
        self._regr_l1_loss = _regr_l1_loss

    def forward(self, outs, targets):
        region_labels   = outs.pop(-1)
        region_outs     = outs.pop(-1)
        grouping_labels = outs.pop(-1)
        grouping_outs   = outs.pop(-1)
        
        stride = 4

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_regrs = outs[2::stride]
        br_regrs = outs[3::stride]
        
        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        
        # keypoints loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        
        # grouping loss
        grouping_loss = 0
        grouping_loss+= self.focal_loss(grouping_outs, grouping_labels)
        
        # region loss
        region_loss = 0
        region_loss+= self.focal_loss(region_outs, region_labels)

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss
        
        loss = (focal_loss + grouping_loss + region_loss + regr_loss) / len(tl_heats)
        
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (grouping_loss / len(tl_heats)).unsqueeze(0), \
                          (region_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)

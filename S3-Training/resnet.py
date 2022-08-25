import torch.nn as nn
import torchvision.transforms as transforms
import math

from s3conv2dshift import S3Conv2dShift3bit, add_reg_sparse_to_loss

def conv3x3_fp(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)

def conv3x3_s3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return S3Conv2dShift3bit(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3_s3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_s3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        conv2d = S3Conv2dShift3bit

        self.conv1 = conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                S3Conv2dShift3bit(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):
    def __init__(self, num_classes=1000, depth=18):

        super(ResNet_imagenet, self).__init__()

        block = None
        layers = []
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        if depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        if depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        if depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        if depth == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def init_model(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            if isinstance(m, S3Conv2dShift3bit):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight_val.data.normal_(0, math.sqrt(2. / n))
                m.weight_val.data.abs_()
                m.weight_shift.data.normal_(0, math.sqrt(2. / n))
                m.weight_shift.data = m.weight_shift.data.abs() * -1
                m.weight_shift2.data.normal_(0, math.sqrt(2. / n))
                m.weight_shift2.data = m.weight_shift2.data.abs() * -1

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def init_model(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            if isinstance(m, S3Conv2dShift3bit):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight_val.data.normal_(0, math.sqrt(2. / n))
                m.weight_val.data.abs_()
                m.weight_shift.data.normal_(0, math.sqrt(2. / n))
                m.weight_shift.data = m.weight_shift.data.abs() * -1
                m.weight_shift2.data.normal_(0, math.sqrt(2. / n))
                m.weight_shift2.data = m.weight_shift2.data.abs() * -1

            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet(**kwargs):
    num_classes, depth, dataset, pretrained = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'pretrained'])

    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        return ResNet_imagenet(num_classes=num_classes, depth=depth)

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes, block=BasicBlock, depth=depth)

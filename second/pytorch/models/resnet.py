import spconv
from torch import nn 
from torch.nn import functional as F
from torchplus.nn import Empty, GroupNorm, Sequential

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out

class SparseBottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None):
        super(SparseBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, indice_key=indice_key)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out.features = self.relu(out.features)

        return out


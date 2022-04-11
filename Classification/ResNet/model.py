import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    基本的residual模块，用于18，34层的网络
    """
    expansion = 1   # 表示block内的不同conv层之间channel数不变

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=stride,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, stride=1,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = nn.ReLU(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


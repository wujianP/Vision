import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    """
    基本的residual block，用于18，34层的网络
    同时兼容了虚线和实线结构，通过stride和out_channel加以区分：
    1）stride=1且in_channel=out_channel为实线残差结构，
    用于layer2/3/4的除第一层外的其余block，和layer1的所有层残差结构
    2）stride=2且in_channel!=out_channel为虚线残差结构
    用于layer2/3/4的第一层block
    """
    expansion = 1   # 表示block内的不同conv层之间channel数不变

    def __init__(self, in_channel, out_channel,
                 stride=1, downsample=None, **kwargs):

        """
        Args:
            in_channel: 该block输入的channel数
            out_channel: 整个block的输出channel，(conv2_1, conv3_1, conv4_1)中为in_channel的2倍，其余为in_channel
            stride: 表示block中第一个conv层的stride，用于downsample在虚线残差结构，仅(conv2_1, conv3_1, conv4_1)中传入为2，其余为1(默认)
            downsample: 用于下采样残差分支，便于与主分支结果形状适配仅(conv2_1, conv3_1, conv4_1)中传入，其余为None
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               stride=stride,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               stride=1,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """瓶颈residual block, 用于50，101，152层的ResNet搭建,
        同时兼容了虚线和实线结构，通过stride和out_channel加以区分：
        1）stride=1且in_channel=out_channel为实线残差结构，
        用于layer1/2/3/4的非第一层的其余block(res50/101/152中)
        2）stride=1且in_channel=out_channel为第一种虚线残差结构
        用于layer1的第一层block(res50/101/152中)
        3）stride=2且in_channel!=out_channel的为第二种虚线残差结构
        用于layer2/3/4的第一层block(res50/101/152中)
        """
    expansion = 4 # 表示该block中conv3卷积核个数是conv1,2的4倍(减少参数量，先降后升)

    def __init__(self, in_channel, out_channel,
                 stride=1, downsample=None):
        """
        Args:
            in_channel: 该block输入的channel数
            out_channel: 该block中，前两层的输出channel，整个block的输出channel为其4倍(expansion)
            stride: 表示block中第二个conv层的stride，用于downsample在虚线残差结构，仅(conv2_1, conv3_1, conv4_1)中传入为2，其余为1(默认)
            downsample: 用于下采样残差分支，便于与主分支结果形状适配仅(conv2_1, conv3_1, conv4_1)中传入，其余为None
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel*self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample =downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            # 是虚线残差结构
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

import torchvision.models.resnet

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=100,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        """
        Args:
            block: 残差结构的类型，basic or bottleneck
            blocks_num: 是一个四元列表，指定每一个layer包含的block数量
            num_classes: 分类头个数
            include_top: 是否包含avg_pool/FC/softmax这样的分类头
            groups:
            width_per_group:
        """
        super(ResNet, self).__init__()
        self.include_top=include_top
        self.groups = groups
        self.width_per_group = width_per_group
        # 每个layer(1-4)的输入channel数，初始为layer1的输入均为64，随后每个layer之后翻一倍
        self.layer_in_channel=64

        # 第一个阶段的7x7卷积
        self.conv1 = nn.Conv2d(3, self.layer_in_channel, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.layer_in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化层
        self.maxpool= nn.MaxPool2d(3, 2, 1)

        # 四个核心的layer
        self.layer1 = self._make_layers(block=block, channel=64, block_num=blocks_num[0], stride=1)
        self.layer2 = self._make_layers(block=block, channel=128, block_num=blocks_num[1], stride=2)
        self.layer3 = self._make_layers(block=block, channel=256, block_num=blocks_num[2], stride=2)
        self.layer4 = self._make_layers(block=block, channel=512, block_num=blocks_num[3], stride=2)

        # 添加分类头
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 每个channel的输出大小(1,1)
            self.fc = nn.Linear(512*block.expansion, num_classes)

        # 初始化卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layers(self,
                     block,
                     block_num,
                     channel,
                     stride=1):
        """
        Args:
            block: 残差模块的类型
            block_num: 该layer中包含的block数量
            channel: 该layer中block中对应的out_channel字段
                    basic block中表示block的输出channel数
                    bottleneck block中表示该block中前两层的输出channel
                    无论如何，block和layer的输出channel都等于expansion*channel
            stride: 用于区分是否需要减半spatial size
        """
        downsample = None
        # 虚线结构
        if stride!=1 or self.layer_in_channel!=channel*block.expansion:
            # res18/34/50/101/152的layer2/3/4的第一层block的虚线结构满足两个条件
            # res50/101/152的layer1的第一层block的虚线结构只满足layer_in_channel!=channel*expansion
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.layer_in_channel,
                         out_channels=channel * block.expansion,
                         kernel_size=1,
                         stride=stride,
                         bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers = []
        # 插入比较特别，可能存在也可能不存在残差结构的第一层block
        layers.append(block(
            self.layer_in_channel,
            channel,
            downsample=downsample,
            stride=stride
        ))
        # 更新为下一layer的输入channel数(即为本layer的输出channel数)
        self.layer_in_channel = channel*block.expansion
        # 插入其余block，均为实线结构
        for _ in range(1, block_num):
            layers.append(block(
                self.layer_in_channel,
                channel
            ))

        return nn.Sequential(*layers)   # *表示参数解析分发

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.include_top:
            out = self.avgpool(out)
            # 注意要展平，(B, C, 1, 1)->(B, C)
            out = torch.flatten(out, 1)
            out = self.fc(out)

        return out

def resnet18(num_classes=1000, include_top=True):
    return ResNet(block=BasicBlock, blocks_num=[2,2,2,2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    return ResNet(block=BasicBlock, blocks_num=[3,4,6,3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, blocks_num=[3,4,6,3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, blocks_num=[3,4,23,3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, blocks_num=[3,8,36,3], num_classes=num_classes, include_top=include_top)


if __name__ == '__main__':
    net = resnet101()
    input =torch.rand(4, 3, 224, 224)
    print(net(input).shape)
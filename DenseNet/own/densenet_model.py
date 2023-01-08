import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        output = self.bn1(x)
        output = self.relu1(output)
        output = self.conv1(output)

        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv2(output)

        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate)
        return torch.cat([x, output], 1)

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            if i == 0:
                self.layer = nn.Sequential(
                    DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,drop_rate)
                )
            else:
                layer = DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,drop_rate)
                self.layer.add_module("denselayer%d" % (i+1), layer)
    
    def forward(self,input):
        return self.layer(input)

class Transition(nn.Sequential):
    def __init__(self, num_input_feature, num_output_features):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_feature)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self,input):
        output = self.bn(input)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)

        return output

"""
    growth_rate:（int）DenseLayer中使用的增长率和最终输出通道数/过滤器数量(论文中的k)
    block_config：（4个int组成的列表）每个DenseBlock中的层数
    num_init_features：（int）第一个Conv2d中的过滤器数量（输出通道数）
    bn_size：（int）瓶颈层中使用的因子
    compression_rate：（float）过渡层中使用的压缩率
    drop_rate：（float）每个DenseLayer之后的丢弃率
    num_classes：（int）分类的类数
"""
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # 前部
        self.features = nn.Sequential(
            #第一层
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(),
            #第二层
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate,drop_rate)
            if i == 0:
                self.block_tran = nn.Sequential(
                    block
                )
            else:
                self.block_tran.add_module("denseblock%d" % (i + 1), block)#添加一个block
            num_features += num_layers*growth_rate#更新通道数
            if i != len(block_config) - 1:#除去最后一层不需要加Transition来连接两个相邻的DenseBlock
                transition = Transition(num_features, int(num_features*compression_rate))
                self.block_tran.add_module("transition%d" % (i + 1), transition)#添加Transition
                num_features = int(num_features * compression_rate)#更新通道数

        # 后部 bn+ReLU
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):#如果是卷积层，参数kaiming分布处理
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):#如果是批量归一化则伸缩参数为1，偏移为0
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):#如果是线性层偏移为0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        block_output = self.block_tran(features)
        tail_output = self.tail(block_output)
        out = F.avg_pool2d(tail_output, 7, stride=1).view(tail_output.size(0), -1)#平均池化
        out = self.classifier(out)
        return out


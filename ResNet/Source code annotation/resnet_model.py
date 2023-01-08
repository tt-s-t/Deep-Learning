# ResNet源码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#注意这里没有写padding，那就使用默认的padding(default=0).此外，padding还支持字符串选项输入'same' or 'valid'
def conv1x1(inplanes, planes, stride=1): 
    return nn.Conv2d(inplanes, planes, kernel_size=1, stride = stride, bias = False)

#这里bias都设置为0，即不加可训练的偏置 #padding 默认值为1，也就是对应feature map上下左右各加上1个像素点
def conv3x3(inplanes, planes, stride=1, padding=1): 
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

#the bottleneck structure in the right subfigure in Fig.5
#瓶颈结构的瓶颈是指的1x1卷积的通道维度变换
#如Fig5右图所示，输入的256首先经过1x1卷积变为64通道，在64通道的维度下进行3x3卷积，然后使用#1x1卷积变换回256通道，与原始的256通道相加，然后relu激活
#实际的ResNet50以上的网络，输出都是输入的通道的4倍，也即expansion
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride) #注意这里planes代表的是低维通道
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion) #变换为高维通道，并在高维通道相加做残差
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out+identity
        out = self.relu(out)
        return out        

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        #对应第一层conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        #对应第二层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages部分，分别第1，2，3，4个stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #对应最后一层
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):#isinstance()函数来判断一个对象是否是一个已知的类型，类似 type()，其实就是没有正态分布的话咱就kaiming均匀分布。
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#kaiming均匀分布
            elif isinstance(m, nn.BatchNorm2d):#批量处理
                nn.init.constant_(m.weight, 1)#torch.nn.init.constant(tensor, val)，用val的值填充输入的张量或变量，即变成全1分布
                nn.init.constant_(m.bias, 0)#偏置变为0

    #将多个block拼接成一个stage，实现block之间的衔接；block指前面定义的BottleNeck或BasicBlock;planes指进行变换的通道数;blocks指这个stage中block的数目;stride：filter在原图上扫描时，需要跳跃的格数
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:#拓维操作
        #形状不匹配的两种情况:(1)通道数目不匹配，（2）stride导致的feature map的尺寸不匹配
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )# self.inplanes是block的输入通道数，planes是做3x3卷积的空间的通道数，expansion是残差结构中输出维度是输入维度的多少倍，同一个stage内，第一个block，inplanes=planes， 输出为planes*block.expansion
        # 第二个block开始，该block的inplanes等于上一层的输出的通道数planes*block.expansion（类似于卷积后的结果进入下一个卷积时，前一个卷积得到的output的输出为下一个卷积的input）
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        #在一个stage中，只对stage中的第一个block中使用下采样(stride!=1)，其他的block都不影响feature map的维度的
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))#这里就没有downsample了

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
        x = x.view(x.size(0), -1)#推平
        x = self.fc(x)
        return x


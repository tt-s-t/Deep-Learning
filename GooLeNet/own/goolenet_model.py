import torch
import torch.nn as nn
import torch.nn.functional as F

#conv+ReLU
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

#前部
class Front(nn.Module):
    def __init__(self):
        super(Front, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2,ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2,ceil_mode=True)

    def forward(self,input):
        #输入：(N,3,224,224)
        x = self.conv1(input)#(N,64,112,112)
        x = self.maxpool1(x)#(N,64,56,56)
        x = self.conv2(x)#(N,64,56,56)
        x = self.conv3(x)#(N,192,56,56)
        x = self.maxpool2(x)#(N,192,28,28)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_1_1, ch3x3_1, ch3x3_2_1, ch3x3_2, pool_ch):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_1_1, kernel_size=1),
            BasicConv2d(ch3x3_1_1, ch3x3_1, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_2_1, kernel_size=1),
            BasicConv2d(ch3x3_2_1, ch3x3_2, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_ch, kernel_size=1)
        )

    def forward(self, x):
        #输入(N,Cin,Hin,Win)
        branch1 = self.branch1(x)#(N,C1,Hin,Win)
        branch2 = self.branch2(x)#(N,C2,Hin,Win)
        branch3 = self.branch3(x)#(N,C3,Hin,Win)
        branch4 = self.branch4(x)#(N,C4,Hin,Win)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)#(N,C1+C2+C3+C4,Hin,Win)

#辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 输入：aux1:(N,512,14,14), aux2: (N,528,14,14)
        x = self.averagePool(x)# aux1:(N,512,4,4), aux2: (N,528,4,4)
        x = self.conv(x)# (N,128,4,4)
        x = torch.flatten(x, 1)# (N,2048)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x))# (N,1024)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)# (N,num_classes)
        return x

# GooLeNet网络主体
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.front = Front()

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2,ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2,ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        #输入：(N,3,224,224)
        x = self.front(x)#(N,192,28,28)
        x = self.inception3a(x)#(N,256,28,28)
        x = self.inception3b(x)#(N,480,28,28)
        x = self.maxpool3(x)#(N,480,14,14)
        x = self.inception4a(x)#(N,512,14,14)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)#(N,512,14,14)
        x = self.inception4c(x)#(N,512,14,14)
        x = self.inception4d(x)#(N,528,14,14)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)#(N,832,14,14)
        x = self.maxpool4(x)#(N,832,7,7)
        x = self.inception5a(x)#(N,832,7,7)
        x = self.inception5b(x)#(N,1024,7,7)

        x = self.avgpool(x)#(N,1024,1,1)
        x = torch.flatten(x, 1)#(N,1024)
        x = self.dropout(x)
        x = self.fc(x)#(N,num_classes)
        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x
 
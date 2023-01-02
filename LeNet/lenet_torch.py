import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import torch.optim as optim
import torchvision.transforms as transforms
import onnx
import visdom
from collections import OrderedDict

#定义网络各层

class C1(nn.Module):
    def __init__(self): #调用父类，进行网络层的初始化
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ])) #Sequential序贯模型，最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。包括一个输入通道为1，输出通道为6，5*5卷积核的卷积层，一次激励函数，一次最大池化

    def forward(self, img): #将上一层的结果传到这一层进行计算
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ])) #Linear线性变换：将输入特征转为输出特征

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1)) #先得到SoftMax的结果，每个数减他们的最大值，防止下溢和上溢，再LOG
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output

#构建网络

class LeNet5(nn.Module): #导入神经网络基类Module进行搭建
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self): #初始化网络
        super(LeNet5, self).__init__() #调用父类的网络初始化

        #同时添加对应的各个网络层属性,一共6层？7层

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2() 
        self.c3 = C3() 
        self.f4 = F4() 
        self.f5 = F5() 

    def forward(self, img): #执行网络，返回loss
        output = self.c1(img) 

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output

viz = visdom.Visdom()#打开visdom.server
bestaccuracy = 0

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))#下载准备训练数据
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))#下载准备测试数据
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=4)#数据加载器加载训练数据，每支有256个样本
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=4)#数据加载器加载测试数据，每支有1024个样本

#创建对象

net = LeNet5()#创建网络
criterion = nn.CrossEntropyLoss()#计算交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=2e-3)#parameters()获取网络参数；优化网络

#为可视化做准备
cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train() #加与不加都行
    loss_list, batch_list = [], []#分别记录对应次数下的损失
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad() #梯度归零

        output = net(images) #数据进入网络

        loss = criterion(output, labels) #计算损失

        loss_list.append(loss.detach().cpu().item()) #将损失记下，并阻断反向传播
        batch_list.append(i+1) #对应次数加一

        if i % 10 == 0: #10次为一组进行总结
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization更新可视化
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward() #执行反向传播,计算每个参数的梯度
        optimizer.step() #更新所有参数


def test():
    net.eval() #加与不加都行
    total_correct = 0 #记录正确数目
    avg_loss = 0.0 #记录平均错误
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum() #将损失累加起来
        pred = output.detach().max(1)[1] #max(1)得到每行最大值的第一个（得到概率最大的那个）
        total_correct += pred.eq(labels.view_as(pred)).sum() #累加与pred同类型的labels（即为正确）的数值，即记录正确分数

    avg_loss /= len(data_test) #平均误差
    global bestaccuracy
    if(float(total_correct) / len(data_test) > bestaccuracy):
        torch.save(net.cpu().state_dict(), 'model.pth')
    bestaccuracy = max(bestaccuracy,float(total_correct) / len(data_test))
    print('bestaccuracy is %f' % bestaccuracy)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test))) #输出信息
    return bestaccuracy

def train_and_test(epoch): #训练epoch次后测试
    train(epoch) #训练epoch次
    test() #测试
    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True) #产生标准正态分布的随机矩阵，batch_size=1， 1通道（灰度图像），图片尺寸：32x32
    torch.onnx.export(net, dummy_input, "lenet.onnx") #生成onnx，提取预训练模型，便于了我们的算法及模型在不同的框架之间的迁移，比如Caffe2到PyTorch

    onnx_model = onnx.load("lenet.onnx") #载入onnx格式预训练模型
    onnx.checker.check_model(onnx_model) #检查onnx格式模型是否有错误

def main(): #开始训练和测试
    for e in range(1, 16):
        train_and_test(e)

if __name__ == '__main__':
    main()

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as Data
import torch.optim as optim

##########################生成数据部分##########################
num_inputs=2#特征数
num_examples=1000#样本数

#真实模型的w和b
true_w=[2,-3.4]
true_b=4.2

#生成x——正态分布（均值为0，标准差为1）
features=torch.tensor(np.random.normal(scale=1,size=(num_examples,num_inputs))).to(torch.float32)

#生成label(加上均值为0，标准差为0.01的正态分布)
labels=(np.matmul(features,true_w)+true_b).reshape(-1,1)
labels=torch.tensor(labels+np.random.normal(scale=0.01,size=labels.shape)).to(torch.float32)
##########################生成数据部分##########################

##########################数据加载部分##########################
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset=dataset,batch_size=10,shuffle=True,
    num_workers=0
)
##########################数据加载部分##########################

##########################模型定义部分##########################
class LinearNet(nn.Module):
    def __init__(self,num_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_feature,1)
    def forward(self,x):
        return self.linear(x)
##########################模型定义部分##########################

##########################模型训练部分##########################
net = nn.Sequential(
    nn.Linear(num_inputs,1)
)
#初始化网络参数
init.normal_(net[0].weight,mean=0,std=0.01)
init.constant_(net[0].bias,val=0)
#损失函数
loss = nn.MSELoss()
#优化器（小批量随机梯度下降算法）
optimizer = optim.SGD(net.parameters(),lr=0.03)

num_epochs = 10
for epoch in range(1,num_epochs+1):
    for x,y in data_iter:
        output = net(x)
        l = loss(output,y.view(-1,1))
        optimizer.zero_grad()#清空过往梯度
        l.backward()#计算当前梯度，反向传播
        optimizer.step()#模型更新
    print('epoch%d,loss%f'%(epoch,l.item()))#item()返回的是一个浮点型数据
##########################模型训练部分##########################

##########################模型参数获取##########################
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
##########################模型参数获取##########################


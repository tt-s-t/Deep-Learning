import numpy as np
import random
import matplotlib.pyplot as plt

##########################生成数据部分##########################
num_inputs=2#特征数
num_examples=1000#样本数

#真实模型的w和b
true_w=[2,-3.4]
true_b=4.2

#生成x——正态分布（均值为0，标准差为1）
features=np.random.normal(scale=1,size=(num_examples,num_inputs))

#生成label(加上均值为0，标准差为0.01的正态分布)
lables=(np.matmul(features,true_w)+true_b).reshape(-1,1)
lables=lables+np.random.normal(scale=0.01,size=lables.shape)
##########################生成数据部分##########################

#读取数据集
def data_iter(batch_size,features,lables):
    num_examples=len(features)#获取总体样本量
    index=list(range(num_examples))#获取每个样本的索引
    random.shuffle(index)#使索引顺序变成随机的，即样本读取顺序随机
    for i in range(0,num_examples,batch_size):#每次读取batch_size数据量出来
        j=np.array(index[i:min(i+batch_size,num_examples)])#获取这次小批量样本的索引
        yield features[j],lables[j]


class LinearLayer(object):
    def __init__(self,num_examples,num_inputs,batch_size):
        self.num_examples=num_examples
        self.num_inputs=num_inputs
        self.batch_size=batch_size
        #初始化模型参数
        self.w=np.random.normal(scale=0.01,size=(self.num_inputs,1))
        self.b=np.zeros(shape=(1,))
    def forward(self,input):#前向传播
        self.input=input
        self.output=np.matmul(input,self.w)+self.b
        return self.output
    def squard_loss(self,feature,label):#损失函数
        y_hat=np.matmul(feature,self.w)+self.b
        self.loss=np.sum((y_hat-label.reshape(y_hat.shape))**2/self.num_inputs)/self.num_examples
        return self.loss
    def backward(self,label,lr):#定义优化算法(小批量随机梯度下降算法)
        temp=self.output-label.reshape(self.output.shape)
        grad_w=np.matmul(self.input.T,temp)#w的梯度
        grad_b=temp#b的梯度
        self.w=self.w-lr*grad_w/batch_size#w的更新
        self.b=np.sum(self.b-lr*grad_b/batch_size)/batch_size#b的更新


##########################模型训练部分##########################
#超参数设定
lr=0.03
num_epochs=10
batch_size=10

net=LinearLayer(num_examples,num_inputs,batch_size)

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,lables):
        net.forward(x)
        net.backward(y,lr)
    loss=net.squard_loss(features,lables)
    print('epoch'+str(epoch+1)+', loss:'+str(loss))
##########################模型训练部分##########################

##########################结果展示部分##########################
print(net.w,net.b)#输出模型参数
labels_pre=net.forward(features)

plt.figure(1)
#绘制散点图
plt.scatter(range(num_examples), lables, c='b',marker='o')
plt.scatter(range(num_examples), labels_pre, c='y',marker='.')
plt.xlabel('examples',fontsize=20)
plt.ylabel('labels',fontsize=20)
plt.show()

plt.figure(2)
#绘制散点图
plt.scatter(features[:,1], lables, c='b',marker='o')
plt.scatter(features[:,1], labels_pre, c='y',marker='.')
plt.xlabel('examples',fontsize=20)
plt.ylabel('labels',fontsize=20)
plt.show()
##########################结果展示部分##########################
import random
import numpy as np
from layer_softmax_linear import SoftmaxLossLayer,FullyConnectedLayer

#读取数据集
def data_iter(batch_size,num_examples,features,lables):
    index=list(range(num_examples))#获取每个样本的索引
    random.shuffle(index)#使索引顺序变成随机的，即样本读取顺序随机
    for i in range(0,num_examples,batch_size):#每次读取batch_size数据量出来
        j=np.array(index[i:min(i+batch_size,num_examples)])#获取这次小批量样本的索引
        yield features[j],lables[j]

#模型
class Model(object):
    def __init__(self,num_examples,num_input,num_output,batch_size,num_epochs,lr):
        self.num_examples=num_examples
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.lr=lr
        self.linear=FullyConnectedLayer(num_input, num_output)
        self.softmax=SoftmaxLossLayer()
    def forward(self,input):#前向传播
        layer1=self.linear.forward(input)
        self.output=self.softmax.forward(layer1)
        return self.output
    def backward(self):#后向传播
        bottom_diff=self.softmax.backward()
        self.linear.backward(bottom_diff,self.lr)
    def get_loss(self,label):#得到损失loss
        self.loss=self.softmax.get_loss(label)
        return self.loss
    def train(self,features,labels):#训练
        for epoch in range(self.num_epochs):
            loss=0
            for x,y in data_iter(self.batch_size,self.num_examples,features,labels):
                self.forward(x)#前向传播
                loss=loss+self.get_loss(y)#得到一次batchsize的loss
                self.backward()#后向传播
            loss=loss/(self.num_examples//self.batch_size)
            print('epoch'+str(epoch+1)+', loss:'+str(loss))
    def parms(self):#返回模型参数
        return self.linear.save_param()
    def test(self,input):#用训练好的模型测试样本，返回标签和各类别的概率
        layer1=self.linear.forward(input)
        output=self.softmax.forward(layer1)
        label=np.argmax(output,axis=1).reshape(-1,1)#求每行最大值的索引，对应预测的类
        return label,output
    def evaluate(self,features,labels):#验证
        label,_=self.test(features)
        accuracy = np.mean(label==labels.reshape(-1,1))
        print("准确率为：",accuracy)
    



            





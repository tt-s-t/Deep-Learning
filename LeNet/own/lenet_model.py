import numpy as np
from layer_linear_relu_logsoftmax import ReluLayer,FullyConnectedLayer,LogSoftmaxLossLayer
from layer_conv_pool_flatten import ConvolutionalLayer,MaxPoolingLayer,FlattenLayer

#LeNet
class LeNet(object):
    def __init__(self):
        #第一层
        self.c1_conv = ConvolutionalLayer(5,1,6)#(28,28)
        self.c1_relu = ReluLayer()
        self.c1_pool = MaxPoolingLayer(2,2)#(14,14)
        #第二层
        self.c2_1_conv = ConvolutionalLayer(5,6,16)#(10,10)
        self.c2_1_relu = ReluLayer()
        self.c2_1_pool = MaxPoolingLayer(2,2)#(5,5)
        self.c2_2_conv=self.c2_1_conv#(10,10)
        self.c2_2_relu=self.c2_1_relu
        self.c2_2_pool=self.c2_1_pool#(5,5)
        #第三层
        self.c3_conv = ConvolutionalLayer(5,16,120)#(1,1)
        self.c3_relu = ReluLayer()
        #展平
        self.flatten = FlattenLayer([120,1,1],[120*1*1])
        #第四层
        self.f4_linear=FullyConnectedLayer(120,84)
        self.f4_relu=ReluLayer()
        #第五层
        self.f5_linear=FullyConnectedLayer(84,10)
        self.f5_logsoftmax=LogSoftmaxLossLayer()

    def forward(self,input): # 前向传播计算
        output = self.c1_pool.forward(self.c1_relu.forward((self.c1_conv.forward(input))))
        x = self.c2_1_pool.forward(self.c2_1_relu.forward(self.c2_1_conv.forward(output)))
        output = self.c2_2_pool.forward(self.c2_2_relu.forward(self.c2_2_conv.forward(output)))
        output = output+x
        output = self.c3_relu.forward(self.c3_conv.forward(output))
        output = self.flatten.forward(output)
        output = self.f4_relu.forward(self.f4_linear.forward(output))
        output = self.f5_logsoftmax.forward(self.f5_linear.forward(output))
        return output

    def backward(self,lr): # 反向传播计算
        bottom_diff = self.f5_linear.backward(self.f5_logsoftmax.backward(),lr)
        bottom_diff = self.f4_linear.backward(self.f4_relu.backward(bottom_diff),lr)
        bottom_diff = self.flatten.backward(bottom_diff)
        bottom_diff = self.c3_conv.backward(self.c3_relu.backward(bottom_diff),lr)
        bottom_diff_2 = self.c2_2_conv.backward(self.c2_2_relu.backward(self.c2_2_pool.backward(bottom_diff)),lr)
        bottom_diff_1 = self.c2_1_conv.backward(self.c2_1_relu.backward(self.c2_1_pool.backward(bottom_diff)),lr)
        bottom_diff = self.c1_conv.backward(self.c1_relu.backward(self.c1_pool.backward(bottom_diff_1+bottom_diff_2)),lr)
        return bottom_diff
    def get_loss(self,label):#得到损失loss
        self.loss=self.f5_logsoftmax.get_loss(label)
        return self.loss


#模型
class Model(object):
    def __init__(self,num_examples,batch_size,num_epochs,lr):
        self.num_examples=num_examples
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.lr=lr
        self.lenet = LeNet()
        
    def forward(self,input):#前向传播
        self.output=self.lenet.forward(input)
        return self.output

    def backward(self):#后向传播
        self.lenet.backward(self.lr)

    def get_loss(self,label):#得到损失loss
        self.loss=self.lenet.get_loss(label)
        return self.loss

    def train(self,data_train_loader):#训练
        for epoch in range(self.num_epochs):
            loss=0
            for i,(feature,label) in enumerate(data_train_loader):
                self.forward(feature)#前向传播
                loss=loss+self.get_loss(label)#得到一次batchsize的loss
                print(loss/(i+1))
                self.backward()#后向传播
            loss=loss/(self.num_examples//self.batch_size)
            print('epoch'+str(epoch+1)+', loss:'+str(loss))
    def parms(self):#返回模型参数
        return self.linear.save_param()
    def test(self,input):#用训练好的模型测试样本，返回标签和各类别的概率
        output=self.forward(input)
        label=np.argmax(output,axis=1).reshape(-1,1)#求每行最大值的索引，对应预测的类
        return label,output
    def evaluate(self,data_test_loader):#验证
        accuracy_list=[]
        for _,(features,labels) in enumerate(data_test_loader):
            label,_=self.test(features)
            accuracy = np.mean(label==labels.reshape(-1,1))
            accuracy_list.append(accuracy)
        print("准确率为：",np.mean(np.array(accuracy)))
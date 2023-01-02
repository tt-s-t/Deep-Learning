import numpy as np

####################全连接网络定义部分########################
class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        # 参数初始化
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(self.num_input, self.num_output))#W(d,c)
        self.b = np.zeros([1, self.num_output])#B(1,c)
    def forward(self, input):  # 前向传播计算
        self.input = input#X(n,d)
        # 全连接层的前向传播，计算输出结果
        self.output = np.matmul(self.input,self.w)+self.b#利用矩阵乘法(Y=XW+B)——（n,c）
        return self.output
    def backward(self, top_diff,lr):  # 反向传播的计算
        #全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.matmul(self.input.T,top_diff)#Y对W求导为X^T，再乘上一个梯度
        self.d_bias = top_diff#Y对B求导为全1矩阵
        bottom_diff = np.matmul(top_diff,self.w.T)#Y对X(输入,即上一层的输出)求导为W^T
        self.update_param(lr)
        return bottom_diff
    def update_param(self, lr):  # 参数更新
        # 对全连接层参数利用参数进行更新
        self.w = self.w - lr*self.d_weight
        self.b = np.reshape(np.average(self.b - lr*self.d_bias, axis=0),(1, self.num_output))#从(n,c)重新变为(1,c)
    def load_param(self, weight, bias):  # 参数加载
        assert self.w.shape == weight.shape
        assert self.b.shape == bias.shape
        self.w = weight
        self.b = bias
    def save_param(self):  # 参数保存
        return self.w, self.b
####################全连接网络定义部分########################

####################logsoftmax网络定义部分#######################
class LogSoftmaxLossLayer(object):
    def __init__(self):
        pass
    def forward(self, input):  # 前向传播的计算
        input_max = np.max(input, axis=1, keepdims=True)#生成(N,1)大小的每行的最大值
        input_exp = np.exp(input - input_max)#生成(N,C)大小的e^(元素-每行最大值)，input越接近最大的input_max的预测属于该类的概率越大
        sum_denominator = np.sum(input_exp,axis=1,keepdims=True)#取e^(元素-每行最大值)每行的和,变成(N,1)
        self.prob = input_exp / sum_denominator#得到softmax最终公式（即属于每一类的概率）,分母（N，1）拓展成（N，C）
        return self.prob
    def get_loss(self, label):   # 计算损失（这边要的是一个确切的数值结果）
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0#第几个样本最终属于哪一类(概率为1，其他为0)
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # softmax 损失层的反向传播，计算本层损失（这边要的是一个损失矩阵）
        bottom_diff = (self.prob - self.label_onehot)/self.batch_size        
        return bottom_diff
####################logsoftmax网络定义部分#######################

#########################relu部分############################
class ReluLayer(object):
    def __init__(self):
        pass
    def forward(self, input):  # 前向传播的计算
        self.input = input
        # ReLU层的前向传播，计算输出结果
        output = np.maximum(input, 0)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # ReLU层的反向传播，计算本层损失
        temp = self.input
        temp [temp >0] =1
        temp [temp <0] = 0
        bottom_diff = np.multiply(temp ,top_diff)#对应元素位置相乘
        return bottom_diff
#########################relu部分############################


from own.data_pre import load_data
import torch
from torch.utils.data import TensorDataset

##########################数据准备部分##########################
#路径准备
data_path='./mnist_data/'
train_images_path='train-images-idx3-ubyte'
train_labels_path='train-labels-idx1-ubyte'
test_images_path='t10k-images-idx3-ubyte'
test_labels_path='t10k-labels-idx1-ubyte'

train_images,train_labels=load_data(data_path+train_images_path,data_path+train_labels_path)#训练集的图像和labels
test_images,test_labels=load_data(data_path+test_images_path,data_path+test_labels_path)#测试集的图像和labels

#图像和labels装载
data_train = TensorDataset(torch.tensor(train_images,dtype=torch.float32),torch.tensor(train_labels,dtype=float))
data_test = TensorDataset(torch.tensor(test_images,dtype=torch.float32),torch.tensor(test_labels,dtype=float))

#装载数据
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,batch_size = 100,shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,batch_size = 100,shuffle = True)
##########################数据准备部分##########################

##########################参数设定部分##########################
num_examples=train_images.shape[0]#样本数
num_inputs=train_images.shape[1]#特征数目
num_outputs=10#类别数
hidden_size1=128#隐藏层数1
hidden_size2=64#隐藏层数2
batch_size=100
num_epochs=10
lr=0.01
##########################参数设定部分##########################

##########################模型构建部分##########################
class Model(torch.nn.Module):
    def __init__(self,num_inputs,hidden_size1,hidden_size2,num_outputs):
        super(Model,self).__init__()
        
        self.linear1=torch.nn.Linear(num_inputs,hidden_size1)
        self.relu1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(hidden_size1,hidden_size2)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(hidden_size2,num_outputs)
        self.softmax=torch.nn.Softmax()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        output = self.softmax(x)
        return output
##########################模型构建部分##########################

##########################模型训练部分##########################
model=Model(num_inputs,hidden_size1,hidden_size2,num_outputs)#实例化网络
loss_sol=torch.nn.CrossEntropyLoss()#选择交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters())#优化器选择

#开始训练
for epoch in range(num_epochs) :
    sum_loss=0
    train_correct=0
    for data in data_loader_train:
        inputs,labels=data
        outputs=model(inputs)#进行一次前向传播
        optimizer.zero_grad()#梯度清零
        loss=loss_sol(outputs,labels.long())#损失获取
        loss.backward()#反向传播
        optimizer.step()#更新参数
 
        _,id=torch.max(outputs.data,1)#获取训练预测的类别标签
        sum_loss+=loss.data#loss总计
        train_correct+=torch.sum(id==labels.data)#准确个数统计
    print('%d loss:%.03f' % (epoch+1, sum_loss/len(data_loader_train)))
    print('correct:%.03f%%' % (100*train_correct/len(data_train)))
##########################模型训练部分##########################

##########################模型测试部分##########################
model.eval()
test_correct = 0
for data in data_loader_test :
    inputs, lables = data
    outputs = model(inputs)
    _, id = torch.max(outputs.data, 1)
    test_correct += torch.sum(id == lables.data)
print("test correct:%.3f%%" % (100*test_correct/len(data_test)))
##########################模型测试部分##########################


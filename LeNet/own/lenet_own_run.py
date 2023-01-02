from lenet_model import Model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

##########################数据准备部分##########################
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
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)#数据加载器加载训练数据，每支有256个样本
data_test_loader = DataLoader(data_test, batch_size=1024)
##########################数据准备部分##########################

##########################模型构建部分##########################
#参数定义部分
num_examples=data_train.data.shape[0]
batch_size=100
num_epochs=10
lr=0.01
 
net=Model(num_examples,batch_size,num_epochs,lr)
net.train(data_train_loader)
net.evaluate(data_test_loader)
##########################模型构建部分##########################

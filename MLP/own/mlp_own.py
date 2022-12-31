from model import Model
from data_pre import load_data
 
##########################数据准备部分##########################
#路径准备
data_path='./mnist_data/'
train_images_path='train-images-idx3-ubyte'
train_labels_path='train-labels-idx1-ubyte'
test_images_path='t10k-images-idx3-ubyte'
test_labels_path='t10k-labels-idx1-ubyte'

train_images,train_labels=load_data(data_path+train_images_path,data_path+train_labels_path)#训练集
test_images,test_labels=load_data(data_path+test_images_path,data_path+test_labels_path)#测试集
##########################数据准备部分##########################


##########################模型构建部分##########################
#参数定义部分
num_examples=train_images.shape[0]#样本数
num_inputs=train_images.shape[1]#特征数目
num_outputs=10#类别数
hidden_size1=128#隐藏层数1
hidden_size2=64#隐藏层数2
batch_size=100
num_epochs=10
lr=0.01
 
net=Model(num_examples,num_inputs,num_outputs,hidden_size1,hidden_size2,batch_size,num_epochs,lr)
net.train(train_images,train_labels)
net.evaluate(test_images,test_labels)
##########################模型构建部分##########################
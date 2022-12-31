from sklearn import datasets
from model import Model
 
##########################数据准备部分##########################
iris = datasets.load_iris() # 导入鸢尾花数据集
features = iris.data # 特征集
labels = iris.target # label集
##########################数据准备部分##########################
 
##########################模型构建部分##########################
#参数定义部分
num_examples=features.shape[0]#样本数
num_inputs=features.shape[1]#特征数目
num_outputs=3#类别数
batch_size=15
num_epochs=100
lr=0.01
 
net=Model(num_examples,num_inputs,num_outputs,batch_size,num_epochs,lr)
net.train(features,labels)
net.evaluate(features,labels)
parms=net.parms()
print(parms)



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

##########################模型训练部分##########################
X_train,X_test,Y_train,Y_test=train_test_split(features,lables,test_size=0.3,random_state=1)
model = LinearRegression()
model.fit(X_train, Y_train)
##########################模型训练部分##########################

##########################获取结果部分##########################
w, b = model.coef_, model.intercept_
print('得到的参数w为：',w)
print('得到的参数b为：',b)
score = model.score(X_test, Y_test)
print('模型测试得分：'+str(score))
#Y_pred = model.predict(X_test)
#print(Y_pred)
##########################获取结果部分##########################
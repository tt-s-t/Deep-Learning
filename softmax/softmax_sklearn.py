from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#数据准备
iris = datasets.load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y)

#数据标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

#模型训练
model = LogisticRegression()
model.fit(x_train,y_train)
print("参数W为：",model.coef_)
print("参数b为：",model.intercept_)
y_pred = model.predict(x_test)
print("测试集准确率为：",accuracy_score(y_pred,y_test))

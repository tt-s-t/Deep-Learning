from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

#数据准备
iris = datasets.load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y)

#数据标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

#k折交叉验证准备
kf = KFold(n_splits=2)#k=2

#模型训练
score_max=0
model = LogisticRegression()
for train_index, test_index in kf.split(x_train):
    model.fit(x_train[train_index],y_train[train_index])
    score=model.score(x_train[test_index],y_train[test_index])
    if(score>score_max):
        score_max=score
        w_best=model.coef_
        b_best=model.intercept_
        model_best=model

print("参数W为：",w_best)
print("参数b为：",b_best)
y_pred = model_best.predict(x_test)
print("测试集准确率为：",accuracy_score(y_pred,y_test))

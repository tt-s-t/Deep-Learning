import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]), download=True)
data_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
]), download=True)
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4)#数据加载器加载训练数据
data_test_loader = DataLoader(data_test, batch_size=16, num_workers=4)#数据加载器加载测试数据

model=torchvision.models.AlexNet()
model.add_module('add_relu',nn.ReLU())
model.add_module('add_linear',nn.Linear(1000,10))
#model.features[0]=nn.Conv2d(1,64,(11,11),4,2)
#model.add_module('add_linear1',nn.Linear(1000,512))
#model.add_module('add_linear2',nn.Linear(512,128))
#model.add_module('add_linear3',nn.Linear(128,10))
#print(model)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# config
epochs = 12#迭代次数
lr = 0.1#学习率

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train():
    print('start training')
    # 训练模型
    for epoch in range(epochs):
        model.train()#训练模式
        epoch_loss = 0
        epoch_accuracy = 0
        for i, (data, label) in enumerate(data_train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)#输出
            loss = criterion(output, label)#计算loss

            optimizer.zero_grad()#清空过往梯度（因为每次循环都是一次完整的训练）
            loss.backward()#反向传播
            optimizer.step()#更新参数

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(data_train_loader)#当前训练平均准确率
            epoch_loss += loss / len(data_train_loader)#累计loss

        print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')

def test():
    best_accuracy = 0
    model.eval() #加与不加都行
    total_correct = 0 #记录正确数目
    avg_loss = 0.0 #记录平均错误
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        avg_loss += criterion(output, labels).sum() #将损失累加起来
        pred = output.detach().max(1)[1] #max(1)得到每行最大值的第一个（得到概率最大的那个）
        total_correct += pred.eq(labels.view_as(pred)).sum() #累加与pred同类型的labels（即为正确）的数值，即记录正确分数

    avg_loss /= len(data_test) #平均误差
    if(float(total_correct) / len(data_test) > best_accuracy):
        torch.save(model.cpu().state_dict(), 'model.pth')
    best_accuracy = max(best_accuracy,float(total_correct) / len(data_test))
    print('bestaccuracy is %f' % best_accuracy)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test))) #输出信息

def main(): #开始训练和测试
    train()
    test()

if __name__ == '__main__':
    main()
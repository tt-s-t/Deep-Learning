import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

# config
epochs = 5#迭代次数
lr = 0.1#学习率
batch_size = 16

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),  # 随机左右翻转
                                     transforms.RandomVerticalFlip(), # 随机上下翻转
                                     transforms.RandomRotation(degrees=5),#随机旋转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset = datasets.CIFAR10('cifar', True,transform=data_transform["train"], download=True)
validate_dataset = datasets.CIFAR10('cifar', True,transform=data_transform["val"], download=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=2)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = torchvision.models.densenet121()
model.classifier.out_features = 10#修改输出类别数
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print('start training')
# 训练模型
for epoch in range(epochs):
    model.train()#训练模式
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in tqdm(train_loader, leave=False):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()#清空过往梯度（因为每次循环都是一次完整的训练）
        loss.backward()#反向传播
        optimizer.step()#更新参数

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)#当前训练平均准确率
        epoch_loss += loss / len(train_loader)#累计loss

    print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}, train acc:{epoch_accuracy:.4f}')

model.eval()
acc = 0.0  # accumulate accurate number / epoch
with torch.no_grad():
    for data,label in tqdm(validate_loader, leave=False):
        data = data.to(device)
        label = label.to(device)
        outputs = model(data)  # eval model only have last output layer
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, label).sum().item()

val_accurate = acc / len(validate_dataset)
print(val_accurate)
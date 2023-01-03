import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
          super().__init__()
          self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
          self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216,out_features=4096,bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096,out_features=4096,bias=True),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=1000,bias=True),
            #为了适应数据集，因此再添加(更好的做法是在前面调整通道数，但是为了演示保持AlexNet的完整性，因此我只在最后再加了几层)
            nn.ReLU(),
            nn.Linear(in_features=1000,out_features=10,bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

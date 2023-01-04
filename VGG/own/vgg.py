import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self):
          super().__init__()

          self.features = nn.Sequential(
            nn.Conv2d(3,64,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,(3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
          )
          self.classifier = nn.Sequential(
            nn.Linear(25088,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,1000),
            #后面是为了适应数据集而做的补充
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000,10)
          )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x





import torch.nn as nn

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
    return block

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nin_block(1, 96, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Dropout(0.5),
            nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(256, 10, kernel_size=3, stride=1, padding=1),
            #全局平均代替最后的全连接层
            nn.AdaptiveAvgPool2d((1,1))
            )
    
    def forward(self,input):
        x = self.net(input)
        x = x.view(x.size(0), 10)
        #print(x.shape)
        return x

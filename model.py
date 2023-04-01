import torch 
import torch.nn as nn

class convblock(nn.Module):
    def __init__(self, inchannels, outchannels, kernel, stride, padding, bn=True, act=True):
        super().__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel, stride, padding)
        self.batnorm = nn.BatchNorm2d(outchannels) if bn else nn.Identity()
        self.relu = nn.ReLU() if act else nn.Identity()
    def forward(self, x):
        return self.relu(self.batnorm(self.conv(x)))

class Stem(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.seq1 = nn.Sequential(
                convblock(inchannels, 32, 3, 2, 0),
                convblock(32, 32, 3, 1, 0),
                convblock(32, 64, 3, 1, 1)
            )
        self.mp = nn.MaxPool2d(3, 2, 0)
        self.conv1 = convblock(64, 96, 3, 2, 0)
        self.seq2 = nn.Sequential(
                convblock(160, 64, 1, 1, 1),
                convblock(64, 64, (7, 1), 1, 1),
                convblock(64, 64, (1, 7), 1, 1),
                convblock(64, 96, 3, 1, 0)
            )
        self.seq3 = nn.Sequential(
                convblock(160, 64, 1, 1, 0),
                convblock(64, 96, 3, 1, 0),
            )
        self.conv2 = convblock(192, 192, 3, 2, 0)
    def forward(self, x):
        x = self.seq1(x)
        x = torch.cat((self.mp(x), self.conv1(x)), dim=1)
        x = torch.cat((self.seq2(x), self.seq3(x)), dim=1)
        x = torch.cat((self.mp(x), self.conv2(x)), dim=1)
        return x 

class Inception_ResNet_A(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = convblock(inchannels, 32, 1, 1, 0)
        self.seq1 = nn.Sequential(
                convblock(inchannels, 32, 1, 1, 0),
                convblock(32, 32, 3, 1, 1)
            )
        self.seq2 = nn.Sequential(
                convblock(inchannels, 32, 1, 1, 0),
                convblock(32, 48, 3, 1, 1),
                convblock(48, 64, 3, 1, 1)
            )
        self.conv2 = convblock(128, 384, 1, 1, 0, act=False)
    def forward(self, x):
        x = self.relu(x)
        x = x+self.conv2(torch.cat((self.conv1(x), self.seq1(x), self.seq2(x)), dim=1))
        return self.relu(x)

class Inception_ResNet_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = convblock(inchannels, 192, 1, 1, 0)
        self.seq = nn.Sequential(
                convblock(inchannels, 128, 1, 1, 1),
                convblock(128, 160, (1, 7), 1, 1),
                convblock(160, 192, (7, 1), 1, 1),
            )
        self.conv2 = convblock(384, 1152, 1, 1, 0, act=False)
    def forward(self, x):
        x = self.relu(x)
        x = x+self.conv2(torch.cat((self.seq(x), self.conv1(x)), dim=1)) 
        return self.relu(x) 

class Inception_ResNet_C(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = convblock(inchannels, 192, 1, 1, 0)
        self.seq = nn.Sequential(
                convblock(inchannels, 192, 1, 1, 0),
                convblock(192, 224, (1, 3), 1, 0),
                convblock(224, 256, (3, 1), 1, 1)
            )
        self.conv2 = convblock(448, 2144, 1, 1, 0, act=False)
    def forward(self, x):
        x = self.relu(x)
        x = x+self.conv2(torch.cat((self.conv1(x), self.seq(x)), dim=1))
        return self.relu(x)

class Reduction_A(nn.Module):
    def __init__(self, inchannels, n=384, k=256, l=256, m=384):
        super().__init__()
        self.mp = nn.MaxPool2d(3, 2, 0)
        self.conv1 = convblock(inchannels, n, 3, 2, 0)
        self.seq = nn.Sequential(
                convblock(inchannels, k, 1, 1, 1),
                convblock(k, l, 3, 1, 0),
                convblock(l, m, 3, 2, 0),
            )
    def forward(self, x):
        x = torch.cat((self.mp(x), self.conv1(x), self.seq(x)), dim=1)
        return x

class Reduction_B(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.mp = nn.MaxPool2d(3, 2, 0)
        self.seq1 = nn.Sequential(
                convblock(inchannels, 256, 1, 1, 0),
                convblock(256, 384, 3, 2, 0)
            )
        self.seq2 = nn.Sequential(
                convblock(inchannels, 256, 1, 1, 0),
                convblock(256, 288, 3, 2, 0)
            )
        self.seq3 = nn.Sequential(
                convblock(inchannels, 256, 1, 1, 1),
                convblock(256, 288, 3, 1, 0),
                convblock(288, 320, 3, 2, 0)
            )
    def forward(self, x):
        x = torch.cat((self.mp(x), self.seq1(x), self.seq2(x), self.seq3(x)), dim=1)
        return x

class Inception_ResNet_v2(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.inception_resnet_v2 = nn.Sequential(
                Stem(inchannels),
                *[Inception_ResNet_A(384) for i in range(5)],
                Reduction_A(384),
                *[Inception_ResNet_B(1152) for i in range(10)],
                Reduction_B(1152),
                *[Inception_ResNet_C(2144) for i in range(5)],
            )
        self.classifier = nn.Sequential(
                nn.AvgPool2d(8),
                nn.Dropout2d(0.8),
                nn.Flatten(),
                nn.Linear(2144, outchannels),
                nn.Softmax(dim=1)
            )
    def forward(self, x):
        x = self.classifier(self.inception_resnet_v2(x))
        return x

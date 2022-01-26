import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
torch.manual_seed(123)

class USCNN(nn.Module):
    # k:カーネルサイズ,s:ストライド,p:パディング
    def __init__(self, h, nx, ny, nVarIn=1, nVarOut=1, initWay=None, k=5, s=1, p=2):
        super(USCNN, self).__init__()
        self.initWay = initWay
        self.nVarIn = nVarIn
        self.nVarOut = nVarOut
        self.k = k
        self.s = s
        self.p = p
        self.deltaX = h
        self.nx = nx
        self.ny = ny

        # ネットワーク層や活性化関数を定義
        self.relu = nn.ReLU()
        self.US = nn.Upsample(size=[self.ny - 2, self.nx - 2], mode='bicubic')
        self.conv1 = nn.Conv2d(self.nVarIn, 16, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=k, stride=s, padding=p)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=k, stride=s, padding=p)
        self.conv4 = nn.Conv2d(16, self.nVarOut, kernel_size=k, stride=s, padding=p)
        self.pixel_shuffle = nn.PixelShuffle(1)

        # initWayの指定する方法で重みを初期化(l44)
        if self.initWay is not None:
            self._initialize_weights()

    # forwardを定義
    def forward(self, x):
        x = self.US(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        if self.initWay == 'kaiming':
            init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.conv4.weight)
        elif self.initWay == 'ortho':
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)
        else:
            print('Only Kaiming or Orthogonal initializer can be used!')
            exit()
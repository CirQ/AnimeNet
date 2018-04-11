import torch
import torch.nn as nn
from torch.autograd import Variable

from util import initialize_weights


class _ResidualBlockG(nn.Module):
    def __init__(self):
        super(_ResidualBlockG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # initialize_weights(self)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        output = torch.add(out, x)
        return output


class _SubPixelCNN(nn.Module):
    def __init__(self):
        super(_SubPixelCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.shuf = nn.PixelShuffle(upscale_factor=2)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        # initialize_weights(self)

    def forward(self, x):
        out = self.shuf(self.conv(x))
        output = self.relu(self.bn(out))
        return output


class _ResidualBlockD(nn.Module):
    def __init__(self, ch_in, ch, k, s):
        super(_ResidualBlockD, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch, kernel_size=k, stride=s, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=k, stride=s, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        # initialize_weights(self)

    def forward(self, x):
        out = self.conv2(self.relu1(self.conv1(x)))
        output = self.relu2(torch.add(out, x))
        return output


class _BlockD(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, s=1):
        super(_BlockD, self).__init__()
        self.res = nn.Sequential(_ResidualBlockD(ch_in=ch_in, ch=ch_in, k=k, s=s),
                                 _ResidualBlockD(ch_in=ch_in, ch=ch_in, k=k, s=s))
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        # initialize_weights(self)

    def forward(self, x):
        output = self.relu(self.conv(self.res(x)))
        return output


class Generator(nn.Module):
    def __init__(self, feature_num):
        super(Generator, self).__init__()

        self.dense_in = nn.Linear(in_features=128+feature_num, out_features=64*16*16)
        self.bn_in = nn.BatchNorm2d(num_features=64)
        self.relu_in = nn.ReLU()

        self.residual = self.make_layer(_ResidualBlockG, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features=64)
        self.relu_mid = nn.ReLU()

        self.subpixel = self.make_layer(_SubPixelCNN, 3)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.tanh_out = nn.Tanh()

        initialize_weights(self)

    @staticmethod
    def make_layer(block, num_of_layer):
        layers = [block() for _ in range(num_of_layer)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense_in(x).view(-1, 64, 16, 16)
        residual = self.relu_in(self.bn_in(out))
        out = self.residual(residual)
        out = self.relu_mid(self.bn_mid(self.conv_mid(out)))
        out = self.subpixel(torch.add(out, residual))
        output = self.tanh_out(self.conv_out(out))
        return output


class Discriminator(nn.Module):
    def __init__(self, feature_num):
        super(Discriminator, self).__init__()

        self.conv_in = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.relu_in = nn.LeakyReLU(negative_slope=0.2)

        self.features = nn.Sequential(
            _BlockD(ch_in=32, ch_out=64),
            _BlockD(ch_in=64, ch_out=128),
            _BlockD(ch_in=128, ch_out=256),
            _BlockD(ch_in=256, ch_out=512),
            _BlockD(ch_in=512, ch_out=1024))

        self.dense_p = nn.Linear(1024*2*2, 1)
        self.sigmoid_p = nn.Sigmoid()

        self.dense_t = nn.Linear(1024*2*2, feature_num)
        self.sigmoid_t = nn.Sigmoid()

        initialize_weights(self)

    def forward(self, x):
        out = self.relu_in(self.conv_in(x))
        out = self.features(out).view(out.size()[0], -1)
        out_p = self.sigmoid_p(self.dense_p(out)).squeeze(1)
        out_t = self.sigmoid_t(self.dense_t(out))
        return out_p, out_t


def unit_test(tn=10):
    g, d = Generator(tn), Discriminator(tn)
    x = Variable(torch.randn(2, 128+tn))

    o = g(x)
    print(o.size())

    p, t = d(o)
    print(p.size(), t.size())


if __name__ == '__main__':
    unit_test()

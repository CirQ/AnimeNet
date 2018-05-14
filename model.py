import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils

from util import *



class Generator(nn.Module):
    def __init__(self, feature_num, d=128):
        super(Generator, self).__init__()

        self.zlen = 128 + feature_num
        self.deconv1 = nn.ConvTranspose2d(self.zlen, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        self.deconv6 = nn.ConvTranspose2d(d//2, 3, 4, 2, 1)

        initialize_weights(self)

    def forward(self, x):
        x = x.view(-1, self.zlen, 1, 1)
        out = F.relu(self.deconv1_bn(self.deconv1(x)))
        out = F.relu(self.deconv2_bn(self.deconv2(out)))
        out = F.relu(self.deconv3_bn(self.deconv3(out)))
        out = F.relu(self.deconv4_bn(self.deconv4(out)))
        out = F.relu(self.deconv5_bn(self.deconv5(out)))
        output = F.tanh(self.deconv6(out))
        return output


class Discriminator(nn.Module):
    def __init__(self, feature_num, d=128):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, d*4, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d*4)
        self.dense_p = nn.Linear(d*64, 1)
        self.dense_t = nn.Linear(d*64, feature_num)

        initialize_weights(self)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2_bn(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.conv3_bn(self.conv3(out)), 0.2)
        out = F.leaky_relu(self.conv4_bn(self.conv4(out)), 0.2)
        out = F.leaky_relu(self.conv5_bn(self.conv5(out)), 0.2)
        out = out.view(out.size()[0], -1)
        out_p = self.dense_p(out).squeeze(1)
        out_t = self.dense_t(out)
        return out_p, out_t


def unit_test(tn=10):
    batch_size = 1
    img_size = 128
    g, d = Generator(tn), Discriminator(tn)
    x = Variable(torch.ones(batch_size, img_size+tn))

    o = g(x)
    print(o.size())

    vutils.save_image(o.data.view(batch_size, 3, img_size, img_size), 'samples/fake_samples.png')
    p, t = d(o)
    print(p.size(), t.size())


def visialize(tn=30):
    batch_size = 1
    img_size = 128
    g, d = Generator(tn), Discriminator(tn)
    x = Variable(torch.ones(batch_size, img_size+tn))
    gout = g(x)
    dout = d(gout)
    make_dot(gout).render('generator', cleanup=True)
    make_dot(dout[0]).render('discriminator', cleanup=True)


if __name__ == '__main__':
    unit_test()

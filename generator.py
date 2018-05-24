#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-05-23 17:09:50

import os
import time

import torch
from torch import FloatTensor
from torch.autograd import Variable
import torchvision.utils as vutil

from model import Generator

class G(object):
    def __init__(self):
        self.gen = Generator(30)
        if torch.cuda.is_available():
            self.gen = self.gen.cuda()
        self.switch(os.path.join('checkpoint', 'checkpoint_epoch_014.pth'))

    def switch(self, gen_path):
        if os.path.isfile(gen_path):
            checkpoint = torch.load(gen_path)
            self.gen.load_state_dict(checkpoint['g'])

    def __call__(self, tags):
        samples_out_path = os.path.join('samples', 'samples_%d.png' % (int(time.time()) % 100) )
        z = Variable(FloatTensor(1, 128))
        z.data.normal_(0, 1)
        tags = Variable(FloatTensor(tags)).view([1, -1])
        z = torch.cat((z, tags), 1)
        if torch.cuda.is_available():
            z = z.cuda()
        sample = self.gen(z)
        vutil.save_image(sample.data.view(1, 3, 128, 128), samples_out_path)
        return samples_out_path

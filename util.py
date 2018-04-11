#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-04-11 15:39:27

import torch.nn as nn

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import UNETsource


class MUN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3,32, kernel_size=1)
        self.model = UNETsource.UNetModern(hidden_features=32, norm=True,mid_attn=True, ch_mults=(1, 1, 2, 2))
        self.output = nn.Conv2d(32,34, kernel_size=1)

        
    
    def forward(self, x):
        x = self.input(x)
        x = self.model(x)
        return self.output(x)
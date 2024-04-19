#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import UNETsource
import random
import numpy as np


class MUN(nn.Module):
    def __init__(self, ODD = True, threshold=7.5e-6):
        super().__init__()
        self.input = nn.Conv2d(3,32, kernel_size=1)
        self.model = UNETsource.UNetModern(hidden_features=32, norm=True,mid_attn=True, ch_mults=(1, 1, 2, 2))
        self.output = nn.Conv2d(32,34, kernel_size=1)
        self.ODD = ODD
        self.threshold = threshold

        
    
    def forward(self, x):
        x = self.input(x)
        x = self.model(x)
        outputs = self.output(x)

        if self.ODD:
            cls_outputs = ODD(outputs, self.threshold)
            return outputs, cls_outputs
        else:
            return outputs

def confidence_score(output):
    sm = torch.nn.Softmax(dim=1)
    output = sm(output)
    
    median_per_pixel, _ = torch.median(output, dim=1)
    median_per_pixel = median_per_pixel.squeeze().detach().cpu().numpy()
    confidence_score = np.mean(median_per_pixel)
    return confidence_score


def ODD(output, threshold):
    confidence = confidence_score(output)
    odd = (confidence < threshold)
    return odd

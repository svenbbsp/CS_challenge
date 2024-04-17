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
            #cls_outputs = confidence_score_3(outputs)
            return outputs, cls_outputs
        else:
            return outputs
        


def confidence_score(output):
    

    confidence_per_pixel = torch.amax(output, dim=1).squeeze().detach().cpu().numpy().flatten()
    confidence_score = np.median(confidence_per_pixel)
    return confidence_score

def confidence_score_2(output):
    sm = torch.nn.Softmax(dim=1)
    output = sm(output)
    max_per_pixel = torch.amax(output, dim=1).squeeze().detach().cpu().numpy()
    mean_per_pixel = torch.mean(output, dim=1).squeeze().detach().cpu().numpy()
    max_minus_mean = max_per_pixel - mean_per_pixel
    confidence_score = np.mean(max_minus_mean)
    return confidence_score

def confidence_score_3(output):
    sm = torch.nn.Softmax(dim=1)
    output = sm(output)
    
    median_per_pixel, _ = torch.median(output, dim=1)
    median_per_pixel = median_per_pixel.squeeze().detach().cpu().numpy()
    confidence_score = np.mean(median_per_pixel)
    return confidence_score


def ODD(output, threshold):
    confidence = confidence_score_3(output)
    odd = (confidence < threshold)
    return odd

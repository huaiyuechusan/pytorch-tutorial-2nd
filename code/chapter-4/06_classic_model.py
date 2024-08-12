# -*- coding:utf-8 -*-
"""
@file name  : 06_classic_model.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-13
@brief      : torchvision 经典模型学习

"""
import torch
import torch.nn as nn
from torchvision import models

model_alexnet = models.alexnet()

model_vgg16 = models.vgg16()

# googlenet的Inception模块里的ch3x3red参数含义
# ch3x3red 通常代表在3x3卷积之前用于降维的通道数。
# 这里的“red”是“reduce”的缩写，意味着这个通道数是通过降维操作得到的。
model_googlenet = models.googlenet()

model_resnet50 = models.resnet50()


for m in model_alexnet.modules():
    if isinstance(m, torch.nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# -*- coding:utf-8 -*-
"""
@file name  : 02_containers.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-30
@brief      : 熟悉常用容器：sequential, modulelist
"""
import torch
import torch.nn as nn
from torchvision.models import alexnet


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # ========================== Sequential ==========================
    # 点击进入查看alexnet的网络结构
    model = alexnet(pretrained=False)
    fake_input = torch.randn((1, 3, 224, 224))
    output = model(fake_input)

    # ========================== ModuleList ==========================

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
            # self.linears = [nn.Linear(10, 10) for i in range(10)]    # 观察model._modules，将会是空的

        def forward(self, x):
            for sub_layer in self.linears:
                x = sub_layer(x)
            return x

    model = MyModule()
    fake_input = torch.randn((32, 10))
    output = model(fake_input)
    print(output.shape)

    # ========================== ModuleDict ==========================
    class MyModule2(nn.Module):
        def __init__(self):
            super(MyModule2, self).__init__()
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(3, 16, 5),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict({
                    'lrelu': nn.LeakyReLU(),
                    'prelu': nn.PReLU()
            })

        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x

    model2 = MyModule2()
    fake_input = torch.randn((1, 3, 7, 7))
    convout = model2(fake_input, "conv", "lrelu")
    poolout = model2(fake_input, "pool", "prelu")
    print(convout.shape, poolout.shape)
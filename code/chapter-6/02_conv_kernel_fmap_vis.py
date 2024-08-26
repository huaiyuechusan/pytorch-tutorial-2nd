# -*- coding:utf-8 -*-
"""
@file name  : 02_conv_kernel_fmap_vis.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-09
@brief      : 卷积核可视化，特征图可视化
"""
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models


if __name__ == "__main__":
    # ----------------------------------- kernel visualization -----------------------------------
    writer = SummaryWriter(comment='kernel', filename_suffix="_test_your_filename_suffix")
    alexnet = models.alexnet(pretrained=True)
    kernel_num = -1
    vis_max = 1
    # 遍历alexnet的module
    for sub_module in alexnet.modules():
        # 非卷积层则跳过
        if isinstance(sub_module, nn.Conv2d):
            # 超过预期可视化的层数，则停止
            kernel_num += 1
            if kernel_num > vis_max:
                break

            # 获取conv2d层的权重，即卷积核权重
            kernels = sub_module.weight
            c_out, c_int, k_w, k_h = tuple(kernels.shape)

            # 根据卷积核个数进行遍历
            for o_idx in range(c_out):
                # 一个卷积核是4D的，包括两个通道 c_int, c_out,这里将每一个二维矩阵看作一个最细粒度的卷积核，进行绘制。
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # make_grid需要 BCHW，这里拓展C维度
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(kernel_num), kernel_grid, global_step=o_idx)

            # 对总的卷积核进行可视化
            kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
            writer.add_image('{}_all'.format(kernel_num), kernel_grid, global_step=42)

            print("{}_convlayer shape:{}".format(kernel_num, tuple(kernels.shape)))

    writer.close()
    # ----------------------------------- feature map visualization -----------------------------------

    writer = SummaryWriter(comment='fmap_vis', filename_suffix="_test_your_filename_suffix")

    # 数据
    # you can download lena from anywhere. tip: lena(Lena Soderberg, 莱娜·瑟德贝里)
    path_img = r"E:\PyTorch-Tutorial-2nd\data\imgs\lena.png"  # your path to image
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 模型
    alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    """
    为什么要将批次数和通道数调换位置？
    使用 make_grid 函数将特征图可视化时，通常您想要展示的是每个通道的内容，而不是每个批次的内容。
    1. 单个批次：原始形状 (1, 64, 55, 55) 表示只有单个批次，
    这意味着 make_grid 会尝试将64个通道作为同一个图像的通道来处理，而不是将它们视为独立的图像。这不符合我们的可视化需求。
    2. 通道作为独立的图像：通过转置 fmap_1，将其形状变为 (64, 1, 55, 55)，
    我们实际上是在告诉 make_grid 函数，现在有64个独立的“图像”，每个“图像”有一个通道。
    这样，make_grid 就可以将每个通道的内容视为一个独立的图像，并在网格中分别显示它们。
    """
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()















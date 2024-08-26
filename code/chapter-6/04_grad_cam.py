# -*- coding:utf-8 -*-
"""
@file name  : 04_grad_cam_pp.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-15
@brief      : Grad-CAM++ 演示
"""
import cv2
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models


def load_class_names(p_clsnames):
    """
    加载标签名
    :param p_clsnames:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    return class_names


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (224, 224))
    # [:, :, ::-1]: 这是一个numpy切片操作，用于选择和重新排列数组中的元素。
    # :: 第一个和第二个冒号分别表示选择所有行（height）和所有列（width）。
    # ::-1: 第三个切片表示从最后一个元素到第一个元素，步长为-1。这实际上会反转数组在该维度上的顺序。
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    # .detach(): 这个方法返回一个新的张量，从当前的计算图中分离出来。
    # 这意味着新张量不会计算梯度，也就是说，之后对这个新张量的操作不会影响原始张量的梯度。
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir):
    # cv2.applyColorMap(...): 这是OpenCV的函数，用于将灰度图像转换为彩色图像。
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    # 获取类别loss，类别loss为分类类别最大的那个神经元的值
    if not index:
        # argmax：获得最大值的索
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    # np.newaxis: 在数组的维度中增加一个维度，这个维度的大小为1。
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 1000).scatter_(1, index, 1)
    one_hot.requires_grad = True
    # 计算one_hot和ouput_vec的点积
    class_vec = torch.sum(one_hot * ouput_vec)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    # 初始化一个名为cam的数组，其形状与特征图（feature_map）的空间维度相同（高度H和宽度W），类型为浮点数。
    # 初始值全部设为0，这个数组将用于存储类激活图。
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    # 计算梯度（grads）在空间维度（高度和宽度）上的平均值，得到一个权重数组weights。
    # 这里的axis=(1, 2)表示沿着高度和宽度这两个维度进行平均。
    weights = np.mean(grads, axis=(1, 2))
    # 遍历权重数组weights中的每个权重w及其索引i。
    # 将每个权重与对应的特征图通道相乘，并将结果累加到cam数组中。
    # 这样，cam数组将包含加权特征图的总和。
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)   # 确保所有值非负，将小于0的值替换为0。
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':

    path_img = "both.png"
    path_cls_names = "imagenet1000.json"
    output_dir = "./Result"
    input_size = 224

    classes = load_class_names(path_cls_names)
    resnet_50 = models.resnet50(pretrained=True)

    fmap_block = []
    grad_block = []

    # 图片读取
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)

    # 注册hook
    # Module前向传播中的hook,module在前向传播后，自动调用hook函数。
    resnet_50.layer4[-1].register_forward_hook(farward_hook)
    # Module反向传播中的hook,每次计算module的梯度后，自动调用hook函数。
    resnet_50.layer4[-1].register_full_backward_hook(backward_hook)

    # forward
    output = resnet_50(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))
    # backward
    resnet_50.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (input_size, input_size))) / 255
    show_cam_on_image(img_show, cam, output_dir)

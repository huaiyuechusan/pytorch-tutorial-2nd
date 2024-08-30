# -*- coding: utf-8 -*-
"""
# @file name  : 03_torch_device.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : torch.device 使用
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # 查看可用GPU数量
    print("device_count: {}".format(torch.cuda.device_count()))
    # 查看当前使用的设备的序号
    print("current_device: ", torch.cuda.current_device())
    # 查看设备的计算力
    print(torch.cuda.get_device_capability(device=None))
    # 获取设备的名称
    print(torch.cuda.get_device_name())
    # 查看cuda是否可用
    print(torch.cuda.is_available())
    # 获取当前CUDA设备支持的所有计算能力架构列表
    print(torch.cuda.get_arch_list())
    # 查看GPU属性
    print(torch.cuda.get_device_properties(0))
    # 查询gpu空余显存以及总显存。
    print(torch.cuda.mem_get_info(device=None))
    # 类似模型的summary，它将GPU的详细信息进行输出。
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # 清空缓存，释放显存碎片
    print(torch.cuda.empty_cache())


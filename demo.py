import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':

    # 检查是否有 GPU 可用
    print(torch.cuda.is_available())

    # 检查可用的 GPU 数量
    print(torch.cuda.device_count())

    # 获取当前 GPU 设备
    print(torch.cuda.current_device())

    # 获取 GPU 设备名称
    print(torch.cuda.get_device_name(0))


    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    # parser.add_argument('--scale')
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4
    
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    # 使用 transforms.ToTensor() 将图片转换为 PyTorch 张量，同时将像素值标准化到 [0, 1] 范围内。

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    h = int(img.shape[-2] * (args.scale))
    w = int(img.shape[-1] * (args.scale))

    '''
    获取图像的高度 img.shape[-2] 和宽度 img.shape[-1]。
将高度和宽度分别乘以缩放因子 args.scale（这里假设 args.scale 是一个可以转换为整数的字符串，例如 ‘2’）。
将结果转换为整数，得到放大后的高度 h 和宽度 w。
    '''

    scale = h / img.shape[-2]
    # 计算高度缩放的比例，这是放大后高度 h 与原始高度 img.shape[-2] 的比值。

    coord = make_coord((h, w), flatten=False).cuda()
    # 使用 make_coord 函数创建一个坐标网格，其大小为放大后的尺寸 (h, w)。
    # flatten=False 表示返回的坐标网格是二维的。

    cell = torch.ones(1,2).cuda()
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    '''
创建一个形状为 (1, 2) 的张量 cell，并初始化为 1，然后移动到 GPU 上。
更新 cell 张量的第一个元素，将其乘以 2 / h，这是水平方向的单位长度。
更新 cell 张量的第二个元素，将其乘以 2 / w，这是垂直方向的单位长度。
    '''
    
    cell_factor = max(scale/scale_max, 1)
    #pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
    #    coord.unsqueeze(0), cell_factor*cell, bsize=300).squeeze(0)
    pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0),coord.unsqueeze(0), cell_factor*cell).squeeze(0)
    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(3, h, w).cpu()
    transforms.ToPILImage()(pred).save(args.output)

'''
模型加载：
你使用了torch.load(args.model)['model']来加载模型，这假设了保存的模型是一个字典，其中包含一个键为'model'的项。
这通常是可以的，但请确保你的模型保存脚本确实保存了这样的字典。
'''

'''
make_coord函数：
你调用了make_coord函数来生成坐标网格，但没有显示这个函数的定义。
你需要确保这个函数在你的代码库中是可用的，并且它接受正确的参数并返回适合你的模型的坐标张量。
'''

'''
图像预处理：
你正确地将图像转换为张量，并进行了归一化（从[0, 255]缩放到[-0.5, 0.5]）。这是常见的图像预处理步骤，特别是对于神经网络。
'''

'''
cell张量的作用：
你创建了一个cell张量，并对其进行了一些操作，但它的具体作用在你的代码片段中并不明显。
这取决于你的模型如何使用这个cell张量。确保你的模型期望这样的输入。
'''

'''
模型预测：
你的代码中有两种预测调用的注释掉和未注释的版本。请确保你使用的是适合你的模型版本的调用。
如果你的模型是一个特殊的模型，它可能需要特定的输入格式或调用方式。
'''

'''
结果处理：
你将预测结果从模型输出转换回图像格式，并保存为文件。这是正确的，但请确保pred张量的形状和类型适合转换为图像。
'''


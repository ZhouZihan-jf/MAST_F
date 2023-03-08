import torch
import torch.nn as nn

import pdb
from .submodule import ResNet18
from .colorizer import Colorizer

import numpy as np


class MAST(nn.Module):
    def __init__(self, args):
        super(MAST, self).__init__()

        # Model options
        self.p = 0.3
        self.C = 7
        self.args = args

        self.feature_extraction = ResNet18(5)  # ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.D = 4

        # Use smaller R for faster training
        if args.training:
            self.R = 6
            self.feature_extraction = ResNet18(5)
        else:
            self.R = 12
            self.feature_extraction = ResNet18(3)

        self.colorizer = Colorizer(self.D, self.R, self.C)

    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None, current_ind=None):
        # print(f"rgb_r;{rgb_r[0].cpu().size()},quantized_r:{quantized_r[0].cpu().size()},rgb_t:{rgb_t.cpu().size()}")

        # 前一张图片drop后的图片，drop掉的通道，后一张图片drop后的图片
        # rgb_r list 长度为1
        # rgb_r[0] tensor 3,3,256,256  修改后 tensor 12 5 256 256

        # quantized_r list 长度为1
        # quantized_r[0] tensor 3,1,256,256  修改后 tensor 12 1 256 256

        # rgb_t tensor 3,3,256,256  修改后 tensor 12 5 256 256
        # ref_x, ref_y, tar_x,
        feats_r = [self.post_convolution(self.feature_extraction(rgb)) for rgb in rgb_r]
        feats_t = self.post_convolution(self.feature_extraction(rgb_t))
        # 特征图
        # batch_size,64,64,64
        # feats_r是一个list，长度为1，里面的tensor.shape = batch_size,64,64,64

        # 64,64,64的特征图，前一张图片drop掉的通道
        quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t

    def dropout2d_lab(self, arr):  # drop same layers for all images 为所有图像放置相同的图层
        if not self.training:
            return arr

        # 恒等于1
        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        # 选择1，2中的1个(array([1])或者array([2]))
        drop_ch_ind = np.random.choice(np.arange(1, 3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind  # return channels not masked

    # 将lab中dropout的通道改换成flow里的通道
    def dropout2d_lab_fill_flow(self, lab, flow):  # drop same layers for all images 为所有图像放置相同的图层
        if not self.training:
            return lab

        # 恒等于1
        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        # 选择1，2中的1个(array([1])或者array([2]))
        drop_ch_ind = np.random.choice(np.arange(1, 3), drop_ch_num, replace=False)

        for l in lab:
            for f in flow:
                for dropout_ch in drop_ch_ind:
                    l[:, dropout_ch] = f[:, dropout_ch]
            l *= (3 / (3 - drop_ch_num))

        return lab, drop_ch_ind  # return channels not masked

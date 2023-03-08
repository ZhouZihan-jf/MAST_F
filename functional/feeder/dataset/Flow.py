import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
from torch.utils.data import Dataset
from cvbase.optflow.visualize import flow2rgb

TAG_FLOAT = 202021.25


def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def readFlow(sample_dir, resolution, to_rgb):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
    flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb: flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')


def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = ((rgb / 255.0) - 0.5) * 2.0
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    rgb = np.clip(rgb, -1., 1.)
    return einops.rearrange(rgb, 'h w c -> c h w')


def readSeg(sample_dir):
    gt = cv2.imread(sample_dir) / 255
    return einops.rearrange(gt, 'h w c -> c h w')


class FlowPair(Dataset):
    def __init__(self, data_dir, resolution, to_rgb=False, with_rgb=False, with_gt=True):
        self.eval = eval
        self.to_rgb = to_rgb
        self.with_rgb = with_rgb
        self.data_dir = data_dir
        self.flow_dir = data_dir[0]
        self.resolution = resolution
        self.with_gt = with_gt

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        flowgaps = random.choice(list(self.flow_dir.values()))
        vid = random.choice(flowgaps)
        flos = random.choice(vid)
        rgbs, flows, gts, imgdirs = [], [], [], []

        for flo in flos:
            flosplit = flo.split(os.sep)
            rgb_dir = os.path.join(self.data_dir[1], flosplit[-2], flosplit[-1]).replace('.flo', '.jpg')
            gt_dir = os.path.join(self.data_dir[2], flosplit[-2], flosplit[-1]).replace('.flo', '.png')
            img_dir = gt_dir.split('/')[-2:]

            flows.append(readFlow(str(flo), self.resolution, self.to_rgb))
            if self.with_rgb: rgbs.append(readRGB(rgb_dir, self.resolution))
            if self.with_gt: gts.append(readSeg(gt_dir))
            imgdirs.append(img_dir)

        out = np.stack(flows, 0) if not self.with_rgb else np.stack([np.stack(flows, 0), np.stack(rgbs, 0)], -1)
        gt_out = np.stack(gts, 0) if self.with_gt else 0
        return out, gt_out


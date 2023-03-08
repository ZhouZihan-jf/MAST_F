import os, sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch.nn.functional as F
import pdb
import functional.feeder.dataset.Flow as Flow

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0  # 归一化
    image = cv2.resize(image, (256, 256))
    return image


def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)


def lab_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image = transforms.ToTensor()(image)
    # Normalize to range [-1, 1]
    image = transforms.Normalize([50, 0, 0], [50, 127, 127])(image)
    return image


class myImageFloder(data.Dataset):
    def __init__(self, filepath, filenames, training):
        self.refs = filenames
        self.filepath = filepath

    def __getitem__(self, index):
        refs = self.refs[index]

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_lab = [lab_preprocess(ref) for ref in images]
        images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_lab, images_rgb, 1

    def __len__(self):
        return len(self.refs)


class myImageAndFlowFloder(data.Dataset):
    def __init__(self, filepath, imgfilenames, flowfilenames, training, resolution=(256, 256), to_rgb=False):
        self.refs = imgfilenames
        self.data_dir = flowfilenames
        self.flow_dir = flowfilenames[0]
        self.filepath = filepath
        self.resolution = resolution
        self.to_rgb = to_rgb

    def __getitem__(self, index):
        refs = self.refs[index]
        flowgaps = random.choice(list(self.flow_dir.values()))
        vid = random.choice(flowgaps)
        flos = random.choice(vid)
        flows = []

        for flo in flos:
            flosplit = flo.split(os.sep)
            flows.append(Flow.readFlow(str(flo), self.resolution, self.to_rgb))

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_lab = [lab_preprocess(ref) for ref in images]
        images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_lab, images_rgb, flows, 1

    def __len__(self):
        return len(self.refs)

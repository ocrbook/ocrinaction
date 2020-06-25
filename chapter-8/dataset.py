#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

import six
import sys
from PIL import Image
import numpy as np
import cv2


import os

class RawDataset(Dataset):

    def __init__(self, gt_file, gt_dir, transform=None, target_transform=None):

        self.image_list = []
        self.label_list = []

        self.transform = transform
        self.target_transform = target_transform

        gt_lines = open(gt_file, 'r').readlines()

        image_path_list = []
        label_list = []

        lx = {}

        for line in gt_lines:
            if len(line) > 2:
                image_path, label_str = line.split(', ')[:2]

                # 长度大于32的字符串过滤掉，保证CTC正常工作
                if len(label_str) < 32:
                    image_path = os.path.join(gt_dir, image_path)
                    image_path_list.append(image_path)
                    label_str = label_str.replace('\"', '').replace(' ', '').replace('\n', '')
                    label_list.append(label_str)

                for l in label_str:
                    lx[l] = 1

        lx_list = list(lx.keys())
        lx_list.sort()
        lx_str = ''
        for l in lx_list:
            lx_str += l

        # 类别数与字符的对应字典
        print('lexicon:', lx_str)

        self.lexicon = lx_str
        self.nSamples = len(image_path_list)
        self.image_list = image_path_list
        self.label_list = label_list

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        image_path = self.image_list[index % self.nSamples]
        label = self.label_list[index % self.nSamples]

        try:
            #img = cv2.imread(image_path, 0)
            img = Image.open(image_path).convert("L")
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)

# 对图片像素进行正则化处理
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

# 随机顺序采样
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

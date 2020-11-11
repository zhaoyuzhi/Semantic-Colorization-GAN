import os
import cv2
from PIL import Image
import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset

import utils

class ImageNetTrainSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.stringlist = utils.text_readlines(opt.stringlist)
        self.scalarlist = utils.text_readlines(opt.scalarlist)
    
    def __getitem__(self, index):
        ### image part
        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        if self.opt.task == 'jpeg_artifact':
            img.save('temp.jpg', quality = self.opt.jpeg_quality)
            img = Image.open('temp.jpg').convert('RGB')
        img = np.array(img)                                                 # read one image
        if self.opt.task == 'noise':
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_var, size = img.shape)
            img = img + noise
            img = np.clip(img, 0, 255)

        # scaled size should be greater than opts.crop_size
        H = img.shape[0]
        W = img.shape[1]
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
        # crop
        if self.opt.crop_size > 0:
            rand_h = random.randint(0, H - self.opt.crop_size)
            rand_w = random.randint(0, W - self.opt.crop_size)
            img = img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        img = cv2.resize(img, (self.opt.train_size, self.opt.train_size), interpolation = cv2.INTER_CUBIC)
        # normalization
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # additional image processing
        if self.opt.task == 'gray':
            img = img[[0], :, :] * 0.299 + img[[1], :, :] * 0.587 + img[[2], :, :] * 0.114
        if self.opt.task == 'inpainting':
            # mask
            if self.opt.mask_type == 'single_bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
            if self.opt.mask_type == 'bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
            if self.opt.mask_type == 'free_form':
                mask = self.random_ff_mask(shape = self.opt.train_size, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num) 
            mask = torch.from_numpy(mask).contiguous()
            img = img * (1 - mask) + mask
            img = torch.cat((img, mask), 0)
        if self.opt.task == 'grayinpainting':
            img = img[[0], :, :] * 0.299 + img[[1], :, :] * 0.587 + img[[2], :, :] * 0.114
            # mask
            if self.opt.mask_type == 'single_bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
            if self.opt.mask_type == 'bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
            if self.opt.mask_type == 'free_form':
                mask = self.random_ff_mask(shape = self.opt.train_size, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num) 
            mask = torch.from_numpy(mask).contiguous()
            img = img * (1 - mask) + mask
            img = torch.cat((img, mask), 0)
            
        ### target part
        stringname = imgname[:9]                                            # category by str: like n01440764
        for index, value in enumerate(self.stringlist):
            if stringname == value:
                target = self.scalarlist[index]                             # target: 1~1000
                target = int(target) - 1                                    # target: 0~999
                target = np.array(target)                                   # target: 0~999
                target = torch.from_numpy(target).long()
                break
        return img, target

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def __len__(self):
        return len(self.imglist)

class ImageNetValSet(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.text_readlines(opt.namelist)
        self.targetlist = utils.text_readlines(opt.targetlist)
    
    def __getitem__(self, index):
        ### image part
        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # scaled size should be greater than opts.crop_size
        H = img.shape[0]
        W = img.shape[1]
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
        # crop
        if self.opt.crop_size > 0:
            rand_h = random.randint(0, H - self.opt.crop_size)
            rand_w = random.randint(0, W - self.opt.crop_size)
            img = img[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size, :]
        img = cv2.resize(img, (self.opt.train_size, self.opt.train_size), interpolation = cv2.INTER_CUBIC)
        # normalization
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # additional image processing
        if self.opt.task == 'gray':
            img = img[[0], :, :] * 0.299 + img[[1], :, :] * 0.587 + img[[2], :, :] * 0.114
        if self.opt.task == 'inpainting':
            # mask
            if self.opt.mask_type == 'single_bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
            if self.opt.mask_type == 'bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
            if self.opt.mask_type == 'free_form':
                mask = self.random_ff_mask(shape = self.opt.train_size, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num) 
            mask = torch.from_numpy(mask).contiguous()
            img = img * (1 - mask) + mask
            img = torch.cat((img, mask), 0)
        if self.opt.task == 'grayinpainting':
            img = img[[0], :, :] * 0.299 + img[[1], :, :] * 0.587 + img[[2], :, :] * 0.114
            # mask
            if self.opt.mask_type == 'single_bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
            if self.opt.mask_type == 'bbox':
                mask = self.bbox2mask(shape = self.opt.train_size, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
            if self.opt.mask_type == 'free_form':
                mask = self.random_ff_mask(shape = self.opt.train_size, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num) 
            mask = torch.from_numpy(mask).contiguous()
            img = img * (1 - mask) + mask
            img = torch.cat((img, mask), 0)
            
        ### target part
        target = self.targetlist[index]                                     # target of one image 1~1000
        target = int(target) - 1                                            # target: 0~999
        target = np.array(target)                                           # target: 0~999
        target = torch.from_numpy(target).long()
        return img, target

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def __len__(self):
        return len(self.imglist)

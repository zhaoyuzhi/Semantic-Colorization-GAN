import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import utils

class ColorizationDataset(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_jpgs(opt.baseroot_sal)
        '''
        if opt.smaller_coeff > 1:
            imglist = self.create_sub_trainset(imglist, opt.smaller_coeff)
        '''
        self.imglist = imglist

    def create_sub_trainset(self, imglist, smaller_coeff):
        # Sample the target images
        namelist = []
        for i in range(len(imglist)):
            if i % smaller_coeff == 0:
                a = random.randint(0, smaller_coeff - 1) + i
                namelist.append(imglist[a])
        return namelist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot_rgb, imgname)
        salpath = os.path.join(self.opt.baseroot_sal, imgname)

        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # Grayscale input image
        sal = cv2.imread(salpath, -1)                               # Saliency map output image

        # Random cropping
        if self.opt.crop_size > 0:
            h, w = img.shape[:2]
            rand_h = random.randint(0, h - self.opt.crop_size)
            rand_w = random.randint(0, w - self.opt.crop_size)
            img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size, :]
            grayimg = grayimg[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
            sal = sal[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]

        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype = np.float32)
        grayimg = grayimg / 255.0
        img = np.ascontiguousarray(img, dtype = np.float32)
        img = img / 255.0
        sal = np.ascontiguousarray(sal, dtype = np.float32)
        sal = sal / 255.0

        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).unsqueeze(0).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        sal = torch.from_numpy(sal).unsqueeze(0).contiguous()

        return grayimg, img, sal
    
    def __len__(self):
        return len(self.imglist)

class ColorizationDataset_Val(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_files(opt.baseroot_rgb)
        self.imglist = imglist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot_rgb, imgname)

        # Read the images
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # Grayscale input image

        # Normalized to [-1, 1]
        grayimg = np.ascontiguousarray(grayimg, dtype = np.float32)
        grayimg = grayimg / 255.0
        img = np.ascontiguousarray(img, dtype = np.float32)
        img = img / 255.0

        # To PyTorch Tensor
        grayimg = torch.from_numpy(grayimg).unsqueeze(0).contiguous()
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return grayimg, img, imgpath
    
    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":
    
    a = torch.randn(1, 3, 256, 256)
    b = a[:, [0], :, :] * 0.299 + a[:, [1], :, :] * 0.587 + a[:, [2], :, :] * 0.114
    b = torch.cat((b, b, b), 1)
    print(b.shape)

    c = torch.randn(1, 1, 256, 256)
    d = a * c
    print(d.shape)

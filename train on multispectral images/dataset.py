import os
import numpy as np
import cv2
import glob
import random
import math
import torch
import torch.utils.data as data

import utils

def bulid_data_list(filelist, opt, tag):
    lwir_list = []
    visible_list = []
    saliency_list = []
    for i, item in enumerate(filelist):
        # path processing
        filename = item.split('\\')[-1]
        subfolder_name = item.split('\\')[-3]
        set_name = item.split('\\')[-4]
        # append
        if tag == 'train':
            lwir_list.append(os.path.join(opt.train_path, set_name, subfolder_name, 'lwir', filename))
            visible_list.append(os.path.join(opt.train_path, set_name, subfolder_name, 'visible', filename))
            saliency_list.append(os.path.join(opt.train_path, set_name, subfolder_name, 'saliency map', filename))
        if tag == 'val':
            lwir_list.append(os.path.join(opt.val_path, set_name, subfolder_name, 'lwir', filename))
            visible_list.append(os.path.join(opt.val_path, set_name, subfolder_name, 'visible', filename))
            saliency_list.append(os.path.join(opt.val_path, set_name, subfolder_name, 'saliency map', filename))
    return lwir_list, visible_list, saliency_list

class NIR_Colorization_dataset(data.Dataset):
    def __init__(self, opt, tag):
        self.opt = opt
        if tag == 'train':
            filelist = utils.get_half_files(opt.train_path)
            lwir_list, visible_list, saliency_list = bulid_data_list(filelist, opt, tag)
            self.lwir_list = lwir_list
            self.visible_list = visible_list
            self.saliency_list = saliency_list
        if tag == 'val':
            filelist = utils.get_half_files(opt.val_path)
            lwir_list, visible_list, saliency_list = bulid_data_list(filelist, opt, tag)
            self.lwir_list = lwir_list
            self.visible_list = visible_list
            self.saliency_list = saliency_list

    def __getitem__(self, index):

        # Read images
        lwir_path = self.lwir_list[index]
        visible_path = self.visible_list[index]
        
        lwir_img = cv2.imread(lwir_path)
        visible_img = cv2.imread(visible_path)
        
        # Processing
        lwir_img = cv2.resize(lwir_img, (self.opt.crop_size, self.opt.crop_size))
        visible_img = cv2.resize(visible_img, (self.opt.crop_size, self.opt.crop_size))

        # Normalization and add noise
        lwir_img = lwir_img.astype(np.float) / 255.0
        visible_img = visible_img.astype(np.float) / 255.0

        # to tensor
        lwir_img = torch.from_numpy(lwir_img).float().permute(2, 0, 1).contiguous()
        visible_img = torch.from_numpy(visible_img).float().permute(2, 0, 1).contiguous()

        return lwir_img, visible_img

    def __len__(self):
        return len(self.lwir_list)

class NIR_Colorization_with_Sal_dataset(data.Dataset):
    def __init__(self, opt, tag):
        self.opt = opt
        if tag == 'train':
            filelist = utils.get_half_files(opt.train_path)
            lwir_list, visible_list, saliency_list = bulid_data_list(filelist, opt, tag)
            self.lwir_list = lwir_list
            self.visible_list = visible_list
            self.saliency_list = saliency_list
        if tag == 'val':
            filelist = utils.get_half_files(opt.val_path)
            lwir_list, visible_list, saliency_list = bulid_data_list(filelist, opt, tag)
            self.lwir_list = lwir_list
            self.visible_list = visible_list
            self.saliency_list = saliency_list

    def __getitem__(self, index):

        # Read images
        lwir_path = self.lwir_list[index]
        visible_path = self.visible_list[index]
        saliency_path = self.saliency_list[index]
        
        lwir_img = cv2.imread(lwir_path)
        visible_img = cv2.imread(visible_path)
        saliency_img = cv2.imread(saliency_path)
        
        # Processing
        lwir_img = cv2.resize(lwir_img, (self.opt.crop_size, self.opt.crop_size))
        visible_img = cv2.resize(visible_img, (self.opt.crop_size, self.opt.crop_size))
        saliency_img = cv2.resize(saliency_img, (self.opt.crop_size, self.opt.crop_size))

        # Normalization and add noise
        lwir_img = lwir_img.astype(np.float) / 255.0
        visible_img = visible_img.astype(np.float) / 255.0
        saliency_img = saliency_img.astype(np.float) / 255.0

        # to tensor
        lwir_img = torch.from_numpy(lwir_img).float().permute(2, 0, 1).contiguous()
        visible_img = torch.from_numpy(visible_img).float().permute(2, 0, 1).contiguous()
        saliency_img = torch.from_numpy(saliency_img).float().permute(2, 0, 1).contiguous()

        return lwir_img, visible_img, saliency_img

    def __len__(self):
        return len(self.lwir_list)

if __name__ == "__main__":
    
    a = torch.randn(1, 3, 256, 256)
    b = a[:, [0], :, :] * 0.299 + a[:, [1], :, :] * 0.587 + a[:, [2], :, :] * 0.114
    b = torch.cat((b, b, b), 1)
    print(b.shape)

    c = torch.randn(1, 1, 256, 256)
    d = a * c
    print(d.shape)

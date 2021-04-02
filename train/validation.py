import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--load_name', type = str, \
        default = './models/SCGAN_noGAN_epoch40_bs32.pth', \
            help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channel for U-Net decoder')
    parser.add_argument('--latent_channels', type = int, default = 128, help = 'start channel for APN')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type for generator')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type for discriminator')
    # dataset
    parser.add_argument('--baseroot_rgb', type = str, \
        default = 'E:\\submitted papers\\VCGAN\\VCGAN comparison\\SCGAN\\videvo', \
            help = 'color image baseroot')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()
    print(opt)
    
    # Define the network
    generator = utils.create_generator_val(opt).cuda()

    # Define the dataset
    trainset = dataset.ColorizationDataset_Val(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # For loop training
    for i, (true_L, true_RGB, imgpath, h, w) in enumerate(dataloader):

        # print
        imgpath = imgpath[0]
        print(i, imgpath)

        # To device
        true_L = true_L.cuda()
        true_RGB = true_RGB.cuda()

        # Forward
        with torch.no_grad():
            fake_RGB = generator(true_L)

        # Recover normalization: * 255 because last layer is sigmoid activated
        fake_RGB = fake_RGB * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = fake_RGB.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save validation images
        img_copy = cv2.resize(img_copy, (w, h))
        cv2.imwrite(imgpath, img_copy)
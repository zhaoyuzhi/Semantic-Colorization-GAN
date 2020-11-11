import argparse
import os
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
    parser.add_argument('--val_path', type = str, default = './validation_results', help = 'save the validation results to certain path')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
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
    parser.add_argument('--baseroot_rgb', type = str, default = './dataset/ILSVRC2012_val_rgb', help = 'color image baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()
    print(opt)
    utils.check_path(opt.val_path)
    
    # Define the network
    generator = utils.create_generator_val(opt)

    # Define the dataset
    trainset = dataset.ColorizationDataset_Val(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # For loop training
    for i, (true_L, true_RGB) in enumerate(dataloader):

        # To device
        true_L = true_L.cuda()
        true_RGB = true_RGB.cuda()

        # Forward
        fake_RGB = generator(true_L)

        # Save validation images
        img_list = [fake_RGB, true_RGB]
        name_list = ['pred', 'gt']
        utils.save_sample_png(sample_folder = opt.val_path, sample_name = str(i), img_list = img_list, name_list = name_list)


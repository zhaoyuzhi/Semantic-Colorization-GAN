import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

def create_generator(opt):
    # Initialize the networks
    colorizationnet = network.SCGAN(opt)
    if opt.load_name == '':
        print('Generator is created!')
        # Init the networks
        network.weights_init(colorizationnet, init_type = opt.init_type, init_gain = opt.init_gain)
        pretrained_dict = torch.load(opt.global_feature_network_path)
        load_dict(colorizationnet.global_feature_network, pretrained_dict)
        print('Generator is loaded with %s!' % (opt.global_feature_network_path))
    else:
        pretrained_dict = torch.load(opt.load_name)
        load_dict(colorizationnet, pretrained_dict)
        print('Generator is loaded!')
    return colorizationnet
    
def create_generator_val(opt):
    # Initialize the networks
    colorizationnet = network.SCGAN_val(opt)
    pretrained_dict = torch.load(opt.load_name)
    load_dict(colorizationnet, pretrained_dict)
    print('Generator is loaded!')
    # It does not gradient
    for param in colorizationnet.parameters():
        param.requires_grad = False
    return colorizationnet

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator70(opt)
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    return discriminator

def create_perceptualnet(opt):
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Pre-trained VGG-16
    pretrained_dict = torch.load(opt.perceptual_path)
    load_dict(perceptualnet, pretrained_dict)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_dict):
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def get_files(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_dirs(path):
    # Read a folder, return a list of names of child folders
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2]
            ret.append(a)
    return ret

def get_jpgs(path):
    # Read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_relative_dirs(path):
    # Read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            a = os.path.join(root, filespath)
            a = a.split('\\')[-2] + '/' + a.split('\\')[-1]
            ret.append(a)
    return ret

def text_save(content, filename, mode = 'a'):
    # Save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_sample_png(sample_folder, sample_name, img_list, name_list):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

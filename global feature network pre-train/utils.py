import os
import torch
import numpy as np

from network_VGG import *
from network_ResNet import *

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

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

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_network(opt):
    # Initialize the network according to the type, sub-type, normalization, task
    if opt.type == 'vgg':
        if opt.sub_type == 'vgg16':
            if opt.task == 'gray':
                network = VGG16(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16(2, 1000)
        if opt.sub_type == 'vgg16_bn':
            if opt.task == 'gray':
                network = VGG16_BN(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16_BN(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16_BN(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16_BN(2, 1000)
        if opt.sub_type == 'vgg16_in':
            if opt.task == 'gray':
                network = VGG16_IN(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16_IN(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16_IN(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16_IN(2, 1000)
        if opt.sub_type == 'vgg16_fc':
            if opt.task == 'gray':
                network = VGG16_FC(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16_FC(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16_FC(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16_FC(2, 1000)
        if opt.sub_type == 'vgg16_bn_fc':
            if opt.task == 'gray':
                network = VGG16_BN_FC(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16_BN_FC(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16_BN_FC(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16_BN_FC(2, 1000)
        if opt.sub_type == 'vgg16_in_fc':
            if opt.task == 'gray':
                network = VGG16_IN_FC(1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = VGG16_IN_FC(3, 1000)
            if opt.task == 'inpainting':
                network = VGG16_IN_FC(4, 1000)
            if opt.task == 'grayinpainting':
                network = VGG16_IN_FC(2, 1000)
    if opt.type == 'resnet':
        if opt.sub_type == 'resnet18_bn':
            if opt.task == 'gray':
                network = ResNet_BN(BasicBlock_BN, [2, 2, 2, 2], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_BN(BasicBlock_BN, [2, 2, 2, 2], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_BN(BasicBlock_BN, [2, 2, 2, 2], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_BN(BasicBlock_BN, [2, 2, 2, 2], 2, 1000)
        if opt.sub_type == 'resnet18_in':
            if opt.task == 'gray':
                network = ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_IN(BasicBlock_IN, [2, 2, 2, 2], 2, 1000)
        if opt.sub_type == 'resnet34_bn':
            if opt.task == 'gray':
                network = ResNet_BN(BasicBlock_BN, [3, 4, 6, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_BN(BasicBlock_BN, [3, 4, 6, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_BN(BasicBlock_BN, [3, 4, 6, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_BN(BasicBlock_BN, [3, 4, 6, 3], 2, 1000)
        if opt.sub_type == 'resnet34_in':
            if opt.task == 'gray':
                network = ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_IN(BasicBlock_IN, [3, 4, 6, 3], 2, 1000)
        if opt.sub_type == 'resnet50_bn':
            if opt.task == 'gray':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 6, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 6, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 6, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 6, 3], 2, 1000)
        if opt.sub_type == 'resnet50_in':
            if opt.task == 'gray':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 6, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 6, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 6, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 6, 3], 2, 1000)
        if opt.sub_type == 'resnet101_bn':
            if opt.task == 'gray':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 23, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 23, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 23, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 4, 23, 3], 2, 1000)
        if opt.sub_type == 'resnet101_in':
            if opt.task == 'gray':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 23, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 23, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 23, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 4, 23, 3], 2, 1000)
        if opt.sub_type == 'resnet152_bn':
            if opt.task == 'gray':
                network = ResNet_BN(Bottleneck_BN, [3, 8, 36, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_BN(Bottleneck_BN, [3, 8, 36, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 8, 36, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_BN(Bottleneck_BN, [3, 8, 36, 3], 2, 1000)
        if opt.sub_type == 'resnet152_in':
            if opt.task == 'gray':
                network = ResNet_IN(Bottleneck_IN, [3, 8, 36, 3], 1, 1000)
            if opt.task == 'rgb' or opt.task == 'lab':
                network = ResNet_IN(Bottleneck_IN, [3, 8, 36, 3], 3, 1000)
            if opt.task == 'inpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 8, 36, 3], 4, 1000)
            if opt.task == 'grayinpainting':
                network = ResNet_IN(Bottleneck_IN, [3, 8, 36, 3], 2, 1000)
    # Init or Load value for the network
    if opt.finetune_path == "":
        weights_init(network, init_type = opt.init_type, init_gain = opt.init_gain)
        print('network is created!')
    else:
        pretrained_net = torch.load(opt.finetune_path)
        network = load_dict(network, pretrained_net)
        print('network is loaded!')
    return network

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

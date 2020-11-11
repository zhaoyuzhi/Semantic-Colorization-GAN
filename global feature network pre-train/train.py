import argparse
import os

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Basic setting
    parser.add_argument('--checkpoint_interval', type = int, default = 10, help = 'interval between model checkpoints')
    parser.add_argument('--finetune_path', type = str, default = '', help = 'pre-trained model name')
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'whether multiple gpus are needed')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'if input data structure is unchanged, set it as True')
    # Training setting
    parser.add_argument('--epochs', type = int, default = 150, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'size of the batches')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'SGD: initial learning rate')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'SGD: momentum')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'SGD: weight-decay, L2 normalization')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 30, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.1, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'SGD: momentum')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'SGD: weight-decay, L2 normalization')
    # Dataset setting
    parser.add_argument('--baseroot', type = str, \
        default = './dataset/ILSVRC2012_train_256', \
            help = 'the training folder')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'size of image crop')
    parser.add_argument('--train_size', type = int, default = 256, help = 'size of image for training')
    parser.add_argument('--stringlist', type = str, default = './txt/mapping_string.txt', help = 'mapping_string')
    parser.add_argument('--scalarlist', type = str, default = './txt/mapping_scalar.txt', help = 'mapping_scalar')
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    # Network setting
    parser.add_argument('--type', type = str, default = 'vgg', help = 'vgg | resnet')
    parser.add_argument('--sub_type', type = str, default = 'vgg16_bn_fc', help = 'sub network types of vgg | resnet')
    parser.add_argument('--task', type = str, default = 'gray', help = 'gray | rgb | inpainting | grayinpainting')
    opt = parser.parse_args()
    print(opt)

    '''
    # Multi-GPU setting
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
    
    # Enter main function
    import trainer
    trainer.Trainer(opt)

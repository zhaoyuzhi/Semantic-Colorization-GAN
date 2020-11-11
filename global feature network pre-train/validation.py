import argparse
import os

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Basic setting
    parser.add_argument('--finetune_path', type = str, \
        default = './models/vgg/vgg16/rgb/vgg16_bn_fc_gray_epoch150_bs256.pth', \
            help = 'pre-trained model name')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'whether multiple gpus are needed')
    # Validation setting
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches, only 1 is accepted')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Dataset setting
    parser.add_argument('--baseroot', type = str, \
        default = './dataset/ILSVRC2012_val_256', \
            help = 'the validation folder')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'size of image crop')
    parser.add_argument('--train_size', type = int, default = 256, help = 'size of image for training')
    parser.add_argument('--namelist', type = str, default = './txt/ILSVRC2012_val_name.txt', help = 'mapping_string')
    parser.add_argument('--targetlist', type = str, default = './txt/imagenet_2012_validation_scalar.txt', help = 'mapping_scalar')
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

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Enter main function
    import trainer
    trainer.Valer(opt)

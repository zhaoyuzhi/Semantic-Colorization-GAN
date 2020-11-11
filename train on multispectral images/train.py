import argparse
import os

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # pre-train, saving, and loading parameters
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 10, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--save_path', type = str, default = './models', help = 'save the pre-trained model to certain path')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'save the pre-trained model to certain path')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--global_feature_network_path', type = str, \
        default = './trained_models/vgg16_bn_fc_gray_epoch150_bs256.pth', \
            help = 'the path that contains the pre-trained ResNet model')
    parser.add_argument('--perceptual_path', type = str, default = './trained_models/vgg16_pretrained.pth', help = 'the path that contains the pre-trained VGG-16 model')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'True for more than 1 GPU, we recommend to use 4 NVIDIA Tesla v100 GPUs')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # training parameters
    parser.add_argument('--epochs', type = int, default = 40, help = 'number of epochs of training') # change if fine-tune
    parser.add_argument('--train_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 2e-4, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 1e-4, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = int, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 100000, help = 'lr decrease at certain iteration and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--gan_mode', type = str, default = 'noGAN', help = 'type of GAN: [noGAN | LSGAN | WGAN | WGANGP], WGAN is recommended')
    # loss balancing parameters
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'coefficient for L1 Loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.05, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_attn', type = float, default = 0.5, help = 'coefficient for Attention Loss')
    parser.add_argument('--lambda_percep', type = float, default = 5, help = 'coefficient for Perceptual Loss')
    parser.add_argument('--lambda_gp', type = float, default = 10, help = 'coefficient for WGAN-GP coefficient')
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
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'intialization type for generator and discriminator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the standard deviation if Gaussian normalization')
    # dataset
    parser.add_argument('--baseroot_rgb', type = str, default = './dataset/ILSVRC2012_train_rgb', help = 'color image baseroot')
    parser.add_argument('--baseroot_sal', type = str, default = './dataset/ILSVRC2012_train_sal', help = 'saliency map baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--smaller_coeff', type = int, default = 1, help = 'sample images')
    opt = parser.parse_args()
    print(opt)
    
    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    import trainer
    if opt.gan_mode == 'noGAN':
        trainer.trainer_noGAN(opt)
    if opt.gan_mode == 'LSGAN':
        trainer.trainer_LSGAN(opt)
    if opt.gan_mode == 'WGAN':
        trainer.trainer_WGAN(opt)
    if opt.gan_mode == 'WGANGP':
        trainer.trainer_WGANGP(opt)

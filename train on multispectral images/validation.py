import argparse
import os
import torch

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--val_path', type = str, default = './validation_results', help = 'saving path that is a folder')
    parser.add_argument('--load_name', type = str, \
        default = './models/train_pre/G_epoch40_bs1.pth', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'relu', help = 'activation type of networks')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    # Dataset parameters
    '''
    parser.add_argument('--train_path', type = str, \
        default = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train', help = 'train baseroot')
    '''
    parser.add_argument('--val_path', type = str, \
        default = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\val', help = 'val baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator_val(opt).cuda()
    test_dataset = dataset.NIR_Colorization_dataset(opt, 'val')
    print('The total number of images:', len(test_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.savepath)

    # forward
    val_PSNR = 0
    val_SSIM = 0
    for i, (in_img, RGBout_img) in enumerate(test_loader):
        # To device
        # A is for input image, B is for target image
        in_img = in_img.cuda()
        RGBout_img = RGBout_img.cuda()
        
        # Forward propagation
        with torch.no_grad():
            out, sal = generator(in_img)

        # Sample data every iter
        img_list = [out, RGBout_img]
        name_list = ['pred', 'gt']
        utils.save_sample_png(sample_folder = opt.savepath, sample_name = '%d' % (i), img_list = img_list, name_list = name_list)
        
        # PSNR
        val_PSNR_this = utils.psnr(out, RGBout_img, 1) * in_img.shape[0]
        print('The %d-th image PSNR %.4f' % (i, val_PSNR_this))
        val_PSNR = val_PSNR + val_PSNR_this
        # SSIM
        val_SSIM_this = utils.ssim(out, RGBout_img) * in_img.shape[0]
        print('The %d-th image SSIM %.4f' % (i, val_SSIM_this))
        val_SSIM = val_SSIM + val_SSIM_this

    val_PSNR = val_PSNR / len(test_dataset)
    print('The average PSNR equals to', val_PSNR)
    val_SSIM = val_SSIM / len(test_dataset)
    print('The average SSIM equals to', val_SSIM)

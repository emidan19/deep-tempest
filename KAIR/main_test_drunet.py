import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# Testing code for DRUNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# Adapted by Emilio Martínez (emiliomartinez98@gmail.com)
'''


def main(json_path='options/test_drunet.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--add_noise', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    add_noise = parser.parse_args().add_noise

    # ----------------------------------------
    # configure logger
    # ----------------------------------------

    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt = option.dict_to_nonedict(opt)

    model_path = opt['path']['pretrained_netG']
    model_epoch = (model_path.split('/')[-1]).split('_G')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    border = opt['scale']

    opt_netG = opt['netG']

    in_nc = opt_netG['in_nc']
    out_nc = opt_netG['out_nc']
    nc = opt_netG['nc']
    nb = opt_netG['nb']
    act_mode = opt_netG['act_mode']
    bias = opt_netG['bias']

    from models.network_unet import UNetRes as net
    model = net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, bias=bias)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))


    """  
    # ----------------------------------------
    # Step--3 (load paths)
    # ----------------------------------------
    """
    L_paths = util.get_image_paths(opt['datasets']['test']['dataroot_L'])
    H_paths = util.get_image_paths(opt['datasets']['test']['dataroot_H'])
    add_noise = add_noise or (L_paths == H_paths)
    noise_sigma = opt['datasets']['test']['sigma_test']

    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_loss = 0.0
    avg_edgeJaccard = 0.0
    idx = 0

    for L_path, H_path in zip(L_paths,H_paths):
        idx += 1
        image_name_ext = os.path.basename(L_path)
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        logger.info('Creating inference on test image...')

        ###################
        ### Load images ###
        ###################

        # Load ground-truth image and use mean of channels if is RGB
        img_H = util.imread_uint(H_path, n_channels=3)
        img_Htensor = util.uint2tensor4(img_H)

        if img_H.ndim == 3:
            img_H = np.mean(img_H, axis=2)

        if L_path==H_path:
            img_L = img_H.copy()
        else:   
            img_L = util.imread_uint(L_path, n_channels=1)
        
        img_Ltensor = util.uint2tensor4(img_L)        

        # Add noise
        if add_noise:
            noise_level = torch.FloatTensor([int(noise_sigma)])/255.0
            noise = torch.randn(img_Ltensor.size()).mul_(noise_level).float()
            img_Ltensor.add_(noise)
        img_Ltensor = img_Ltensor.to(device)

        # Inference on image
        img_Etensor = model(img_Ltensor)
        img_E = util.tensor2uint(img_Etensor)

        # -----------------------
        # save noisy L
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_{}std.png'.format(img_name, noise_sigma))
        util.imsave(img_L, save_img_path)
        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_model{}_{}std.png'.format(img_name, model_epoch, noise_sigma))
        util.imsave(img_E, save_img_path)

        logger.info(f'Inference of {img_name} completed. Saved at {img_dir}.')

        # ----------------------------------
        # calculate PSNR, SSIM and Jaccard
        # ----------------------------------
        current_psnr = util.calculate_psnr(img_E, img_H, border=border)
        current_ssim = util.calculate_ssim(img_E, img_H, border=border)
        current_edgeJaccard = util.calculate_edge_jaccard(img_E, img_H)
        
        # -----------------------
        # calculate loss
        # -----------------------

        current_loss = model.G_lossfn(img_Etensor,img_Htensor)


        logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB ; SSIM = {:.3f} ; edgeJaccard = {:.3f} ; G_loss = {:.3e}'.format(idx, image_name_ext, current_psnr, current_ssim, current_edgeJaccard, current_loss))

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_edgeJaccard += current_edgeJaccard
        avg_loss += current_loss

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_edgeJaccard = avg_edgeJaccard / idx
    avg_loss = avg_loss / idx

    # testing log
    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:.3f}, Average edgeJaccard : {:.3f}, Average loss : {:.3e}\n'.format(current_epoch, current_step, avg_psnr, avg_ssim, avg_edgeJaccard, avg_loss))

if __name__ == '__main__':
    main()

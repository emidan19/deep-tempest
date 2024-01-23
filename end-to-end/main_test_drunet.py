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

# OCR metrics
# First, must install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html
# Second, install CER/WER and tesseract python wrapper libraries
# pip install fastwer
# pip install pybind11
# pip install pytesseract 
import pytesseract
import fastwer


def calculate_cer_wer(img_E, img_H):
    # Transcribe ground-truth image to text
    text_H = pytesseract.image_to_string(img_H).strip().replace('\n',' ')

    # Transcribe estimated image to text
    text_E = pytesseract.image_to_string(img_E).strip().replace('\n',' ')

    cer = fastwer.score_sent(text_E, text_H, char_level=True)
    wer = fastwer.score_sent(text_E, text_H)

    return cer, wer

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

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

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
    noise_sigma = opt['datasets']['test']['sigma_test']

    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_edgeJaccard = 0.0
    avg_cer = 0.0
    avg_wer = 0.0
    idx = 0

    for L_path, H_path in zip(L_paths,H_paths):
        idx += 1
        image_name_ext = os.path.basename(L_path)
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        logger.info('Creating inference on test image...')

        # Load image
        img_L_original = util.imread_uint(L_path, n_channels=3)[50:-50,100:-100,:]
        img_L = img_L_original[:,:,:2]
        img_L = util.uint2single(img_L)
        img_L = util.single2tensor4(img_L)
        
        # Add noise
        if noise_sigma > 0:
            noise_level = torch.FloatTensor([int(noise_sigma)])/255.0
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)
        img_L = img_L.to(device)

        # Inference on image
        img_E = model(img_L)
        img_L_tmp = util.tensor2uint(img_L)
        img_L = np.zeros_like(img_L_original)
        img_L[:,:,:2] = img_L_tmp
        img_E = util.tensor2uint(img_E)

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

        # Load H image and compute metrics
        img_H = util.imread_uint(H_path, n_channels=3)
        if img_H.ndim == 3:
            img_H = np.mean(img_H, axis=2)
            img_H = img_H.astype('uint8')

        # ----------------------------------------
        # compute PSNR, SSIM, edgeJaccard and CER
        # ----------------------------------------
        current_psnr = util.calculate_psnr(img_E, img_H)
        current_ssim = util.calculate_ssim(img_E, img_H)
        current_edgeJaccard = util.calculate_edge_jaccard(img_E, img_H)
        current_cer, current_wer = calculate_cer_wer(img_E, img_H)

        logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB ; SSIM = {:.3f} ; edgeJaccard = {:.3f} ; CER = {:.3f}% ; WER = {:.3f}%'.format(idx, image_name_ext, current_psnr, current_ssim, current_edgeJaccard, current_cer, current_wer))

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_edgeJaccard += current_edgeJaccard
        avg_cer += current_cer
        avg_wer += current_wer

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_edgeJaccard = avg_edgeJaccard / idx
    avg_cer = avg_cer / idx
    avg_wer = avg_wer / idx

    # Average log
    logger.info('[Average metrics] PSNR : {:<4.2f}dB, SSIM = {:.3f} : edgeJaccard = {:.3f} : CER = {:.3f}% : WER = {:.3f}%'.format(avg_psnr, avg_ssim, avg_edgeJaccard, avg_cer, avg_wer))

if __name__ == '__main__':
    main()

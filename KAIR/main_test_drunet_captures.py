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

import warnings
warnings.filterwarnings('ignore')

# OCR metrics
# First, must install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html
# Second, install CER/WER and tesseract python wrapper libraries
# pip install pybind11
# pip install fastwer
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


def main(json_path='options/train_drunet_finetuning.json'):

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
    print(f"\nDevice: {device}\n")
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
    noise_sigma = opt['datasets']['test']['sigma_test']

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)

    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_loss = 0.0
    avg_edgeJaccard = 0.0
    avg_cer = 0.0
    avg_wer = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        # Inference

        ## With abs value/max-entropy thresholding
        L_visual = test_data['L']
        L_img = util.tensor2uint(L_visual)
        # E_img = np.abs(L_img[:,:,0] + 1j*L_img[:,:,1])
        # E_img = (255 * (E_img/np.max(E_img))).astype("uint8")
        E_img = util.max_entropy_init(L_img) # using global thresholding
        

        H_visual = test_data['H']
        H_img = util.tensor2uint(H_visual)
        ## With drunet
        # Load image

        # E_visual = model(test_data['L'].cuda())
        # E_img = util.tensor2uint(E_visual)

        # H_visual = test_data['H']
        # H_img = util.tensor2uint(H_visual)

        # -----------------------
        # save estimated image E
        # -----------------------
        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        save_img_path = os.path.join(img_dir, '{:s}_E.png'.format(img_name))
        util.imsave(E_img, save_img_path)

        # -----------------------
        # calculate PSNR and SSIM
        # -----------------------
        current_psnr = util.calculate_psnr(E_img, H_img)
        current_ssim = util.calculate_ssim(E_img, H_img)
        current_edgeJaccard = util.calculate_edge_jaccard(E_img, H_img)
        current_cer, current_wer = calculate_cer_wer(E_img, H_img)

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

    # testing log
    logger.info('[Average metrics] PSNR : {:<4.2f}dB, SSIM = {:.3f} : edgeJaccard = {:.3f} : CER = {:.3f}% : WER = {:.3f}%'.format(avg_psnr, avg_ssim, avg_edgeJaccard, avg_cer, avg_wer))

if __name__ == '__main__':
    main()

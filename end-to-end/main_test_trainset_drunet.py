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

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt = option.dict_to_nonedict(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    border = opt['scale']
    opt_netG = opt['netG']
    in_nc = opt_netG['in_nc']
    out_nc = opt_netG['out_nc']
    nc = opt_netG['nc']
    nb = opt_netG['nb']
    act_mode = opt_netG['act_mode']
    bias = opt_netG['bias']

    """  
    # ----------------------------------------
    # Step--2 (load paths)
    # ----------------------------------------
    """
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                        shuffle=False, num_workers=1,
                                        drop_last=False, pin_memory=True)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    """  
    # ----------------------------------------
    # Step--3 (Run models)
    # ----------------------------------------
    """
    model_epochs_str = [str(epoch) for epoch in np.arange(2,19)*10]
    epochs_path = "denoising/drunet/models/"

    for epoch in model_epochs_str:
        current_epoch = int(epoch)
        model_path = os.path.join(epochs_path,f"{epoch}_G.pth")

        logger_name = f'test_model{epoch}_G'
        opt["path"]["pretrained_netG"] = model_path
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

        # from models.network_unet import UNetRes as net
        # model = net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, bias=bias)
        # model.load_state_dict(torch.load(model_path), strict=True)
        # model = model.to(device)
        model = define_Model(opt)
        model.init_train()
        # model.eval()
        # for k, v in model.named_parameters():
        #     v.requires_grad = False
        

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

        for test_data in test_loader:
            idx += 1

            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            model.feed_data(test_data)
            model.test()

            visuals = model.current_visuals()
            E_visual = visuals['E']
            E_img = util.tensor2uint(E_visual)
            H_visual = visuals['H']
            H_img = util.tensor2uint(H_visual)

            # -----------------------
            # calculate PSNR and SSIM
            # -----------------------
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)
            current_ssim = util.calculate_ssim(E_img, H_img, border=border)
            current_edgeJaccard = util.calculate_edge_jaccard(E_img, H_img)

            logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB ; SSIM = {:.3f} ; edgeJaccard = {:.3f}'.format(idx, image_name_ext, current_psnr, current_ssim, current_edgeJaccard))

            avg_psnr += current_psnr
            avg_ssim += current_ssim
            avg_edgeJaccard += current_edgeJaccard

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_edgeJaccard = avg_edgeJaccard / idx

        # testing log
        logger.info('<epoch:{:3d}, Average PSNR : {:<.2f}dB, Average SSIM : {:.3f}, Average edgeJaccard : {:.3f}%\n'.format(current_epoch, avg_psnr, avg_ssim, avg_edgeJaccard))
if __name__ == '__main__':
    main()

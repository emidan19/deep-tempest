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

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-

    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], pretrained_path = opt['path']['pretrained_netG'])
    opt['path']['pretrained_netG'] = init_path_G
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'test'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for test
    # ----------------------------------------

    dataset_opt = opt['datasets']['test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                                shuffle=False, num_workers=1,
                                drop_last=False, pin_memory=True)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''
    # avg_psnr = 0.0
    # avg_ssim = 0.0
    # idx = 0

    for test_data in test_loader:
        # idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        logger.info('Creating inference on test image...')

        model.feed_data(test_data, need_H = False)
        model.test()

        visuals = model.current_visuals(need_H=False)
        E_img = util.tensor2uint(visuals['E'])

        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        util.imsave(E_img, save_img_path)

        logger.info(f'Inference of {img_name} completed. Saved at {img_dir}.')

        # -----------------------
        # calculate PSNR
        # -----------------------
        # current_psnr = util.calculate_psnr(E_img, H_img)

        # -----------------------
        # calculate SSIM
        # -----------------------
        # current_ssim = util.calculate_ssim(E_img, H_img)

        # logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB, SSIM = {:<4.2f}'.format(idx, image_name_ext, current_psnr, current_ssim))

        # avg_psnr += current_psnr
        # avg_ssim += current_ssim

    # avg_psnr = avg_psnr / idx
    # avg_ssim = avg_ssim / idx

    # testing log
    # logger.info('Average PSNR : {:<.2f}dB, Average SSIM : {:<4.2f}\n'.format(avg_psnr, avg_ssim))

if __name__ == '__main__':
    main()

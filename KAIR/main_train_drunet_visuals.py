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


def save_deeptempest_result(imgs_dict, save_path):

    # Name of original image and saving path
    file_name = imgs_dict['image_name']
    save_img_path = os.path.join(save_path, file_name)

    # High resolution image
    H = imgs_dict['H_vis']
    util.imsave(H, save_img_path+'_H.png')

    # Low resolution image (real/imaginary part and module)
    L = util.tensor2uint(imgs_dict['L'])
    L_complex = np.pad(L,((0,0),(0,0),(0,1)))    # Real and imaginary
    util.imsave(L_complex, save_img_path+'_L_complex.png')

    L = L.astype('float')
    L = np.abs(L[:,:,0] + 1j*L[:,:,1])
    L = 255*(L - L.min())/(L.max() - L.min())   # Module
    util.imsave(L, save_img_path+'_L_module.png')

    E = imgs_dict['E_vis']

    util.imsave(E, save_img_path+'_E.png')

'''
# --------------------------------------------
# training code for DRUNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
'''


def drunet_pipeline(json_path='options/train_drunet.json'):

    test_imgs_list = []

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    # parser.add_argument('--launcher', default='pytorch', help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--dist', default=False)

    opt = option.parse(json_path, is_train=True)
    # opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # if opt['rank'] == 0:
    #     util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G', pretrained_path = opt['path']['pretrained_netG'])
    opt['path']['pretrained_netG'] = init_path_G
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    # if opt['rank'] == 0:
    #     option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
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
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
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
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''

    avg_psnr = 0.0
    avg_loss = 0.0
    idx = 0

    for j, test_data in enumerate(test_loader):
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        # img_dir = os.path.join(opt['path']['images'], img_name)
        # util.mkdir(img_dir)

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])
        H_img = util.tensor2uint(visuals['H'])

        test_imgs_list.append({'L':test_data['L'], 
                                'H':test_data['H'],
                                'H_vis': H_img,
                                'E_vis': E_img,
                                'image_name': img_name
                                })

        # -----------------------
        # save estimated image E
        # -----------------------
        # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        # util.imsave(E_img, save_img_path)

        # -----------------------
        # calculate PSNR
        # -----------------------
        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
        # -----------------------
        # calculate loss
        # -----------------------
        current_loss = model.G_lossfn_weight * model.G_lossfn(model.E, model.H)

        logger.info('{:->4d}--> {:>10s} | PSNR = {:<4.2f}dB ; G_loss = {:.3e}'.format(idx, image_name_ext, current_psnr, current_loss))

        avg_psnr += current_psnr
        avg_loss += current_loss

    avg_psnr = avg_psnr / idx
    avg_loss = avg_loss / idx

    # testing log
    logger.info('Average PSNR : {:<.2f}dB, Average loss : {:.3e}\n'.format(avg_psnr, avg_loss))

    return test_imgs_list

if __name__ == '__main__':
    imgs = drunet_pipeline()
    save_path = 'testsets/web_subset/visuals' # COMPLETE SAVING PATH
    for img in imgs:
        save_deeptempest_result(img, save_path)


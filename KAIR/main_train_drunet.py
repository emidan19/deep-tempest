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
# training code for DRUNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
'''


def main(json_path='options/train_drunet.json'):

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

    init_epoch_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G', pretrained_path = opt['path']['pretrained_netG'])
    #init_epoch_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    #opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    #current_epoch = max(init_epoch_G, init_epoch_optimizerG)
    current_epoch = init_epoch_G

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
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.floor(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=True,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

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
    # Step--4 (main training)
    # ----------------------------------------
    '''
    current_step = current_epoch*train_size

    for epoch in range(opt['train']['epochs']):  # keep running
        
        # Update epoch
        current_epoch += 1

        epoch_loss = 0.0

        if opt['dist']:
            train_sampler.set_epoch(current_epoch)

        idx = 0
        for i, train_data in enumerate(train_loader):

            idx += 1
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information (loss)
            # -------------------------------

            logs = model.current_log()
            batch_loss = logs['G_loss'] # get batch loss / iter loss

            epoch_loss += batch_loss

        # -------------------------------
        # Training information
        # -------------------------------      

        # Epoch loss
        epoch_loss = epoch_loss / idx

        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> G_loss: {:.3e} '.format(current_epoch, 
                                                                                current_step, 
                                                                                model.current_learning_rate(),
                                                                                epoch_loss
                                                                                )
        logger.info(message)

        # -------------------------------
        # Save model
        # -------------------------------
        if current_epoch % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
            logger.info('Saving the model.')
            model.save(current_epoch)

        # -------------------------------
        # Testing
        # -------------------------------
        if current_epoch % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

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
                # save estimated image E
                # -----------------------

                if current_epoch % opt['train']['checkpoint_test_save'] == 0:

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_epoch))
                    util.imsave(E_img, save_img_path)

                # -----------------------
                # calculate PSNR and SSIM
                # -----------------------
                current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                current_edgeJaccard = util.calculate_edge_jaccard(E_img, H_img)
                # -----------------------
                # calculate loss
                # -----------------------
                current_loss = model.G_lossfn(E_visual, H_visual)

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

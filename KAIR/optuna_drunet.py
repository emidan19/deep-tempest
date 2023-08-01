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
import torch.nn as nn
import optuna
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

# Define model function with optuna hyperparameters
def define_model(trial, opt):

    # Set learning rate suggestions for trial
    opt['train']['G_optimizaer_lr'] = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    opt['train']["G_tvloss_weight"] = trial.suggest_loguniform("tv_weight", 1e-7, 1e-2)

    # Initialize model
    model = define_Model(opt)
    model.init_train()

    return model

def define_metric(metric_str):

    metric_dict = {}

    if metric_str == 'PSNR':
        metric_dict['func'] = util.calculate_psnr
        metric_dict['direction'] = 'maximize'

    elif metric_str == 'SSIM':
        metric_dict['func'] = util.calculate_ssim
        metric_dict['direction'] = 'maximize'
    
    # TODO implement
    # elif metric_str == 'CER':
    #     metric_dict['func'] = utilOCR.calculate_cer
    #     metric_dict['direction'] = 'minimize'

    else:
        # If none of above, choose MSE
        metric_dict['func'] = nn.MSELoss
        metric_dict['direction'] = 'minimize'
    
    return metric_dict

def train_model(trial, model, metric_dict, criterion, optimizer, num_epochs=25):

    metric = metric_dict['func']
    metric_direction = metric_dict['direction']

    # Time tracker
    since = time.time()

    # Copy model weights to get best weights register
    best_model_wts = copy.deepcopy(model.state_dict())

    best_metric = 0*(metric_direction=='maximize') + 1e6*(metric_direction=='minimize')

    current_step = 0

    # Iter over epoch
    for epoch in range(num_epochs):

        epoch_loss = 0.0

        epoch_metric = 0.0

        # -------------------------------
        # Training phase
        # ------------------------------- 
        for i, train_data in enumerate(train_loader):
            
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

            visuals = model.current_visuals()
            E_visual = visuals['E']
            E_img = util.tensor2uint(E_visual)
            H_visual = visuals['H']
            H_img = util.tensor2uint(H_visual)

            epoch_metric += metric(H_img, E_img)

            epoch_loss += batch_loss     

        # Train loss and metric
        avg_train_loss = epoch_loss/train_size
        avg_train_metric = epoch_metric/train_size

        logs = model.current_log()  # such as loss
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> G_loss: {:.3e} '.format(current_epoch, 
                                                                                current_step, 
                                                                                model.current_learning_rate(),
                                                                                epoch_loss
                                                                                )
        logger.info(message)


        # -------------------------------
        # Validation phase
        # -------------------------------
        val_metric = 0.0
        avg_val_loss = 0.0
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

            current_loss = model.G_lossfn(E_visual, H_visual)

            val_metric += metric(H_img, E_img)

            avg_val_loss += current_loss
        
        # Val loss and metric
        avg_val_loss = avg_val_loss/idx
        val_metric = val_metric/idx

        # Update if validation metric is better
        if val_metric > best_metric:
                        best_metric = val_metric
                        best_model_wts = copy.deepcopy(model.state_dict())
        
        # Report trial epoch and check if should prune
        trial.report(best_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Whole optuna parameters searching time
    time_elapsed = time.time() - since

    return best_metric


# Define optuna objective function
def objective(trial, opt):

    # Generate the model and optimizers
    model = define_model(trial, opt)

    best_metric = train_model(trial, model, metric, num_epochs=25)    
    
    # Save best model for each trial
    # torch.save(best_model.state_dict(), f"model_trial_{trial.number}.pth")

    # Return metric (Objective Value) of the current trial
    return best_metric

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
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)


    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.floor(len(train_set) / dataset_opt['dataloader_batch_size']))
            train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True)
            train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                        shuffle=True,
                                        num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                        drop_last=True,
                                        pin_memory=True,
                                        sampler=train_sampler)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (setup optuna hyperparameters)
    # ----------------------------------------
    '''

    define_model(trial, opt)
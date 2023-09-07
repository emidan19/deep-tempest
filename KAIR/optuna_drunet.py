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
from torch.utils.data import Subset
import optuna
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''

json_path='options/optuna_options.json'

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

# ----------------------------------------
# return None for missing key
# ----------------------------------------
opt = option.dict_to_nonedict(opt)

# Set logger
logger_name = 'optuna_hparams'
utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
logger = logging.getLogger(logger_name)
logger.info(option.dict2str(opt))

'''
# ----------------------------------------
# Step--2 (create dataloader)
# ----------------------------------------
'''

message = 'Loading train and val datasets'
logger.info(message)

seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':

        batch_size = dataset_opt['dataloader_batch_size']
        patch_size = dataset_opt['H_size']
        train_set = define_Dataset(dataset_opt)
        # Keep only one third of the dataset
        indexes = torch.randperm(len(train_set))[:len(train_set)//4]
        train_set = Subset(train_set, indexes)
        train_size = int(math.floor(len(train_set) / batch_size))
        message = f'Training dataset with {train_size} batches (batch size={batch_size}) of {patch_size}x{patch_size} images.'
        logger.info(message)
        train_loader = DataLoader(  train_set,
                                    batch_size=dataset_opt['dataloader_batch_size'],
                                    shuffle=dataset_opt['dataloader_shuffle'],
                                    num_workers=dataset_opt['dataloader_num_workers'],
                                    drop_last=True,
                                    pin_memory=True)

    elif phase == 'test':
        test_set = define_Dataset(dataset_opt)
        # Keep only one third of the dataset
        indexes = torch.randperm(len(test_set))[:len(test_set)//2]
        test_set = Subset(test_set, indexes)
        message = f'Validation dataset of {len(test_set)} images.'
        logger.info(message)
        val_loader = DataLoader(test_set, batch_size=1,
                                    shuffle=False, num_workers=1,
                                    drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError("Phase [%s] is not recognized." % phase)

message = f'Datasets loaded.'
logger.info(message)

dataset = {'train':train_loader, 'val':val_loader}

# Define model function with optuna hyperparameters
def define_model(opt):

    # Initialize model
    model = define_Model(opt)
    model.init_train()

    return model

def define_metric(metric_str):

    metric_dict = {}

    if metric_str == 'PSNR':
        metric_dict['func'] = util.calculate_psnr
        metric_dict['direction'] = 'maximize'
        metric_dict['name'] = 'PSNR'

    elif metric_str == 'SSIM':
        metric_dict['func'] = util.calculate_ssim
        metric_dict['direction'] = 'maximize'
        metric_dict['name'] = 'SSIM'
    
    # TODO IMPLEMENTAR
    # elif metric_str == 'CER':
    #     metric_dict['func'] = utilOCR.calculate_cer
    #     metric_dict['direction'] = 'minimize'
    #     metric_dict['name'] = 'CER'

    elif metric_str == 'edgeJaccard':
        metric_dict['func'] = util.calculate_edge_jaccard
        metric_dict['direction'] = 'maximize'
        metric_dict['name'] = 'edgeJaccard'

    else:
        # If none of above, choose MSE
        metric_dict['func'] = nn.MSELoss()
        metric_dict['direction'] = 'minimize'
        metric_dict['name'] = 'MSE'
    
    return metric_dict

def train_model(trial, model, dataset, metric_dict, num_epochs=25):

    # Load dataset and metrics
    train_loader = dataset['train']
    val_loader = dataset['val']

    metric = metric_dict['func']
    metric_direction = metric_dict['direction']

    best_metric = -1e6*(metric_direction=='maximize') + 1e6*(metric_direction=='minimize')

    current_step = 0

    # Time tracker
    since = time.time()

    # Iter over epoch
    for epoch in range(num_epochs):

        epoch_loss = 0.0

        # epoch_metric = 0.0

        # -------------------------------
        # Training phase
        # ------------------------------- 

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
            # 4) training information (loss and metric)
            # -------------------------------

            # visuals = model.current_visuals()
            # E_visual = visuals['E']
            # E_img = util.tensor2uint(E_visual)
            # H_visual = visuals['H']
            # H_img = util.tensor2uint(H_visual)

            # epoch_metric += metric(H_img, E_img)

            epoch_loss += model.current_log()['G_loss']     

        # Train loss and metric
        avg_train_loss = epoch_loss / idx
        # avg_train_metric = epoch_metric/train_size

        message_train = f'\nepoch:{epoch+1}/{num_epochs}\n'+'-'*14+'\ntrain loss: {:.3e}\n'.format(avg_train_loss)


        # -------------------------------
        # Validation phase
        # -------------------------------
        val_metric = 0.0
        avg_val_loss = 0.0
        idx = 0

        for val_data in val_loader:
            idx += 1

            model.feed_data(val_data)
            model.test()

            visuals = model.current_visuals()
            E_visual = visuals['E']
            E_img = util.tensor2uint(E_visual)
            H_visual = visuals['H']
            H_img = util.tensor2uint(H_visual)

            sizes = E_visual.size()

            current_loss = model.G_lossfn(torch.reshape(E_visual,(1,1,sizes[1],sizes[2])),
                                          torch.reshape(H_visual,(1,1,sizes[1],sizes[2])))

            avg_val_loss += current_loss
            val_metric += metric(H_img, E_img)

        # Val loss and metric
        avg_val_loss = avg_val_loss/idx
        avg_val_metric = val_metric/idx

        message_val = 'val loss: {:.3e}, val {}: {:.3f}\n'.format(avg_val_loss,
                                                                    metric_dict['name'],
                                                                    avg_val_metric
                                                                    )
        # Write epoch log
        logger.info(message_train + message_val +'-'*14)

        # Update if validation metric is better (lower when minimizing, greater when maximizing)
        maximizing = ( (avg_val_metric > best_metric) and metric_dict['direction'] == 'maximize')
        minimizing = ( (avg_val_metric < best_metric) and metric_dict['direction'] == 'minimize') 

        val_metric_is_better = maximizing or minimizing                       

        if val_metric_is_better:
                        best_metric = avg_val_metric
                        # best_model_wts = copy.deepcopy(model.state_dict())
        
        # Report trial epoch and check if should prune
        trial.report(avg_val_metric, epoch)
        if trial.should_prune():
            time_elapsed = time.time() - since
            logger.info('Pruning trial number {}. Used {:.0f}hs {:.0f}min {:.0f}s on pruned training ¯\_(ツ)_/¯'.format(
                trial.number ,time_elapsed // (60*60), (time_elapsed // 60)%60, time_elapsed % 60)
                )
            raise optuna.TrialPruned()


    # Whole optuna parameters searching time
    time_elapsed = time.time() - since
    logger.info('Trial {}: training completed in {:.0f}hs {:.0f}min {:.0f}s'.format(
        trial.number ,time_elapsed // (60*60), (time_elapsed // 60)%60, time_elapsed % 60))

    return best_metric

# Define optuna objective function
def objective(trial):

    # Set learning rate suggestions for trial
    trial_lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    opt['train']['G_optimizaer_lr'] = trial_lr

    trial_tvweight = trial.suggest_float("tv_weight", 1e-13, 1e-6, log=True)
    opt['train']["G_tvloss_weight"] = trial_tvweight

    message = f'Trial number {trial.number} with parameters:\n'
    message = message+f'lr = {trial_lr}\n'
    message = message+f'tv_weight = {trial_tvweight}'

    logger.info(message)

    # Generate the model and optimizers
    model = define_model(opt)

    # Select metric specified at options
    metric_dict = define_metric(opt['optuna']['metric'])

    best_metric = train_model(trial, model, dataset, metric_dict, num_epochs=opt['optuna']['trial_epochs'])    
    
    # Save best model for each trial
    # torch.save(best_model.state_dict(), f"model_trial_{trial.number}.pth")

    # Return metric (Objective Value) of the current trial

    return best_metric

def save_optuna_info(study):

    root_dir = opt['path']['root']

    # Save page for plot contour
    fig = optuna.visualization.plot_contour(study, params=['tv_weight','lr'])
    fig.write_html(os.path.join(root_dir,'optuna_plot_contour.html'))
    # Save page for plot slice
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(os.path.join(root_dir,'optuna_plot_slice.html'))
    # Save page for hyperparameters importances
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(root_dir,'optuna_plot_param_importances.html'))
    # Save page for optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(root_dir,'optuna_plot_optimization_history.html'))
    # Save page for intermediate values plot
    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_html(os.path.join(root_dir,'optuna_plot_intermediate_values.html'))
    # Save page for parallel coordinate plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(root_dir,'optuna_plot_parallel_coordinate.html'))

    return


'''
# ----------------------------------------
# Step--3 (setup optuna hyperparameter search)
# ----------------------------------------
'''
metric_dict = define_metric(opt['optuna']['metric'])
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner( 
            n_startup_trials=10, n_warmup_steps=4, interval_steps=2
        ),
        direction=metric_dict['direction'])

study.optimize(func=objective, n_trials=opt['optuna']['n_trials'])

message = 'Best trial:\n'+str(study.best_trial)
logger.info(message)

logger.info('Saving study information at ' + opt['path']['root'])
save_optuna_info(study)

logger.info('Hyperparameters study ended')
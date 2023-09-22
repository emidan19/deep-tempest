import os.path
import argparse
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import optuna
import time

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils import utils_pnp as pnp
from models.drunet.network_unet import UNetRes as net

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
# ------------------------------------------------------------
# Step--2 (create dataloader) TODO: cargar imagen a ajustar
# ------------------------------------------------------------
'''

message = 'Loading images'
logger.info(message)

try:
    H_paths = util.get_image_paths(opt["path"]["images"])
except Exception as e:
     logger.info(f"Error loading images path. Exiting with following exception:\n{str(e)}")
     exit()

message = f'Dataset loaded.'
logger.info(message)

"""  
# ----------------------------
Step--3 load denoiser prior
# ----------------------------
"""
message = 'Loading denoiser model'
logger.info(message)
try:
    # load model
    denoiser_model_path = os.path.join('./drunet/26_G.pth')
    denoiser_model = net(in_nc=1+1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    denoiser_model.load_state_dict(torch.load(denoiser_model_path), strict=True)
    denoiser_model.eval()
    for _, v in denoiser_model.named_parameters():
        v.requires_grad = False
except Exception as e:
     logger.info(f"Error loading denoiser model. Exiting with following exception:\n{str(e)}")
     exit()


# Define model function with optuna hyperparameters
def define_model(opt):

    #######################################
    ### opt tiene las opciones del PnP: ###
    ### *lr, lambda, sigma1, sigma2,    ###
    ### *prior (denoiser),              ###
    ### *iter_data_term, iter_PNP       ###
    #######################################

    #########################################################################
    # LLENAR CON MODELO PLUG AND PLAY ( {y,z0, x0} --> MODEL --> {zk, xk} ) #
    #########################################################################
    
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
    elif metric_str == 'CER':
        metric_dict['func'],_ = util.calculate_cer_wer
        metric_dict['direction'] = 'minimize'
        metric_dict['name'] = 'CER'

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

def train_model(trial, dataset, metric_dict, denoiser_model=denoiser_model, pnp_opt=opt['plugnplay']):



    metric = metric_dict['func']
    metric_direction = metric_dict['direction']

    best_metric = -1e6*(metric_direction=='maximize') + 1e6*(metric_direction=='minimize')

    # Time tracker
    since = time.time()


    idx = 0

    for i, H_path in enumerate(dataset):
        
        idx += 1

        

        # TODO: Get y, x0 and z0 from original image

        # TODO: Run PNP with pnp_opt

        

        avg_metric = # TODO: assign metric value

        # Update if validation metric is better (lower when minimizing, greater when maximizing)
        maximizing = ( (avg_metric > best_metric) and metric_dict['direction'] == 'maximize')
        minimizing = ( (avg_metric < best_metric) and metric_dict['direction'] == 'minimize') 

        val_metric_is_better = maximizing or minimizing                       

        if val_metric_is_better:
                        best_metric = avg_metric
        
        # Report trial epoch and check if should prune
        trial.report(avg_metric, epoch)
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
    trial_lambda = trial.suggest_float("lambda", 1e-3, 1e2)
    opt['plugnplay']['lambda'] = trial_lambda

    trial_iters_pnp = trial.suggest_int("iters_pnp", 3, 10)
    opt['plugnplay']['iters_pnp'] = trial_iters_pnp

    trial_sigma1 = trial.suggest_float("sigma1", 10, 50)
    opt['plugnplay']['sigma1'] = trial_sigma1

    # sigma2 < sigma1. Force it to be 9 stdev less tops
    trial_sigma2 = trial.suggest_float("sigma2", 1, trial_sigma1-9)
    opt['plugnplay']['sigma2'] = trial_sigma2

    message = f'Trial number {trial.number} with parameters:\n'
    message = message+f'lambda = {trial_lambda}\n'
    message = message+f'iters_pnp = {trial_iters_pnp}\n'
    message = message+f'sigma1 = {trial_sigma1}\n'
    message = message+f'sigma2 = {trial_sigma2}'

    logger.info(message)

    # Select metric specified at options
    metric_dict = define_metric(opt['optuna']['metric'])

    best_metric = train_model(trial, H_paths, metric_dict, denoiser_model=denoiser_model, pnp_opt=opt['plugnplay'])    
    
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
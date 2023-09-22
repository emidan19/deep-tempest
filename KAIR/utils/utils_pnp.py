import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

import torch

import utils_image as util
from forward.degradation import *
from forward.degradation import forward
from drunet.network_unet import UNetRes as net
from drunet.utils_image import max_entropy_init

from collections import OrderedDict
import cv2
from torchvision.transforms.functional import gaussian_blur


"""  
########################
# Load DRUNet denoiser #
########################
"""

model_path = os.path.join('./drunet/26_G.pth')
model = net(in_nc=1+1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for _, v in model.named_parameters():
    v.requires_grad = False

def get_alpha_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lam = 0.23):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    # iter_num decreasing values in logarithmic scale in range [10^2.55, 10^49]
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w + modelSigmaS_lin*(1-w))/255.
    # trade off parameter lambda fixed at 0.23
    alphas = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    return alphas, sigmas

def apply_degradation(x, degradation = 'hdmi', sigma_blur = 10/255):
    # choose between hdmi degradation or gaussian blur degradation
    if degradation == 'hdmi':
        T_x = forward(x)
        return T_x
    elif degradation == 'blur':
        x = x.unsqueeze(0)
        T_x = gaussian_blur(x, kernel_size = (5,5), sigma = sigma_blur)
        x = x.squeeze(0)
        T_x = T_x.squeeze(0)
        return T_x
    return print('not hdmi or blur')

# calculate objective function
def data_term_objective_function(degradation, x, y_obs, z_prev,alpha, sigma_blur, k, energy_term_record, alpha_term_record):
    """
    Inputs:
    degradation (char): 'hdmi' for HDMI degradation function. 'blur' for gaussian blur degradation function.
    x (torch tensor shape [H,W]): image to apply degradation function.
    y_obs (torch tensor shape [H,W]): output of degradation function with added noise when input is x_gt.
    z_prev (torch tensor shape [H,W]): output of denoiser prior. [0, 1] Dynamic Range.
    alpha (float): regularization hyperparameter at iteration k.
    sigma_blur (float): standard deviation of gaussian blur when degradation is set to 'blur'.
    k (int): iteration number.

    Output:
    objective_function_value (float): value of the objective function.
    """
    
    # normalize x input
    #x_copy = x.clone() 
    #x_min = torch.min(x_copy)
    #x_max = torch.max(x_copy)
    #x = (x_copy - x_min) / (x_max - x_min)
    

    # apply degradation function
    T_x = apply_degradation(x, degradation, sigma_blur)

    # calculate objective function at k iteration
    energy_term = torch.norm(y_obs - T_x)**2 
    alpha_term = alpha*torch.norm(x - z_prev)**2
    if k == 0:
        print('Energy term at k = {}: {}'.format(k, energy_term))
        print('Alpha term at k = {}: {}'.format(k, alpha_term))

    print('Energy term at k = {}: {}'.format(k, energy_term))
    print('Alpha term at k = {}: {}'.format(k, alpha_term))

    energy_term_record.append(energy_term.detach())
    alpha_term_record.append(alpha_term.detach())
    obj_function = energy_term + alpha_term 
    #obj_function = torch.norm(y_obs - T_x)**2 + alpha*torch.norm(x - z_prev)**2
    #print('En la iteracion k = {}, el termino de energia vale: {} y el termino alpha vale: {}'.format(k, termino_energia, termino_alpha))

    return obj_function

def optimize_data_term(degradation, x_gt, z_k_prev, x_0, y_obs, i, sigma_blur, total_pixels, alpha, max_iter = 5000, eps = 1e-4, lr = 0.1, k_print = 1, plot = True):
    """
    Inputs:
    degradation (char): 'hdmi' for HDMI degradation function. 'blur' for gaussian blur degradation function.
    x_gt (torch tensor shape [H,W]): ground truth image. [0, 1] Dynamic Range.
    z_i_prev (torch tensor shape [H,W]): output of denoiser prior. [0, 1] Dynamic Range.
    x_0 (torch tensor shape [H,W]): initial condition image with parameter 'requires_grad = True'. [0,1] Dynamic range.
    y_obs (torch tensor shape [H,W]): output of degradation function with added noise when input is x_gt.
    i (int): iteration number of Plug and Play optimization.
    sigma_blur (float): standard deviation of gaussian blur when degradation is set to 'blur'.
    total_pixels (float): the total number of pixels in the image.
    alpha (float): hyperparameter of data term optimization.
    max_iter (int): maximum number of iterations.
    eps (float): minimum tolerance for algorithm to stop.
    lr (float): learning rate of iterative algorithm.
    k_print (int): print results for multiples of k_print iterations.
    plot (boolean): if True, plot OF_record, diff_x_record and diff_x_gt_record.

    Output:
    x_opt (torch tensor shape [H,W]): image result of optimizing objective function. [0, 1] Dynamic Range.  
    """

    # store |x^{k+1} - x^{k}| 
    diff_x_data_term_record = []

    # store jaccard score between x_gt and x_opt 
    jacc_score_data_term_record = []
    
    # store |x^{k} - x_gt| 
    diff_x_gt_data_term_record = []

    # store |y - T(x_k)|^2 
    energy_term_record = []
    
    # store alpha*|x - z_prev|^2 
    alpha_term_record = []

    # store objective function values at step k
    OF_record = []
    
    x_opt = x_0.detach()

    x_opt.requires_grad = True

    # print initial condition image x_0
    plt.figure(figsize = (10,8))
    plt.imshow(x_opt.detach(), cmap = 'gray')
    plt.title('Data term initial condition x_0')
    plt.show()

    # define optimizer
    optimizer = torch.optim.Adam([x_opt], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, eps = 1e-10, verbose = True)

    # store initial learning rate
    lr_0 = lr

    k = 0
    while (k < max_iter) and (lr >= lr_0 / 2**3):

        x_prev = x_opt.detach().clone()
        # empty gradients
        optimizer.zero_grad()
        # calculate objective function
        objective_func = data_term_objective_function(degradation, x_opt, y_obs, z_k_prev, alpha, sigma_blur, k, energy_term_record, alpha_term_record)
        # backpropagation of gradients
        objective_func.backward(retain_graph=True)
        #print(x_opt.grad)
        # update x_opt
        optimizer.step()
        scheduler.step(objective_func)

        OF_record.append(objective_func.detach())

        x_next = x_opt.detach().clone()

        # calculate |x_k+1 - x_k|
        diff_x = torch.norm(x_next - x_prev).detach()
        diff_x_data_term_record.append(diff_x)

        # calculate |x_k - x_gt|
        diff_x_gt = torch.norm(x_next - x_gt).detach()
        diff_x_gt_data_term_record.append(diff_x_gt)

        print("k: {}".format(k), end="\r", flush=True)
        k = k + 1

        if (k % k_print == 0):
            
            print("k: {}".format(k), end="\r", flush=True)
            print('Objective function = {}'.format(objective_func / total_pixels))
            print('diff_x_gt: {}'.format(diff_x_gt / total_pixels))
            print('diff_x: {}'.format(diff_x / total_pixels))
            # print('Jaccard score between x_gt and x_opt: {}'.format(jaccard_score))
            print('Energy term: {}'.format(energy_term_record[-1]))
            print('Alpha term: {}'.format(alpha_term_record[-1]))
            print('-----------')

            _, axes = plt.subplots(1, 2, figsize=(12, 10))
            axes[0].imshow(x_opt.detach().numpy(), cmap = 'gray')
            axes[0].set_title("x_opt. Iteration = {}".format(k))
            axes[1].imshow(x_gt.detach().numpy(), cmap = 'gray')
            axes[1].set_title("x_gt")
            plt.tight_layout()
            plt.show()

            # normalize x input
            x_copy = x_opt.clone() 
            x_min = torch.min(x_copy)
            x_max = torch.max(x_copy)
            x_copy = (x_copy - x_min) / (x_max - x_min)

            y_opt = apply_degradation(x_copy, degradation, sigma_blur)
            y_opt_abs = y_opt.abs()
            y_obs_abs = y_obs.abs()
            y_opt_show = (y_opt_abs - y_opt_abs.min()) / (y_opt_abs.max() - y_opt_abs.min())
            
            y_obs_show = (y_obs_abs - y_obs_abs.min()) / (y_obs_abs.max() - y_obs_abs.min())

            _, axes = plt.subplots(1, 2, figsize=(12, 10))
            axes[0].imshow(y_opt_show.detach().numpy(), cmap = 'gray')
            axes[0].set_title("T(x_k). Iteration = {}".format(k))
            axes[1].imshow(y_obs_show.detach().numpy(), cmap = 'gray')
            axes[1].set_title("y_obs")
            plt.tight_layout()
            plt.show()
        
            plt.tight_layout()
            plt.show()
    
    if plot:
        inicio = 0

        plt.figure(figsize = (7,5))
        plt.plot(OF_record[inicio:] / total_pixels, 'r')
        plt.grid()
        plt.title('Objective Function/total_pixels at iteration i = {}'.format(i))
        plt.show()

        plt.figure(figsize = (7,5))
        plt.plot(diff_x_gt_data_term_record[inicio:]/ total_pixels, 'b')
        plt.grid()
        plt.title('|x_k - x_gt|/total_pixels, at iteration i = {}'.format(i))
        plt.show()

        plt.figure(figsize = (7,5))
        plt.plot(diff_x_data_term_record[inicio:]/ total_pixels, 'g')
        plt.grid()
        plt.title('|x_k+1 - x_k|/total_pixels, at iteration i = {}'.format(i))
        plt.show()

        plt.figure(figsize = (7,5))
        plt.plot(energy_term_record[inicio:] / total_pixels, 'g', label = 'energy term')
        plt.plot(alpha_term_record[inicio:] / total_pixels, 'r', label = 'alpha term')
        plt.grid()
        plt.title('|y - T(x_k)|^2 and alpha*|x - z_prev|^2 at iteration i = {}'.format(i))
        plt.legend()
        plt.show()
    
    return x_opt

def observation(degradation, x, noise_level_img, sigma_blur):
    y = apply_degradation(x, degradation, sigma_blur)
    y = y + torch.normal(mean = 0, std = noise_level_img, size = (y.shape[0], y.shape[1]))
    return y

def plug_and_play_optimization(degradation, x_gt, z_0, x_0, denoiser_prior, noise_level_img, modelSigma1, modelSigma2, lam, sigma_blur = 3,num_iter = 5000, max_iter_data_term = 1000,  eps_data_term = 1e-4, lr = 0.1, k_print_data_term = 10):
    """
    Inputs:
    degradation (char): 'hdmi' for HDMI degradation function. 'blur' for gaussian blur degradation function.
    x_gt (torch tensor shape [H,W]): ground truth image. [0,1] Dynamic range.
    z_0 (torch tensor shape [H,W]): initial condition image for Plug and Play algorithm. [0,1] Dynamic range.
    x_0 (torch tensor shape [H,W]): initial condition image with parameter 'requires_grad = True' for data term optimization subproblem. [0,1] Dynamic range.
    denoiser_prior: deep denoiser model for denoising prior. KAIR's torch model class
    noise_level_img (float): image noise level.
    modelSigma1 (float): upper bound of noise level interval for sigma_k.
    modelSigma2 (float): lower bound of noise level interval for sigma_k.
    sigma_blur (float): standard deviation of gaussian blur when degradation is set to 'blur'.
    lam (float): penalty parameter in regularization term.
    num_iter (int): number of iterations of Plug and Play algorithm.
    max_iter_data_term (int): maximum number of iterations for data term optimization subproblem.
    eps_data_term (int): minimum tolerance for data term optimization subproblem to stop.
    lr (float): learning rate of data term optimization subproblem.
    k_print_data_term (int): print results for multiples of k_print_data_term iterations.

    Output:
    x_opt (torch tensor shape [H,W]): image result of optimizing objective function. [0,1] Dynamic range.
    """

    total_pixels = x_gt.shape[0] * x_gt.shape[1]

    # store |z_k+1 - z^k| 
    diff_z_record = []
    
    # store |z_k - x_gt| 
    diff_x_gt_record = []
    
    # store jaccard score between x_gt and z_opt 
    #jacc_score_record = []
    
    # print initial condition image z_0
    plt.figure(figsize = (10,8))
    plt.imshow(z_0.detach(), cmap = 'gray')
    plt.title('Initial condition z_0')
    plt.show()

    z_opt = z_0

    x_0_data_term = x_0 

    noise_level_model = noise_level_img   
    # generate observation y_obs
    y_obs = observation(degradation, x_gt, noise_level_model, sigma_blur)
    # precalculation of parameters for each iteration
    alphas, sigmas = get_alpha_sigma(sigma=max(0.255/255., noise_level_model), iter_num = num_iter, modelSigma1 = modelSigma1, modelSigma2 = modelSigma2, w = 1.0, lam = lam)
    alphas, sigmas = torch.tensor(alphas), torch.tensor(sigmas)

    print('los valores de alpha son: {}'.format(alphas))
    print('los valores de sigma son: {}'.format(sigmas))

    # iterate algorithm num_iter times
    for i in range(num_iter):
        
        print('Plug & Play {} iteration'.format(i))
        
        z_prev = z_opt.detach().clone()

        # optimize data term
        #alphas[i] = alpha
        #print('alpha = {}. Iteration {}'.format(alphas[i], i))

        print(x_0_data_term.requires_grad)
        x_i = optimize_data_term(degradation, x_gt, z_opt, x_0_data_term, y_obs, i, sigma_blur, total_pixels, alpha = alphas[i], max_iter = max_iter_data_term, eps = eps_data_term, lr = lr, k_print = k_print_data_term, plot = True)
        # output of optimize_data_term is float32 [H,W] array in range [0.0, 1.0]
        #print('Dynamic Range of output at iteration {} of optimize_data_term = [{}, {}]'.format(i, x_i.min(), x_i.max()))
        
        # initial condition of data term optimization in k'th iteration of plug&play algorithm is the solution of data term optiization in k-1'th iteration of plug&play
        x_0_data_term = x_i

        # adjust dimensions
        x_i = x_i.detach().numpy()
        # [H,W] --> [H, W, 1]
        x_i = np.expand_dims(x_i, axis=2)
        x_i_dim4 = util.single2tensor4(x_i)
        x_i_dim4 = torch.cat((x_i_dim4, torch.FloatTensor([sigmas[i]]).repeat(1, 1, x_i_dim4.shape[2], x_i_dim4.shape[3])), dim=1)

        # forward denoiser model
        print('Enter Forward. Sigma = {}. Iteration {}'.format(sigmas[i], i))
        z_opt = denoiser_prior(x_i_dim4)
        z_opt = z_opt[0,0,:,:]
        #print('Dynamic Range of output of cnn at iteration {} is [{}, {}]'.format(i, z_opt.min(), z_opt.max()))

        # normalize z_opt between [0, 1]. [H,W]
        min_z_opt = z_opt.min()
        max_z_opt = z_opt.max()
        z_opt = (z_opt - min_z_opt)/(max_z_opt - min_z_opt)
        #print('Dynamic Range of z_opt after normalize at iteration {} is [{}, {}]'.format(i, z_opt.min(), z_opt.max()))

        # print output of denoiser model
        plt.figure(figsize = (10, 6))
        plt.imshow(z_opt, cmap = 'gray')
        plt.title('Output of denoiser at iteration {}'.format(i))
        plt.show()

        z_next = z_opt.detach().clone()
        
        # calculate |z_k+1 - z_k| / total_pixels
        diff_z = torch.norm(z_next - z_prev).detach()
        diff_z_record.append(diff_z)

        # calculate |z_k - x_gt| / total_pixels
        diff_x_gt = torch.norm(z_next - x_gt).detach()
        diff_x_gt_record.append(diff_x_gt)

        # calculate jaccard score between x_gt and z_opt
        #x_gt_8b = single2uint(x_gt.detach())
        #z_opt_8b = single2uint(z_opt.detach())
        #jaccard_score = calculate_edge_jaccard(x_gt_8b, z_opt_8b)
        #jacc_score_record.append(jaccard_score)


    plt.figure(figsize = (7,5))
    plt.plot(diff_x_gt_record / total_pixels, 'b')
    plt.grid()
    plt.title('|z_i - x_gt| / total_pixels')
    plt.show()

    plt.figure(figsize = (7,5))
    plt.plot(diff_z_record / total_pixels, 'g')
    plt.grid()
    plt.title('|z_i+1 - z_i| / total_pixels')
    plt.show()

    #plt.figure(figsize = (7,5))
    #plt.plot(jacc_score_record, 'r')
    #plt.grid()
    #plt.title('Jaccard score between x_gt and z_opt')
    #plt.show()

    return z_opt

def max_entropy_thresh(img, patch_size = 128):
    """  
    Perform maximum entropy threshold by image patches.
    Input image must be torch tensor
    """
    img_np = img.detach().numpy()
    
    v_total,h_total = img_np.shape
    three_channels_img = np.zeros((v_total,h_total,3))

    I_real = np.real(img_np)
    I_imag = np.imag(img_np)

    three_channels_img[:,:,0], three_channels_img[:,:,1] = I_real, I_imag
    min_value, max_value = np.min(three_channels_img[:,:,:2]), np.max(three_channels_img[:,:,:2])
    three_channels_img[:,:,0] = 255*(I_real-min_value)/(max_value-min_value)
    three_channels_img[:,:,1] = 255*(I_imag-min_value)/(max_value-min_value)

    max_entropy_img = max_entropy_init(three_channels_img[:,:,:2].astype('uint8'), patch_size=patch_size)
    max_entropy_img = util.uint2single(max_entropy_img)
    
    return torch.tensor(max_entropy_img, requires_grad = True)
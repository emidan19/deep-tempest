import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

import torch

import utils_image as util
from forward.degradation import forward
from utils_image import max_entropy_init

from torchvision.transforms.functional import gaussian_blur

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
    
    x_heigh, x_width = x.size()
    total_pixels = x_heigh * x_width

    # apply degradation function
    T_x = apply_degradation(x, degradation, sigma_blur)

    # calculate objective function at k iteration
    energy_term = torch.norm(y_obs - T_x)**2 
    alpha_term = torch.norm(x - z_prev)**2

    print('Energy term at k = {}: {}'.format(k, energy_term/total_pixels))
    print('Alpha term at k = {}: {}'.format(k, alpha_term/total_pixels))

    obj_function = energy_term + alpha_term 

    return obj_function, energy_term, alpha_term

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
    OF_record (list): Values of thw objective function over iterations
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

    # store gradient norm at step k
    grad_norm_record = []
    
    x_opt = x_0.detach()

    x_opt.requires_grad = True

    if plot:
        # print initial condition image x_0
        plt.figure(figsize = (10,8))
        plt.imshow(x_opt.clone().detach(), cmap = 'gray')
        plt.title('Data term initial condition x_0')
        plt.show()

    # define optimizer
    optimizer = torch.optim.Adam([x_opt], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1, 
                                                           patience = 5, eps = 1e-10, verbose = True)

    # store initial learning rate
    lr_0 = lr

    # Init pocket reference
    objective_func_ref = np.inf

    k = 0
    while (k < max_iter) and (lr >= lr_0 / 10**3):

        x_prev = x_opt.clone().detach()
        # empty gradients
        optimizer.zero_grad()
        # calculate objective function
        objective_func, energy_term, alpha_term = data_term_objective_function(degradation, x_opt, y_obs, z_k_prev, alpha, sigma_blur, k, energy_term_record, alpha_term_record)
        # backpropagation of gradients
        objective_func.backward(retain_graph=True)
        # Backprop gradient of x, take l2 norm
        grad_norm_record.append(torch.norm(x_opt.grad) / total_pixels)
        # update x_opt
        optimizer.step()
        scheduler.step(objective_func)

        OF_record.append(objective_func.clone().detach())
        energy_term_record.append(energy_term.clone().detach())        
        alpha_term_record.append(alpha_term.clone().detach())

        x_next = x_opt.clone().detach()

        # Keep minimum argument image at the moment
        if objective_func < objective_func_ref:
            x_pocket = x_next.clone()

        # calculate |x_k+1 - x_k|
        diff_x = torch.norm(x_next - x_prev).clone().detach()
        diff_x_data_term_record.append(diff_x)

        # calculate |x_k - x_gt|
        diff_x_gt = torch.norm(x_next - x_gt).clone().detach()
        diff_x_gt_data_term_record.append(diff_x_gt)

        print("k: {}".format(k), end="\r", flush=True)
        k = k + 1

    
    return x_pocket, energy_term_record, alpha_term_record, grad_norm_record

def observation(degradation, x, noise_level_img, sigma_blur):
    y = apply_degradation(x, degradation, sigma_blur)
    noise_inphase = torch.normal(mean = 0, std = noise_level_img, size = (y.shape[0], y.shape[1]))
    noise_inquadr = torch.normal(mean = 0, std = noise_level_img, size = (y.shape[0], y.shape[1]))
    y = y + noise_inphase + 1j*noise_inquadr
    return y

def plug_and_play_optimization(degradation, x_gt, initializacion_from_y, denoiser_prior, noise_level_img, 
                               modelSigma1, modelSigma2, lam, sigma_blur = 3,
                               num_iter = 5000, max_iter_data_term = 1000,  eps_data_term = 1e-4, 
                               lr = 0.1, k_print_data_term = 10):
    """
    Inputs:
    degradation (char): 'hdmi' for HDMI degradation function. 'blur' for gaussian blur degradation function.
    x_gt (torch tensor shape [H,W]): ground truth image. [0,1] Dynamic range.
    initializacion_from_y: function that takes observation image and process to get PnP initialization
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
    


    noise_level_model = noise_level_img   
    # generate observation y_obs
    y_obs = observation(degradation, x_gt, noise_level_model, sigma_blur)
    # precalculation of parameters for each iteration
    alphas, sigmas = get_alpha_sigma(sigma=max(0.255/255., noise_level_model), iter_num = num_iter, modelSigma1 = modelSigma1, modelSigma2 = modelSigma2, w = 1.0, lam = lam)
    alphas, sigmas = torch.tensor(alphas), torch.tensor(sigmas)

    # Get initializations z0 and x0 from observation y
    x_0 = initializacion_from_y(y_obs)
    z_0 = x_0
    print('Dynamic Range of x_0 = [{},{}]'.format(x_0.min(), x_0.max()))

    # print initial condition image z_0
    plt.figure(figsize = (10,8))
    plt.imshow(z_0.clone().detach(), cmap = 'gray')
    plt.title('Initial condition z_0')
    plt.show()

    z_opt = z_0

    x_0_data_term = x_0 

    print('los valores de alpha son: {}'.format(alphas))
    print('los valores de sigma son: {}'.format(sigmas))

    # iterate algorithm num_iter times
    for i in range(num_iter):
        
        print('Plug & Play {} iteration'.format(i))
        
        z_prev = z_opt.clone().detach()

        # optimize data term
        x_i = optimize_data_term(degradation, x_gt, z_opt, x_0_data_term, y_obs, i, sigma_blur, total_pixels, alpha = alphas[i], max_iter = max_iter_data_term, eps = eps_data_term, lr = lr, k_print = k_print_data_term, plot = True)
        
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

        # normalize z_opt between [0, 1]. [H,W]
        min_z_opt = z_opt.min()
        max_z_opt = z_opt.max()
        z_opt = (z_opt - min_z_opt)/(max_z_opt - min_z_opt)

        # print output of denoiser model
        plt.figure(figsize = (10, 6))
        plt.imshow(z_opt, cmap = 'gray')
        plt.title('Output of denoiser at iteration {}'.format(i))
        plt.show()

        z_next = z_opt.clone().detach()
        
        # calculate |z_k+1 - z_k| / total_pixels
        diff_z = torch.norm(z_next - z_prev).clone().detach()
        diff_z_record.append(diff_z)

        # calculate |z_k - x_gt| / total_pixels
        diff_x_gt = torch.norm(z_next - x_gt).clone().detach()
        diff_x_gt_record.append(diff_x_gt)



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
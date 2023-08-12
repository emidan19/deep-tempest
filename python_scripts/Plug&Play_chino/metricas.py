from piq import ssim
from piq.ms_ssim import multi_scale_ssim
from piq.iw_ssim import information_weighted_ssim
from piq.psnr import psnr

def SSIM(x, y):

    return ssim(x.expand(1, 1, x.shape[0], x.shape[1]), y.expand(1, 1, y.shape[0], y.shape[1])).item()

def MS_SSIM(x, y):

    return multi_scale_ssim(x.expand(1, 1, x.shape[0], x.shape[1]), y.expand(1, 1, y.shape[0], y.shape[1])).item()

def IW_SSIM(x, y):

    return information_weighted_ssim(x.expand(1, 1, x.shape[0], x.shape[1]), y.expand(1, 1, y.shape[0], y.shape[1])).item()

def PSNR(x, y):

    return psnr(x.expand(1, 1, x.shape[0], x.shape[1]), y.expand(1, 1, y.shape[0], y.shape[1])).item()

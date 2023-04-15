#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>

Script that reads all images in folder and simulates HDMI tempest capture 

"""

#%%

# =============================================================================
# Imports
# =============================================================================
import os
import time as time
import numpy as np
from skimage.io import imread
from scipy import signal
from PIL import Image
from DTutils import TMDS_encoding_original, TMDS_serial
import sys

#%%

# Currently supporting png, jpg, jpeg, tif and gif extentions only
def get_images_names_from_folder (folder):
    images_list = [image for image in os.listdir(folder) \
                   if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') or \
                       image.endswith('.tif') or image.endswith('.tiff') or image.endswith('.gif')] 
    return images_list

def get_subfolders_names_from_folder(folder):
    subfolders_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return subfolders_list

def image_transmition_simulation(I, blanking=False):
    
    # Encode image for TMDS
    I_TMDS = TMDS_encoding_original (I, blanking = blanking)
    
    # Serialize pixel bits and sum channel signals
    I_TMDS_Tx = TMDS_serial(I_TMDS)
    
    return I_TMDS_Tx, I_TMDS.shape

def image_capture_simulation(I_Tx, h_total, v_total, N_harmonic, noise_std=0, 
                             fps=60):
    
    # Compute pixelrate and bitrate
    px_rate = h_total*v_total*fps
    bit_rate = 10*px_rate

    # Continuous samples (interpolate)
    interpolator = int(np.ceil(N_harmonic/5)) # Condition for sampling rate and
    sample_rate = interpolator*bit_rate
    Nsamples = interpolator*(10*h_total*v_total)
    if interpolator > 1:
        I_Tx_continuous = np.repeat(I_Tx,interpolator)
    else:
        I_Tx_continuous = I_Tx
    
    # Add Gaussian noise
    if noise_std > 0:
        noise_sigma = noise_std/15.968719423 # sqrt(255)~15.968719423
        I_Tx_noisy = I_Tx_continuous + np.random.normal(0, noise_sigma, Nsamples) + 1j*np.random.normal(0, noise_sigma,Nsamples)
    else:
        I_Tx_noisy = I_Tx_continuous
        
    # Continuous time array
    t_continuous = np.arange(Nsamples)/sample_rate

    # AM modulation frequency according to pixel harmonic
    harm = N_harmonic*px_rate

    # Harmonic oscilator
    baseband_exponential = np.exp(2j*np.pi*harm*t_continuous)
    
    usrp_rate = 50e6

    # AM modulation and SDR sampling
    I_Rx = signal.resample_poly(I_Tx_noisy*baseband_exponential,up=int(usrp_rate), down=sample_rate)

    # Reshape signal to the image size
    I_capture = signal.resample(I_Rx, h_total*v_total).reshape(v_total,h_total)
    
    return I_capture

def save_simulation_image(I,path_and_name):
    
    v_total,h_total = I.shape
    
    I_save = np.zeros((v_total,h_total,3))

    I_real = np.real(I)
    I_imag = np.imag(I)
    
    realmax = I_real.max()
    realmin = I_real.min()
    imagmax = I_imag.max()
    imagmin = I_imag.min()

    # Stretch contrast on every channel
    I_save[:,:,0] = 255*(I_real-realmin)/(realmax-realmin)
    I_save[:,:,1] = 255*(I_imag-imagmin)/(imagmax-imagmin)

    im = Image.fromarray(I_save.astype('uint8'))
    im.save(path_and_name)
    
def main():
    
    # Get foldername argument
    foldername = sys.argv[-1]
    
    # Get images and subfolders names
    images = get_images_names_from_folder(foldername)
    
    simulations_folder = foldername+'/simulations/'

    os.mkdir(simulations_folder)
    
    # timestamp for simulation starting
    t1_image = time.time()
    
    for image in images:
        
        # Read image
        image_path = foldername+'/'+image
        I = imread(image_path)
        
        # TMDS coding and bit serialization
        I_Tx, resolution = image_transmition_simulation(I)
        v_res, h_res, _ = resolution

        # Choose random pixelrate harmonic number
        N_harmonic = np.random.randint(1,10)
        
        I_capture = image_capture_simulation(I_Tx, h_res, v_res, N_harmonic)
        
        path = simulations_folder+image
        
        save_simulation_image(I_capture,path)

    # timestamp for simulation ending
    t2_image = time.time()                

    t_images = t2_image-t1_image
        
    print('\nTiempo total de las '+str(len(images))+' simulaciones:','{:.2f}'.format(t_images)+'s\n')
        
if __name__ == "__main__":    
    main()
    
    
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
import logging
import utils_logger
from datetime import datetime

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
                             fps=60, freq_error=0, phase_error=0):
    
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

    # Harmonic oscilator (including frequency and phase error)
    baseband_exponential = np.exp(2j*np.pi*(harm+freq_error)*t_continuous + 1j*phase_error)
    
    usrp_rate = 50e6
    usrp_BW = usrp_rate/2

    # AM modulation and SDR sampling
    I_Rx = signal.resample_poly(I_Tx_noisy*baseband_exponential,up=int(usrp_BW), down=sample_rate)

    # Reshape signal to the image size
    I_capture = signal.resample(I_Rx, h_total*v_total).reshape(v_total,h_total)
    
    return I_capture

def save_simulation_image(I,path_and_name):
    
    v_total,h_total = I.shape
    
    I_save = np.zeros((v_total,h_total,3))

    I_real = np.real(I)
    I_imag = np.imag(I)
    
    I_save[:,:,0], I_save[:,:,1] = I_real, I_imag
    min_value, max_value = np.min(I_save[:,:,:2]), np.max(I_save[:,:,:2])
    I_save[:,:,0] = 255*(I_real-min_value)/(max_value-min_value)
    I_save[:,:,1] = 255*(I_imag-min_value)/(max_value-min_value)

    im = Image.fromarray(I_save.astype('uint8'))
    im.save(path_and_name)
    
def main():

    logs_dir = './logfiles/'
    # Create logs directory if not exist
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    logger_name = 'simulations_'+datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    utils_logger.logger_info(logger_name, logs_dir+logger_name+'.log')
    logger = logging.getLogger(logger_name)


    # Get foldername argument
    foldername = sys.argv[-1]
    
    message = f'Tempest capture simulation for image folder {foldername}\n'
    logger.info(message)

    # Get images and subfolders names
    images = get_images_names_from_folder(foldername)
    
    # Create simulation directory if not exist at the folder path
    simulations_path = foldername+'/simulations/'
    if not os.path.exists(simulations_path):
        os.mkdir(simulations_path)
        message = f'Created simulation directory at {simulations_path}\n'
        logger.info(message)
    
    # Possible noise std values
    # noise_stds = np.array([ 0, 5,  10,  15,  20,  25])

    
    for image in images:
        
        # timestamp for simulation starting
        t1_image = time.time()

        # Read image
        image_path = foldername+'/'+image
        I = imread(image_path)

        # Choose random pixelrate harmonic number
        N_harmonic = np.random.randint(1,10)


        message = f'Initiate simulation for image {image} with {N_harmonic} pixel harmonic frequency'
        logger.info(message)

        
        # TMDS coding and bit serialization
        I_Tx, resolution = image_transmition_simulation(I, blanking=True)
        v_res, h_res, _ = resolution

        I_capture = image_capture_simulation(I_Tx, h_res, v_res, N_harmonic)
        
        path = simulations_path+image
        
        save_simulation_image(I_capture,path)

        # timestamp for simulation ending
        t2_image = time.time()                

        t_images = t2_image-t1_image

        message = 'Processing time: {:.2f}'.format(t_images)+'s\n'
        logger.info(message)

        
if __name__ == "__main__":    
    main()
    
    
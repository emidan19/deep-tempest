#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>

Script that reads all images in folder and simulates HDMI tempest capture 

"""

# =============================================================================
# Imports
# =============================================================================
import os
import json
import time as time
import numpy as np
from skimage.io import imread
from scipy import signal
from PIL import Image
from DTutils import TMDS_encoding_original, TMDS_serial
import logging
# from utils import utils_logger
from datetime import datetime

# Currently supporting png, jpg, jpeg, tif and gif extentions only
def get_images_names_from_folder (folder):
    images_list = [image for image in os.listdir(folder) \
                   if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') or \
                       image.endswith('.tif') or image.endswith('.tiff') or image.endswith('.gif') or image.endswith('.bmp')] 
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

def image_capture_simulation(I_Tx, h_total, v_total, N_harmonic, sdr_rate = 50e6,
                             noise_std=0, fps=60, freq_error=0, phase_error=0):
    
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

    # AM modulation and SDR sampling
    I_Rx = signal.resample_poly(I_Tx_noisy*baseband_exponential,up=int(sdr_rate), down=sample_rate)

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
    
def main(simulation_options_path = 'options/tempest_simulation.json'):

    # Load JSON options file
    options = json.load(open(simulation_options_path))

    logs_dir = 'logfiles/'
    # Create logs directory if not exist
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    # Get input and output dirs
    input_folder = options['paths']['folder_original_images']

    logger_name = 'simulations_'+datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    utils_logger.logger_info(logger_name, os.path.join(logs_dir,logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    message = f'Tempest capture simulation for image folder {input_folder}\n'
    logger.info(message)

    # Check input directory exists
    if not(os.path.exists(input_folder)):
        message = f'No input folder {input_folder} was found. Exiting\n'
        logger.info(message)
        exit()

    # Create output simulation directory if not exists
    output_folder = options['paths']['folder_simulated_images']
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
        message = f'Created simulation directory at {output_folder}\n'
        logger.info(message)

    message = f'Tempest options:\n {options} \n'
    logger.info(message)

    # Get tempest options
    blanking = options['options']['blanking']
    fps = options['options']['frames_per_second']
    sdr_rate = options['options']['sdr_rate']
    harmonics = options['options']['random']['harmonics']
    freq_error_range = options['options']['random']['freq_error']
    phase_error_range = options['options']['random']['phase_error']
    sigma = options['options']['random']['sigma']

    # Process possible sigma types
    if type(sigma) == list:
        sigma = np.random.randint(sigma[0],sigma[1])
    elif sigma is None:
        sigma = 0

    # Get images and subfolders names
    images = get_images_names_from_folder(input_folder)
    
    # Initialize processing time
    t_all_images = 0

    for image in images:
        
        # timestamp for simulation starting
        t1_image = time.time()

        # Read image
        image_path = os.path.join(input_folder,image)
        I = imread(image_path)

        # Random channel effects
        freq_error = np.random.randint(freq_error_range[0], freq_error_range[1])
        phase_error = np.random.uniform(phase_error_range[0], phase_error_range[1])*np.pi
        
        # Choose random pixel rate harmonic number
        N_harmonic = np.random.choice(harmonics)

        message = f'Initiate simulation for image "{image}" with {N_harmonic} pixel harmonic frequency, {freq_error} Hz and {phase_error} rads error.'
        logger.info(message)

        
        # TMDS coding and bit serialization
        I_Tx, resolution = image_transmition_simulation(I, blanking=blanking)
        v_res, h_res, _ = resolution

        I_capture = image_capture_simulation(I_Tx, h_res, v_res, N_harmonic, sdr_rate,
                                             sigma, fps, freq_error, phase_error)
        
        path = os.path.join(output_folder,image)
        
        save_simulation_image(I_capture,path)

        # timestamp for simulation ending
        t2_image = time.time()                

        t_image = t2_image-t1_image

        t_all_images += t_image

        message = 'Processing time: {:.2f}'.format(t_image)+'s\n'
        logger.info(message)

    # message = 'Total processing time for {} images: {:.2f}'.format(len(images),t_all_images)+'s\n'
    message = 'Total processing time for {} images: {:.2f}s \n'.format(len(images),t_all_images)
    logger.info(message)
if __name__ == "__main__":    
    main()
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>
"""

#%%

# =============================================================================
# Importar librerías
# =============================================================================
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy import signal
import time
from PIL import Image
from DTutils import TMDS_encoding, TMDS_serial
import sys

#%%

# Currently supporting png, jpg, jpeg, tif, tiff and gif extentions only
def get_images_names_from_folder (folder):
    images_list = [image for image in os.listdir('images') \
                   if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') or \
                       image.endswith('.tif') or image.endswith('.tiff') or image.endswith('.gif')]
    return images_list

def get_subfolders_names_from_folder(folder):
    subfolders_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return subfolders_list

def HDMI_capture_image_simulation(I, N_harmonic, SNR, fps=60):
    
    # Encode image for TMDS
    I_TMDS = TMDS_encoding (I, blanking = False)
    
    # Serialize pixel bits and
    
    return


# =============================================================================
# Levantar imagen y desplegarla
# =============================================================================

filename = sys.argv[-1]
I = imread(filename)
# # Quedarse con el nombre sin la extensión del archivo
filename = filename.split('/')[-1].split('.')[0]
# plt.figure()
# plt.imshow(I, cmap='gray')
# plt.show(block=False)

# Resolución de la imagen
v_active, h_active = I.shape[:2]

N_harmonic = input('Choose pixel rate harmonic number: ')
N_harmonic = int(N_harmonic)

SNR = input('Choose SNR in dB (zero or negative for no noise): ')
SNR = int(SNR)

#%%

# =============================================================================
# Codificación de la señal y transmisión
# =============================================================================

# Codificación TMDS
t1_TMDS = time.time()
I_TMDS = TMDS_encoding (I, blanking = False)
t2_TMDS = time.time()

# plt.figure()
# plt.imshow(I_TMDS/np.max(I_TMDS), cmap='gray')

# Serialización y efecto de superposición de canales RGB
t1_serial = time.time()
I_TMDS_Tx = TMDS_serial(I_TMDS)
t2_serial = time.time()

#%%

# Cálculo de tiempos
t_delay_TMDS = t2_TMDS - t1_TMDS
t_delay_serial = t2_serial - t1_serial
print('La codificación demora',t_delay_TMDS,'segundos')
print('La serialización demora',t_delay_serial,'segundos')

#%%

# =============================================================================
#  Efecto de ruido y llevada a bandabase
# =============================================================================

# Resolución (con blanking)
v_total, h_total = I_TMDS.shape[:2]

# Frame rate y tasa de pixels/bits
fps = 60
px_rate = h_total*v_total*fps
bit_rate = 10*px_rate

# Muestras que hacen efecto de contínuo (interpolando)
interpolator = 1
sample_rate = interpolator*bit_rate
Nsamples = interpolator*(10*h_total*v_total)
I_TMDS_Tx_continuous = np.repeat(I_TMDS_Tx,interpolator)

# Adición de ruido Gaussiano especificada la SNR

if SNR > 0:
    signal_power = np.sum(np.abs(np.fft.fft(I_TMDS_Tx_continuous)**2/Nsamples))
    noise_sigma = np.sqrt(signal_power/(10**(SNR/10)))
    I_TMDS_Tx_noisy = I_TMDS_Tx_continuous + np.random.normal(0, noise_sigma, Nsamples)
else:
    I_TMDS_Tx_noisy = I_TMDS_Tx_continuous

# =============================================================================
# Captura de la señal
# =============================================================================


# Tiempo continuo de transmisión de bits
t_continuous = np.arange(Nsamples)/sample_rate

# Armónico elegido para centrar el espectro
harm = N_harmonic*px_rate

# Llevada a bandabase

baseband_exponential = np.exp(2j*np.pi*harm*t_continuous)

#%%

# =============================================================================
# Reconstrucción de la imagen
# =============================================================================

# Tasa de muestreo del SDR
usrp_rate = 50e6

# Muestreo de señal analógica
I_Rx = signal.resample_poly(I_TMDS_Tx_noisy*baseband_exponential,up=int(usrp_rate), down=sample_rate)

# Muestreo a nivel de píxel y
I_reconst_px = signal.resample(I_Rx, h_total*v_total).reshape(v_total,h_total)

# plt.figure()
# plt.imshow(np.abs(I_reconst_px), cmap='gray')
# plt.show()

#%%

# Guardar imagen real e imaginaria, dejar valores entre 0 y 255

I_reconst_px_norm = np.abs(I_reconst_px)
I_reconst_px_norm = I_reconst_px_norm - np.min(I_reconst_px_norm)
I_reconst_px_norm = 255*I_reconst_px_norm/np.max(I_reconst_px_norm)

# Guardar las partes reales e imageniarias ponderadas por el módulo estirado entre 0 y 255
I_save = np.zeros((v_total,h_total,3))
I_save[:,:,0] = 255*(np.real(I_reconst_px)-np.real(I_reconst_px).min())/(np.real(I_reconst_px).max()-np.real(I_reconst_px).min())
I_save[:,:,1] = 255*(np.imag(I_reconst_px)-np.imag(I_reconst_px).min())/(np.imag(I_reconst_px).max()-np.imag(I_reconst_px).min())


filename_save = filename+str(N_harmonic)+'harm_'+str(SNR)+"dB_HMDI_capture_simulation.tif"
im = Image.fromarray(I_save.astype('uint8'))
im.save(filename_save)

#%%

# Levantar la imagen
I_simu = imread(filename_save)

I_simu_norm = np.abs(I_simu[:,:,0]+1j*I_simu[:,:,1])


plt.figure()

plt.title('Simulación')
plt.imshow(I_reconst_px_norm,cmap='gray')


plt.figure()

plt.title('Simulación levantada')
plt.imshow(I_simu_norm,cmap='gray')

plt.show()


# =============================================================================
# Decodificación de la imagen
# =============================================================================

# Dejar valores entre 0 y 1023

# I_reconst_TMDS = I_reconst_px_norm*1023/255

# t1_TMDS_decoding = time.time()
# I_reconst_TMDS_decoded = TMDS_decoding(I_reconst_TMDS.astype('uint16'))
# t2_TMDS_decoding = time.time()

# plt.figure()
# plt.imshow(I_reconst_TMDS_decoded, cmap='gray')
# plt.show()

# print('La decodificación demora',t2_TMDS_decoding - t1_TMDS_decoding,'segundos')

# im = Image.fromarray(np.squeeze(I_reconst_TMDS_decoded.astype('uint8')))
# im.save(filename+"HMDI_capture_simulation_decoded.png")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>
"""

##%%

# =============================================================================
# Importar librerías
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy import signal
import time
from PIL import Image
from DTutils import TMDS_encoding, TMDS_serial
import sys

##%%

# =============================================================================
# Levantar imagen de prueba
# =============================================================================

# Cargar imagen prueba
# I=imread('../images/VAMO!!.png')
# I=imread('../images/test.jpeg')
# I=imread('../images/black_cube_800x600.jpg')
# I=imread('../images/1920x1080_test.png')
# I=imread('../images/1920x1080_test2.png')

filename = sys.argv[-1]
I = imread(filename)
plt.figure()
plt.imshow(I)
# plt.show(block=False)

# Resolución de la imagen
v_active, h_active = I.shape[:2]

##%%

# =============================================================================
# Codificación de la señal y transmisión
# =============================================================================

# Codificación TMDS
t1_TMDS = time.time()
I_TMDS = TMDS_encoding (I, blanking = True)
t2_TMDS = time.time()

# Serialización y efecto de superposición de canales RGB
t1_serial = time.time()
I_TMDS_Tx = TMDS_serial(I_TMDS)
t2_serial = time.time()

##%%

# Calculo de tiempos
t_delay_TMDS = t2_TMDS - t1_TMDS
t_delay_serial = t2_serial - t1_serial
print('La codificación demora',t_delay_TMDS,'segundos')
print('La serialización demora',t_delay_serial,'segundos')
print('Duración total:',t_delay_serial+t_delay_TMDS)

##%%

# =============================================================================
#  Efecto de ruido y llevada a bandabase
# =============================================================================

# Resolución (con blanking)
v_total, h_total = I_TMDS.shape[:2]

# Frame rate y tasa de pixels/bits
fps = 60
px_rate = h_total*v_total*fps
bit_rate = 10*px_rate
Nsamples = 10*h_total*v_total

# Elección de ruido dado SNR

SNR = 75

if SNR > 0:
    signal_power = np.sum(np.abs(np.fft.fft(I_TMDS_Tx))**2/Nsamples)
    noise_sigma = np.sqrt(signal_power/(10**(SNR/10)))
    I_TMDS_Tx_noisy = I_TMDS_Tx + np.random.normal(0, noise_sigma, Nsamples)
else:
    I_TMDS_Tx_noisy = I_TMDS_Tx

# =============================================================================
# Bandabase
# =============================================================================


# Tiempo continuo de transmisión de bits
t_continuous = np.arange(len(I_TMDS_Tx))/bit_rate

# Armónico elegido para centrar el espectro
N_harm = 1
harm = N_harm*px_rate

# Llevada a bandabase
baseband_exponential = np.exp(2j*np.pi*harm*t_continuous)

##%%

# Tasa de muestreo del SDR
usrp_rate = 50e6

# Muestreo de señal analógica
I_Rx = signal.resample_poly(I_TMDS_Tx_noisy*baseband_exponential,up=int(usrp_rate), down=bit_rate)

# Muestreo a nivel de píxel y
I_reconst_px = signal.resample(I_Rx, h_total*v_total).reshape(v_total,h_total)

plt.figure()
plt.imshow(np.abs(I_reconst_px), cmap='gray')
# plt.show(block=False)

##%%
I_reconst_px_norm = np.abs(I_reconst_px)
I_reconst_px_norm = I_reconst_px_norm - np.min(I_reconst_px_norm)
I_reconst_px_norm = 255 - 255*I_reconst_px_norm/np.max(I_reconst_px_norm)

im = Image.fromarray(I_reconst_px_norm.astype('uint8'))
im.save("HMDI_capture_simulation.png")

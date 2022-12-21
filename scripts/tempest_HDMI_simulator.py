#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>
"""

#%%

# Imports
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy import signal
import time
from PIL import Image
from DTutils import TMDS_encoding, TMDS_serial

#%%

# Cargar imagen prueba
I=imread('../images/VAMO!!.png')
# I=imread('../images/black_cube_800x600.jpg')
# I=imread('../images/1920x1080_test.png')
# I=imread('../images/1920x1080_test2.png')
plt.figure()
plt.imshow(I)
plt.show()

v_active, h_active = I.shape[:2]

#%%

# Codificación TMDS
t1_TMDS = time.time()
I_TMDS = TMDS_encoding (I, blanking = True)
t2_TMDS = time.time()
# Serialización y efecto de superposición de canales RGB
t1_serial = time.time()
I_TMDS_Tx = TMDS_serial(I_TMDS)
t2_serial = time.time()

#%%

t_delay_TMDS = t2_TMDS - t1_TMDS
t_delay_serial = t2_serial - t1_serial
print('La codificación demora',t_delay_TMDS,'segundos')
print('La serialización demora',t_delay_serial,'segundos')
print('Duración total:',t_delay_serial+t_delay_TMDS)

#%%

# Resolución (con blanking)
v_total, h_total = I_TMDS.shape[:2]

fps = 60
px_rate = h_total*v_total*fps
bit_rate = 10*px_rate
# Tiempo continuo de transmisión de bits
t_continuous = np.arange(len(I_TMDS_Tx))/bit_rate

# Armónico elegido para centrar el espectro
N_harm = 1
harm = N_harm*px_rate

# Llevada a bandabase
baseband_exponential = np.exp(2j*np.pi*harm*t_continuous)

#%%

# Tasa de muestreo del SDR
usrp_rate = 50e6

# Muestreo de señal analógica
I_Rx = signal.resample_poly(I_TMDS_Tx*baseband_exponential,up=int(usrp_rate), down=bit_rate)

# Muestreo a nivel de píxel y
I_reconst_px = signal.resample(I_Rx, h_total*v_total).reshape(v_total,h_total)

plt.figure()
plt.imshow(np.abs(I_reconst_px), cmap='gray')
plt.show()

#%%

I_reconst_px_norm = I_reconst_px - np.min(I_reconst_px)
I_reconst_px_norm = 255*I_reconst_px_norm/np.max(I_reconst_px_norm)

im = Image.fromarray(I_reconst_px_norm.astype('uint8'))
im.save("HMDI_capture_simulation.png")

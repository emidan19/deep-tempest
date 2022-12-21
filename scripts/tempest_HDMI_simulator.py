#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 2022

@author: Emilio Martínez <emilio.martinez@fing.edu.uy>
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from scipy import signal
import time
from PIL import Image
from DTutils import TMDS_encoding, TMDS_serial
#%%
I=imread('../images/VAMO!!.png')

I_TMDS = TMDS_encoding (I, blanking = True)
#%%
v_total, h_total = I_TMDS.shape[:2]
#%%
I_TMDS_Tx = TMDS_serial(I_TMDS)
#%%
N_harm = 1
fps = 60
px_rate = h_total*v_total*fps
bit_rate = 10*px_rate

t_continuous = np.arange(len(I_TMDS_Tx))/bit_rate

harm = N_harm*px_rate


baseband_exponential = np.exp(-2j*np.pi*harm*t_continuous)
#%%

usrp_rate = 50e6

I_Rx = signal.resample_poly(I_TMDS_Tx*baseband_exponential,up=usrp_rate, down=bit_rate)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Manual  Simulated Tempest TMDS Example
# GNU Radio version: 3.8.5.0

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from binary_serializer import binary_serializer  # grc-generated hier_block
from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import filter
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation

import tempest

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal as sci_signal
from skimage.io import imread

import time


# Get file path
FILEPATH = sys.argv[-1]

# Read image and get original resolution and blanking resolution
I = imread(FILEPATH)
v_in,h_in = I.shape[:2]
v = (v_in==1080)*1125 + (v_in==900)*1000  + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
h = (h_in==1920)*2200 + (h_in==1600)*1800 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800

# Keep file name, without the file extention
FILENAME = FILEPATH.split('/')[-1].split('.')[0]

N_harmonic = input('Choose pixel rate harmonic number: ')
N_harmonic = int(N_harmonic)
SNR = input('Choose SNR in dB (zero or negative for no noise): ')
SNR = int(SNR)

def consumer(serial_data,h, v, samp_rate, usrp_rate, SNR=SNR):

    # Resolucion y fps, con blanking de la imagen
    h_total, v_total = h, v
    N_samples = serial_data.shape[0]
    frame_rate = 60

    if SNR > 0:
        signal_power = np.sum(np.abs(np.fft.fft(serial_data)**2/N_samples))
        noise_sigma = np.sqrt(signal_power/(10**(SNR/10)))
        serial_data = serial_data + np.random.normal(0, noise_sigma, N_samples)

    # Muestreo del SDR
    image_seq = sci_signal.resample_poly(serial_data,up=usrp_rate, down=samp_rate)

    # Muestreo a nivel de píxel 
    image_Rx = sci_signal.resample(image_seq, h_total*v_total).reshape(v_total,h_total)

    image_Rx = np.abs(image_Rx)

    image_Rx = 255*(image_Rx-image_Rx.min())/(image_Rx.max()-image_Rx.min())

    im = Image.fromarray(image_Rx.astype('uint8'))
    im.save(FILENAME+'_tempest_harm'+str(N_harmonic)+'_'+str(SNR)+'_dB.png')


class NO_GUI_tempest_simulated_TMDS(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Manual  Simulated Tempest TMDS Example")

        ##################################################
        # Variables
        ##################################################
        self.refresh_rate = refresh_rate = 60
        self.Vsize = Vsize = v
        self.Hsize = Hsize = h
        self.px_rate = px_rate = Hsize*Vsize*refresh_rate/1000
        self.inter = inter = 1
        self.usrp_rate = usrp_rate = int(50e6/1000)
        self.samp_rate = samp_rate = int(10*px_rate*inter)
        self.rectangular_pulse = rectangular_pulse = [1]*inter
        self.harmonic = harmonic = N_harmonic
        self.Vvisible = Vvisible = v_in
        self.Hvisible = Hvisible = h_in

        ##################################################
        # Blocks
        ##################################################

        self.data_sink_0 = blocks.vector_sink_c(1)
        self.tempest_TMDS_image_source_0 = tempest.TMDS_image_source(FILEPATH, 3)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=10*inter,
                taps=None,
                fractional_bw=0.4)
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_fcc(inter, rectangular_pulse)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
    
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(2)
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(3)
        self.binary_serializer_0_0_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.binary_serializer_0_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.binary_serializer_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, px_rate*harmonic, 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.binary_serializer_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.binary_serializer_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.binary_serializer_0_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0))

        self.connect((self.blocks_multiply_xx_0, 0), (self.data_sink_0, 0))

        self.connect((self.interp_fir_filter_xxx_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 0), (self.binary_serializer_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 1), (self.binary_serializer_0_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 2), (self.binary_serializer_0_0_0, 0))


    def get_refresh_rate(self):
        return self.refresh_rate

    def set_refresh_rate(self, refresh_rate):
        self.refresh_rate = refresh_rate
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/1000)

    def get_Vsize(self):
        return self.Vsize

    def set_Vsize(self, Vsize):
        self.Vsize = Vsize
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/1000)

    def get_Hsize(self):
        return self.Hsize

    def set_Hsize(self, Hsize):
        self.Hsize = Hsize
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/1000)

    def get_px_rate(self):
        return self.px_rate

    def set_px_rate(self, px_rate):
        self.px_rate = px_rate
        self.set_samp_rate(int(10*self.px_rate*self.inter))
        self.analog_sig_source_x_0.set_frequency(self.px_rate*self.harmonic)

    def get_inter(self):
        return self.inter

    def set_inter(self, inter):
        self.inter = inter
        self.set_rectangular_pulse([1]*self.inter)
        self.set_samp_rate(int(10*self.px_rate*self.inter))

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate = usrp_rate

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_rectangular_pulse(self):
        return self.rectangular_pulse

    def set_rectangular_pulse(self, rectangular_pulse):
        self.rectangular_pulse = rectangular_pulse
        self.interp_fir_filter_xxx_0.set_taps(self.rectangular_pulse)

    def get_harmonic(self):
        return self.harmonic

    def set_harmonic(self, harmonic):
        self.harmonic = harmonic
        self.analog_sig_source_x_0.set_frequency(self.px_rate*self.harmonic)

    def get_Vvisible(self):
        return self.Vvisible

    def set_Vvisible(self, Vvisible):
        self.Vvisible = Vvisible

    def get_Hvisible(self):
        return self.Hvisible

    def set_Hvisible(self, Hvisible):
        self.Hvisible = Hvisible



def main(top_block_cls=NO_GUI_tempest_simulated_TMDS, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Run flowgraph until end
    tb.start()
    tb.wait()
    tb.stop()

    samp_rate = tb.get_samp_rate()
    usrp_rate = tb.get_usrp_rate()

    # Get output data from fg's last block
    serial_data = np.array(tb.data_sink_0.data())

    # Resample and reshape to image size
    consumer(serial_data,h=h, v=v, samp_rate=samp_rate, usrp_rate=usrp_rate)


if __name__ == '__main__':
    main()

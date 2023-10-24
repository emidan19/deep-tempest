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
from gnuradio import filter
from gnuradio import gr
import signal

import tempest

import numpy as np
from PIL import Image
from scipy import signal as sci_signal
import time

# Currently supporting png, jpg, jpeg, tif and gif extentions only
def get_images_names_from_folder (folder):
    images_list = [image for image in os.listdir(folder) \
                   if image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg') or \
                       image.endswith('.tif') or image.endswith('.tiff') or image.endswith('.gif')] 
    return images_list

def get_subfolders_names_from_folder(folder):
    subfolders_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return subfolders_list

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

def signal_capture_downsampling(serial_data,h, v, samp_rate, usrp_rate, noise_std):

    # Resolucion y fps, con blanking de la imagen
    h_total, v_total = h, v
    N_samples = serial_data.shape[0]

    # Random uniform phase noise
    # phase_noise = np.exp(1j*np.random.uniform(0,2*np.pi, len(serial_data)))

    if noise_std > 0:
        noise_sigma = noise_std/15.968719423 # sqrt(255)~15.968719423 because of stretching with 255 at saving
        serial_data = serial_data + np.random.normal(0, noise_sigma,N_samples) + 1j*np.random.normal(0, noise_sigma,N_samples)

    # Muestreo del SDR
    image_seq = sci_signal.resample_poly(serial_data,up=usrp_rate, down=samp_rate)

    # Muestreo a nivel de píxel 
    image_Rx = sci_signal.resample(image_seq, h_total*v_total).reshape(v_total,h_total)

    return image_Rx


class NO_GUI_tempest_simulated_TMDS(gr.top_block):


    def reset_all_blocks_but_image_source(self):

        self.data_sink_0 = blocks.vector_sink_c(1)
        
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_fcc(self.inter, self.rectangular_pulse)
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
        self.analog_sig_source_x_0 = analog.sig_source_c(self.samp_rate, analog.GR_COS_WAVE, self.px_rate*self.harmonic, 1, 0, 0)

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



    def __init__(self, Vsize=None, Hsize=None, Vvisible=None, Hvisible=None, FILEPATH=None, blanking=False):
        gr.top_block.__init__(self, "Manual  Simulated Tempest TMDS Example")

        ##################################################
        # Variables
        ##################################################
        self.Vsize = Vsize # To set at main
        self.Hsize = Hsize # To set at main
        self.Vvisible = Vvisible # To set at main
        self.Hvisible = Hvisible # To set at main
        self.FILEPATH = FILEPATH # To set at main
        self.blanking = blanking 

        # Init as 1
        self.harmonic = harmonic = 1 # To set at main

        self.refresh_rate = refresh_rate = 60
        self.px_rate = px_rate = Hsize*Vsize*refresh_rate/1000
        self.inter = inter = 1
        self.usrp_rate = usrp_rate = int(50e6/1000)
        self.samp_rate = samp_rate = int(10*px_rate*inter)
        self.rectangular_pulse = rectangular_pulse = [1]*inter


        ##################################################
        # Blocks
        ##################################################

        self.data_sink_0 = blocks.vector_sink_c(1)
        self.tempest_TMDS_image_source_0 = tempest.TMDS_image_source(FILEPATH, 3, blanking)
        # self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
        #         interpolation=inter,
        #         decimation=10*inter,
        #         taps=None,
        #         fractional_bw=0.4)
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


def set_top_block(top_block_cls=NO_GUI_tempest_simulated_TMDS, options_dict=dict()):

    # Init the top block with parameters
    tb = top_block_cls(Vsize=options_dict['Vsize'], Hsize=options_dict['Hsize'], Vvisible=options_dict['Vvisible'], Hvisible=options_dict['Hvisible'],
                         FILEPATH=options_dict['FILEPATH'],blanking=options_dict['blanking'])

    return tb

def run_simulation_flowgraph(top_block, harmonic, noise_std):

    top_block.set_harmonic(harmonic)

    def sig_handler(sig=None, frame=None):
        top_block.stop()
        top_block.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Run flowgraph until end
    top_block.start()
    top_block.wait()
    top_block.stop()

    # Get top block's parameters
    samp_rate = top_block.get_samp_rate()
    usrp_rate = top_block.get_usrp_rate()
    h = top_block.get_Hsize()
    v = top_block.get_Vsize()

    # Get output data from fg's last block
    serial_data = np.array(top_block.data_sink_0.data())

    # Resample and reshape to image size
    I_capture = signal_capture_downsampling(serial_data,h=h, v=v, samp_rate=samp_rate, usrp_rate=usrp_rate, noise_std=noise_std)

    # Start over flowgraph and the coded image transmition
    top_block.reset_all_blocks_but_image_source()
    top_block.tempest_TMDS_image_source_0.line_num = 0

    return I_capture

def image_simulation(image_path, N_harmonic, noise_std, blanking=False):

    # Set initial options
    options_dict = {'Vsize': I.shape[0],
                    'Hsize': I.shape[1],
                    'Vvisible': I.shape[0],
                    'Hvisible': I.shape[1],
                    'FILEPATH': image_path,
                    'blanking': blanking,
                    }

    # Initialize the top block with encoded image
    top_block = set_top_block(top_block_cls=NO_GUI_tempest_simulated_TMDS, options_dict=options_dict)

    # Run simulation flowgraph
    I_capture = run_simulation_flowgraph(top_block, N_harmonic, noise_std)

    # From complex image to 2 channel image
    v_total, h_total = I_capture.shape
    I_capture_2channels = np.zeros((v_total,h_total,2))

    I_real = np.real(I)
    I_imag = np.imag(I)
    
    realmax, realmin = I_real.max(), I_real.min()
    imagmax, imagmin = I_imag.max(), I_imag.min()

    # Stretch contrast on every channel
    I_capture_2channels[:,:,0] = 255*(I_real-realmin)/(realmax-realmin)
    I_capture_2channels[:,:,1] = 255*(I_imag-imagmin)/(imagmax-imagmin)

    I_capture_2channels = I_capture_2channels.astype('uint8')

    return I_capture_2channels


def main():

    # Get foldername argument
    foldername = sys.argv[-1]
    
    # Get images and subfolders names
    images_tmp = get_images_names_from_folder(foldername)
    old_subfolders = get_subfolders_names_from_folder(foldername)
    
    # Keep images without dedicated folders only
    images = []
    new_subfolders = []
    for image in images_tmp:
        image_name = image.split('.')[0]
        if image_name not in old_subfolders:
            images.append(image)
            new_subfolders.append(image_name)
    
    # Possible std dev noise simulation values
    noise_stds = np.array([ 0, 5,  10,  15,  20,  25,  40, 50])
    
    for image, subfolder in zip(images,new_subfolders):
        
        # Create new directory for simulations
        subfolder_path = foldername+'/'+subfolder
        os.mkdir(subfolder_path)

        # timestamp for simulation starting
        t1_image = time.time()
        
        # Read image
        image_path = foldername+'/'+image
        imagename = image.split('.')[0]
        I = np.array(Image.open(image_path))

        # Set initial options
        options_dict = {'Vsize': I.shape[0],
                        'Hsize': I.shape[1],
                        'Vvisible': I.shape[0],
                        'Hvisible': I.shape[1],
                        'FILEPATH': image_path,
                        'blanking': False,
                        }

        # Initialize the top block with encoded image
        top_block = set_top_block(top_block_cls=NO_GUI_tempest_simulated_TMDS, options_dict=options_dict)

        for i in range(4):

            # Choose random pixelrate harmonic number
            N_harmonic = np.random.randint(1,10)
            
            # Choose random SNR value (SNR=0 for no noise)
            noise_std = np.random.choice(noise_stds)

            path = subfolder_path+'/'+imagename+'_'+str(N_harmonic)+'harm_'+str(noise_std)+"std.png"

            I_capture = run_simulation_flowgraph(top_block, N_harmonic, noise_std)

            save_simulation_image(I_capture,path)

            if i==0:
                # timestamp for simulation ending
                t2_image = time.time()    


        t_image = t2_image-t1_image
        
        print('Tiempo de la primer simulación de '+image+':','{:.2f}'.format(t_image)+'s\n')

if __name__ == '__main__':
    main()

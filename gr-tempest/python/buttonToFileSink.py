#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Gabriel Varela, Emilio Martínez.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

### Classic imports
import numpy as np
from gnuradio import gr
import pmt
from scipy import signal
from datetime import datetime
from PIL import Image

### Deep-TEMPEST imports
import os
import math
import argparse
import time
import random
import torch
import sys
# Adding KAIR folder to the system path
# current_file_path = os.path.abspath(__file__)
# sys.path.insert(0, os.path.join(current_file_path,'../../KAIR/'))
kair_path = '/home/emidan19/deep-tempest/KAIR'
sys.path.insert(0,kair_path)
from utils import utils_option as option
from utils import utils_image as util
from utils.utils_dist import get_dist_info, init_dist
from models.select_model import define_Model
from models.network_unet import UNetRes as net

def load_enhancement_model(json_path='/home/emidan19/deep-tempest/KAIR/options/test_drunet.json'):
    '''
    # ----------------------------------------
    # Step - 1 Prepare options
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    
    """
    # ----------------------------------------
    # Step 2 - distributed settings
    # ---------------------------------------- 
    """
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    opt = option.dict_to_nonedict(opt)

    """
    # ----------------------------------------
    # Step 3 - Load model with option setup
    # ---------------------------------------- 
    """

    model_path = os.path.join(kair_path,opt['path']['pretrained_netG'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt_netG = opt['netG']

    in_nc = opt_netG['in_nc']
    out_nc = opt_netG['out_nc']
    nc = opt_netG['nc']
    nb = opt_netG['nb']
    act_mode = opt_netG['act_mode']
    bias = opt_netG['bias']

    model = net(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, bias=bias)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    return model


class buttonToFileSink(gr.sync_block):
    f"""
    Block that saves num_samples of complex samples after recieving a TRUE boolean message in the 'en' port
    """
    def __init__(self, Filename = "output.png", input_width=740, H_size=2200, V_size=1125, enhance_image=False):
        gr.sync_block.__init__(self,
            name="buttonToFileSink",
            in_sig=[(np.complex64)],
            out_sig=[],
        )
        self.Filename = Filename
        self.input_width = input_width
        self.H_size = H_size
        self.V_size = V_size
        self.num_samples = int(input_width*V_size)
        self.en = False #default
        self.remaining2Save = 0
        self.savingSamples = 0
        self.message_port_register_in(pmt.intern("en")) #declare message port
        self.set_msg_handler(pmt.intern("en"), self.handle_msg) #declare handler for messages
        self.stream_image = [] # initialize list to apppend samples
        if enhance_image:
            # Load model
            self.model = load_enhancement_model()
            # Define inference function
            def inference (img):
                # Preprocess image to network input
                img_L = img[:,:,:2]
                img_L = util.uint2single(img_L)
                img_L = util.single2tensor4(img_L)

                # Model inference on image
                img_E = self.model(img_L)
                img_E = util.tensor2uint(img_E)
                return img_E
            
            self.inference = inference
        else:
            # No enhancement. 
            def inference(img):
                return img
            self.inference = inference


    def work(self, input_items, output_items):      # Don't process, just save samples

        self.available_samples = len(input_items[0]) #available samples at the input for read at this moment

        if self.en == True:

            self.stream_image.extend(input_items[0])

            self.stream_image = self.stream_image[-self.num_samples:]

            if len(self.stream_image)==self.num_samples:#or self.remaining2Save > 0:
                
                # if self.remaining2Save > 0 and self.remaining2Save < self.available_samples: #remaining to save from last time, don't care if true was pressed or not
                #     self.savingSamples = self.remaining2Save
                #     self.remaining2Save = 0 #no more remaining after this
                # elif self.remaining2Save > 0: #the remaining still are more than available
                #     self.savingSamples = self.available_samples #save whatever available
                #     self.remaining2Save = self.remaining2Save - self.savingSamples # refresh whatever left to save next time
                # else: #not remaining from last time but pressed true
                #     if self.num_samples > self.available_samples: #too large for the available samples
                #         self.savingSamples = self.available_samples # save whatever available and wait for the remaining
                #         self.remaining2Save = self.num_samples - self.savingSamples #remaining to save
                #     else:
                #         self.savingSamples = self.num_samples # enough samples available when pressing button
                
                # Save the number of samples calculated before
                self.save_samples() 
                # Back to default
                self.en = False 
                # Empty stream for new upcoming screenshots
                self.stream_image = [] 

        return  self.available_samples #consume all the samples at the input saved or not 
        # return len(output_items)
    
    def save_samples(self):

        # Interpolate signal to original image size
        interpolated_signal = signal.resample(self.stream_image, self.H_size*self.V_size)
        
        # Reshape signal to image
        captured_image_complex = np.array(interpolated_signal).reshape((self.V_size,self.H_size))

        # Create png image 
        captured_image = np.zeros((self.V_size,self.H_size,3))
        captured_image[:,:,0] = np.real(captured_image_complex)
        captured_image[:,:,1] = np.imag(captured_image_complex)
        # Stretching contrast and mantaining complex phase unchanged
        min_value, max_value = np.min(captured_image[:,:,:2]), np.max(captured_image[:,:,:2])
        captured_image[:,:,0] = 255*(captured_image[:,:,0] - min_value) / (max_value - min_value)
        captured_image[:,:,1] = 255*(captured_image[:,:,1] - min_value) / (max_value - min_value)

        # Image to uint8
        captured_image = captured_image.astype('uint8')

        # Date and time of screenshot
        date_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") 

        # Create inference if using deep-tempest enhacenment
        captured_image = self.inference(captured_image)

        # Save image as png
        im = Image.fromarray(captured_image)
        im.save(self.Filename+'-gr-tempest_screenshot_'+date_time+'.png')
        # Show image at runtime
        im.show()


    # Handler of msg
    def handle_msg(self, msg):     
        Msg_value = pmt.cdr(msg) 
        self.en = pmt.to_bool(Msg_value)   #the message input of the button block is (msgName,msgValue) the first part is not useful for this

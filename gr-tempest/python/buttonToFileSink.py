#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Gabriel Varela, Emilio Martínez.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
import pmt
from scipy import signal
from datetime import datetime
from PIL import Image

class buttonToFileSink(gr.sync_block):
    f"""
    Block that saves num_samples of complex samples after recieving a TRUE boolean message in the 'en' port
    """
    def __init__(self, Filename = "output.png", input_width=740, H_size=2200, V_size=1125):
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

        # Save image as png
        im = Image.fromarray(captured_image)
        im.save(self.Filename+'-gr-tempest_screenshot_'+date_time+'.png')
        # Show image at runtime
        im.show()


    # Handler of msg
    def handle_msg(self, msg):     
        Msg_value = pmt.cdr(msg) 
        self.en = pmt.to_bool(Msg_value)   #the message input of the button block is (msgName,msgValue) the first part is not useful for this

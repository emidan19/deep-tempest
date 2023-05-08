#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Gabriel Varela.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
import pmt
import struct
import cmath

class buttonToFileSink(gr.sync_block):
    f"""
    Block that saves num_samples of complex samples after recieving a TRUE boolean message in the 'en' port
    """
    def __init__(self, Filename = "output.dat", num_samples = 10):
        gr.sync_block.__init__(self,
            name="buttonToFileSink",
            in_sig=[(np.complex64)],
            out_sig=[],
        )
        self.Filename = Filename
        self.num_samples = num_samples
        self.en = False #default
        self.remaining2Save = 0
        self.savingSamples = 0
        self.message_port_register_in(pmt.intern("en")) #declare message port
        self.set_msg_handler(pmt.intern("en"), self.handle_msg) #declare handler for messages
    def work(self, input_items, output_items):      # Don't process, just save samples
        if self.en == True or self.remaining2Save > 0:
            self.available_samples = len(input_items[0]) #available samples at the input for read at this moment
            if self.remaining2Save > 0 and self.remaining2Save < self.available_samples: #remaining to save from last time, don't care if true was pressed or not
                self.savingSamples = self.remaining2Save
                self.remaining2Save = 0 #no more remaining after this
            elif self.remaining2Save > 0: #the remaining still are more than available
                self.savingSamples = self.available_samples #save whatever available
                self.remaining2Save = self.remaining2Save - self.savingSamples # refresh whatever left to save next time
            else: #not remaining from last time but pressed true
                if self.num_samples > self.available_samples: #too large for the available samples
                    self.savingSamples = self.available_samples # save whatever available and wait for the remaining
                    self.remaining2Save = self.num_samples - self.savingSamples #remaining to save
                else:
                    self.savingSamples = self.num_samples # enough samples available when pressing button
            self.save_samples(input_items[0][-self.savingSamples:]) #save the number of samples calculated before
            self.en = False #back to default
        return  self.savingSamples #consume all the samples at the input saved or not 
    
    def save_samples(self, samples):
        with open(self.Filename, 'ab') as f: #append binary
            for i in range(0,self.savingSamples):
               f.write(struct.pack('ff', samples[i].real,samples[i].imag)) #write in float real and imaginary part

    def handle_msg(self, msg):     #handler of msg
        Msg_value = pmt.cdr(msg) 
        self.en = pmt.to_bool(Msg_value)   #the message input of the button block is (msgName,msgValue) the first part is not useful for this

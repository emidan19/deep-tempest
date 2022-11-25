#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020
#   Federico "Larroca" La Rocca <flarroca@fing.edu.uy>
#
#   Instituto de Ingenieria Electrica, Facultad de Ingenieria,
#   Universidad de la Republica, Uruguay.
#  
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#  
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#  
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#
#


import numpy as np
from gnuradio import gr

def binarray_to_uint(binarray):

    num = binarray[0]
    for n in range(1,len(binarray)):
        num = (num << 1) + binarray[n]
    return num

def DecTMDS_pixel (D):
    """10-bit pixel TMDS decoding

    Inputs: 
    - D: binary list 

    Output:
    - pix_out: 8-bit TMDS decoded pixel

    """ 

    if D[9]:
        D[:8] = [not(val) for val in D[:8]]

    Q = D.copy()[:8]

    if D[8]:
        for k in range(1,8):
            Q[k] = D[k] ^ D[k-1]
    else:
        for k in range(1,8):
            Q[k] = not(D[k] ^ D[k-1])

    # Return pixel as uint
    return binarray_to_uint(Q)

class TMDS_decoder(gr.basic_block):
    """
    Outputs a pixel value grouping 10 bits and using TMDS decoding.
    Input values must be either 0 or 1.
    """
    def __init__(self):
        gr.basic_block.__init__(self,
            name="TMDS_decoder",
            in_sig=[np.float32],
            out_sig=[np.float32])
        self.binarray = []

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
            ninput_items_required[i] = noutput_items

    def general_work(self, input_items, output_items):
        out_now = 0
        last_input = input_items[0][-1]
        self.binarray.append(int(last_input))
        if len(self.binarray) == 10:
            out = output_items[0]
            out[:] = DecTMDS_pixel(self.binarray)
            self.binarray = []
            out_now = 1
        self.consume_each(1)  # consume(0, len(input_items[0]))
        return out_now

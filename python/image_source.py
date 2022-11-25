#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2015,2016 Chris Kuethe <chris.kuethe@gmail.com>

# Copyright 2020 (minor changes to generate blanking)
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

import numpy
from gnuradio import gr
from PIL import Image
from PIL import ImageOps

class image_source(gr.sync_block):
    """
    Given an image file readable by Python-Imaging, this block
    produces a superposition of the three colors, plus the blanking around it
    (and scales it to match the desired resolution). 
    """

    image_file = None
    repeatmode = 1
    image_data = None

    def __init__(self, image_file, Hvisible, Vvisible, Htotal, Vtotal, repeatmode):
        gr.sync_block.__init__(self,
                name="image_source",
                in_sig=None,
                out_sig=[numpy.float32])
    
        self.image_file = image_file
        self.repeatmode = repeatmode
        self.Vvisible = Vvisible
        self.Hvisible = Hvisible
        self.Htotal = Htotal
        self.Vtotal = Vtotal
        self.Hblanking = (Htotal - Hvisible)//2
        self.Vblanking = (Vtotal - Vvisible)//2
        self.load_image()

    def load_image(self):
        """decode the image into a buffer"""
        self.image_data = Image.open(self.image_file)
        self.image_data = ImageOps.grayscale(self.image_data)
        
        
        newsize = (self.Hvisible, self.Vvisible)
        self.image_data = self.image_data.resize(newsize)
     
        # I create the blanking
        background = Image.new('F', (self.Htotal, self.Vtotal), (0))
        # I paste the image with the blankins
        offset = (self.Hblanking, self.Vblanking)
        background.paste(self.image_data, offset)
        self.image_data = background
        
        (self.image_width, self.image_height) = self.image_data.size
        
        self.set_output_multiple(self.image_width)
        
        self.image_data = list(self.image_data.getdata())
        self.image_len = len(self.image_data)
        self.line_num = 0


    def work(self, input_items, output_items):
        out = output_items[0]
        out[:self.image_width] = self.image_data[self.image_width*self.line_num: self.image_width*(1+self.line_num)]

        self.line_num += 1
        if self.line_num >= self.image_height:
            self.line_num = 0
        
        if self.repeatmode == 2:
            self.load_image()
        return self.image_width


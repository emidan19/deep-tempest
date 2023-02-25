#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023
#   Emilio Martinez <emilio.martinez@fing.edu.uy>
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
from PIL import Image
from tempest.DTutils import TMDS_pix_table, TMDS_cntdiff_table, pixel_fastencoding, TMDS_encoding
from gnuradio import gr


class TMDS_image_source(gr.sync_block):
    """
    TMDS encoding for the input image. Produces blanking. 
    Image input size must be one of the following:
    * 640x480
    * 800x600
    * 1280x720
    * 1920x1080
    """

    def __init__(self, image_file, mode, blanking):
        gr.sync_block.__init__(self,
                name="TMDS_image_source",
                in_sig=None,
                out_sig=[np.float32, np.float32, np.float32])
    
        self.image_file = image_file
        self.mode = mode
        self.blanking = blanking
        self.load_image()

    def load_image(self):

        """Decode the image into a buffer and encode it (or not) TMDS"""
        self.image_data = np.array(Image.open(self.image_file))

        # Check if mode uses TMDS encoding
        if (self.mode==1 or self.mode==3):
            #Encode the image with TMDS
            self.image_data = TMDS_encoding(self.image_data,blanking = self.blanking)
            print('TMDS encoding ready!!!')

        else:

            # Do not encode TMDS
            # Create "ghost dimension" if I is gray-scale image (not RGB)
            if len(self.image_data.shape)!= 3:
              # Gray-scale image
              self.image_data = np.repeat(self.image_data[:, :, np.newaxis], 3, axis=2).astype('uint8')
              chs = 1
            else:
              # RGB image
              chs = 3

            if self.blanking:
              # Use blanking
              v_in, h_in = self.image_data.shape[:2]
              v = (v_in==1080)*1125 + (v_in==900)*1000  + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
              h = (h_in==1920)*2200 + (h_in==1600)*1800 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 
              image_blank = 255*np.ones((v,h,chs))


              hdiff = (h-h_in)//2
              vdiff = (v-v_in)//2
              image_blank[vdiff:vdiff+v_in,hdiff:hdiff+h_in] = self.image_data[:,:,:3]

              self.image_data = image_blank[:,:,:3]

        
        (self.image_height, self.image_width) = self.image_data.shape[:2]
        
        self.set_output_multiple(self.image_width)
        

        self.image_data_red   = list(self.image_data[:,:,0].flatten())
        self.image_data_green = list(self.image_data[:,:,1].flatten())
        self.image_data_blue  = list(self.image_data[:,:,2].flatten())
        self.image_len = len(self.image_data)
        self.line_num = 0

    def get_image_shape(self):
      return (self.image_height,self.image_width)


    def work(self, input_items, output_items):

        if (self.line_num >= self.image_height) and (self.mode==3 or self.mode==4):
            print('[TMDS image source] Last image line transmitted')
            return(-1)

        out_red = output_items[0]
        out_red[:self.image_width] = self.image_data_red[self.image_width*self.line_num: self.image_width*(1+self.line_num)]

        out_green = output_items[1]
        out_green[:self.image_width] = self.image_data_green[self.image_width*self.line_num: self.image_width*(1+self.line_num)]

        out_blue = output_items[2]
        out_blue[:self.image_width] = self.image_data_blue[self.image_width*self.line_num: self.image_width*(1+self.line_num)]

        self.line_num += 1
        if (self.mode==1 or self.mode==2) and (self.line_num >= self.image_height):
            self.line_num = 0

        return self.image_width

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


# import numpy
# from gnuradio import gr

import numpy as np
from skimage.io import imread
# from DTutils import TMDS_pix_table, TMDS_cntdiff_table, pixel_fastencoding, TMDS_encoding
from gnuradio import gr

def uint8_to_binarray(integer):
  """Convert integer into fixed-length 8-bit binary array. LSB in [0].
  Extended and modified code from https://github.com/projf/display_controller/blob/master/model/tmds.py
  """

  b_array = [int(i) for i in reversed(bin(integer)[2:])]
  b_array += [0]*(8-len(b_array))
  return b_array

def uint16_to_binarray(integer):
  """Convert integer into fixed-length 16-bit binary array. LSB in [0].
  Extended and modified code from https://github.com/projf/display_controller/blob/master/model/tmds.py
  """
  b_array = [int(i) for i in reversed(bin(integer)[2:])]
  b_array += [0]*(16-len(b_array))
  return b_array

def binarray_to_uint(binarray):
    
  array = binarray[::-1]
  num = array[0]
  for n in range(1,len(binarray)):
    num = (num << 1) + array[n]

  return num

def TMDS_pixel (pix,cnt=0):
  """8bit pixel TMDS coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cnt: 0's and 1's balance updated with new pixel coding

  """ 
  # Convert 8-bit pixel to binary list D
  D = uint8_to_binarray(pix)

  # Initialize output q
  qm = [D[0]]

  # 1's unbalanced condition at current pixel
  N1_D = D.count(1)

  if N1_D>4 or (N1_D==4 and not(D[0])):

    # XNOR of consecutive bits
    for k in range(1,8):
      qm.append( not(qm[k-1] ^ D[k]) )
    qm.append(0)

  else:
    # XOR of consecutive bits
    for k in range(1,8):
      qm.append( qm[k-1] ^ D[k] )
    qm.append(1)

  # Initialize output qout
  qout = qm.copy()

  # Unbalanced condition with previous and current pixels
  N1_qm = qm[:8].count(1)
  N0_qm = 8 - N1_qm

  if cnt==0 or N1_qm==4:

    qout.append(not(qm[8]))
    qout[:8]=qm[:8] if qm[8] else np.logical_not(qm[:8])

    if not(qm[8]):
      cnt += N0_qm - N1_qm 
    else:
      cnt += N1_qm - N0_qm 

  else:

    if (cnt>0 and N1_qm>4) or (cnt<0 and N1_qm<4):
      qout.append(1)
      qout[:8] = np.logical_not(qm[:8])
      cnt += 2*qm[8] +N0_qm -N1_qm
    else:
      qout.append(0)
      cnt += -2*(not(qm[8])) + N1_qm - N0_qm

  # Return the TMDS coded pixel as uint and 0's y 1's balance
  return binarray_to_uint(qout), cnt


def TMDS_pixel_cntdiff (pix,cnt=0):
  """8bit pixel TMDS coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cntdiff: balance difference given by the actual coded pixel

  """ 
  # Convert 8-bit pixel to binary list D
  D = uint8_to_binarray(pix)

  # Initialize output q
  qm = [D[0]]

  # 1's unbalanced condition at current pixelo
  N1_D = D.count(1)

  if N1_D>4 or (N1_D==4 and not(D[0])):

    # XNOR of consecutive bits
    for k in range(1,8):
      qm.append( not(qm[k-1] ^ D[k]) )
    qm.append(0)

  else:
    # XOR of consecutive bits
    for k in range(1,8):
      qm.append( qm[k-1] ^ D[k] )
    qm.append(1)

  # Initialize output qout
  qout = qm.copy()

  # Unbalanced condition with previous and current pixels
  N1_qm = qm[:8].count(1)
  N0_qm = 8 - N1_qm

  if cnt==0 or N1_qm==4:

    qout.append(not(qm[8]))
    qout[:8]=qm[:8] if qm[8] else [not(val) for val in qm[:8]]

    if not(qm[8]):
      cnt_diff = N0_qm - N1_qm 
    else:
      cnt_diff = N1_qm - N0_qm 

  else:

    if (cnt>0 and N1_qm>4) or (cnt<0 and N1_qm<4):
      qout.append(1)
      qout[:8] = [not(val) for val in qm[:8]]
      cnt_diff = 2*qm[8] +N0_qm -N1_qm
    else:
      qout.append(0)
      cnt_diff = -2*(not(qm[8])) + N1_qm - N0_qm

  # Return the TMDS coded pixel as uint and 0's y 1's balance difference
  uint_out = binarray_to_uint(qout)
  return uint_out, cnt_diff


### Create TMDS LookUp Tables for fast encoding (3 times faster than the other implementation)
byte_range = np.arange(256)
# Initialize pixel coding and cnt-difference arrays
TMDS_pix_table = np.zeros((256,3),dtype='uint16')
TMDS_cntdiff_table = np.zeros((256,3),dtype='int8')

for byte in byte_range:
  p0,p_null, p1 = TMDS_pixel_cntdiff(byte,-1),TMDS_pixel_cntdiff(byte,0),TMDS_pixel_cntdiff(byte,1) # 0's and 1's unbalance respect.
  TMDS_pix_table[byte,0] = p0[0]
  TMDS_pix_table[byte,1] = p_null[0]
  TMDS_pix_table[byte,2] = p1[0]
  TMDS_cntdiff_table[byte,0] = p0[1]
  TMDS_cntdiff_table[byte,1] = p_null[1]
  TMDS_cntdiff_table[byte,2] = p1[1]

def pixel_fastencoding(pix,cnt=0):
  """8bit pixel TMDS fast coding

  Inputs: 
  - pix: 8-bit pixel
  - cnt: 0's and 1's balance. Default in 0 (balanced)

  Outputs:
  - pix_out: TDMS coded 16-bit pixel (only 10 useful)
  - cnt: 0's and 1's balance updated with new pixel coding

  """ 
  balance_idx = int(np.sign(cnt))+1
  pix_out = TMDS_pix_table[pix,balance_idx]
  cnt    += TMDS_cntdiff_table[pix,balance_idx]

  return  pix_out, cnt

def TMDS_encoding (I, blanking = False):
  """TMDS image coding

  Inputs: 
  - I: 3-D image array (v_size, h_size, channels)
  - blanking: Boolean that specifies if horizontal and vertical blanking is applied or not

  Output:
  - I_c: TDMS coded 16-bit image (only 10 useful)

  """ 

  # Create "ghost dimension" if I is gray-scale image (not RGB)
  if len(I.shape)!= 3:
    I = np.repeat(I[:, :, np.newaxis], 3, axis=2).astype('uint8')
    
  chs = 3

  # Get image resolution
  v_in, h_in = I.shape[:2]
  
  if blanking:
    # Get blanking resolution for input image
    
    v = (v_in==1080)*1125 + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
    h = (h_in==1920)*2200 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 

    vdiff = v - v_in
    hdiff = h - h_in

    # Create image with blanking and change type to uint16
    # Assuming the blanking corresponds to 10bit number [0, 0, 1, 0, 1, 0, 1, 0, 1, 1] (LSB first)
    I_c = 852*np.ones((v,h,chs)).astype('uint16')
    
  else:
    vdiff = 0
    hdiff = 0
    I_c = 255*np.ones((v_in,h_in,chs)).astype('uint16')

  # Iterate over channels and pixels
  for c in range(chs):
    for i in range(v_in):
        cnt = [0,0,0]
        for j in range(h_in):
            # Get pixel and code it TMDS between blanking
            pix = I[i,j,c]
            I_c[i + vdiff//2 , j + hdiff//2, c], cnt[c] = pixel_fastencoding (pix,cnt[c])

  return I_c

class TMDS_image_source(gr.sync_block):
    """
    TMDS encoding for the input image. Produces blanking
    """

    def __init__(self, image_file, mode):
        gr.sync_block.__init__(self,
                name="TMDS_image_source",
                in_sig=None,
                out_sig=[np.float32, np.float32, np.float32])
    
        self.image_file = image_file
        self.mode = mode
        self.load_image()

    def load_image(self):

        """decode the image into a buffer"""
        self.image_data = imread(self.image_file)

        if (self.mode==1 or self.mode==3):
            self.image_data = TMDS_encoding(self.image_data,blanking = True)
            print('TMDS encoding ready!!!')
        else:
            v_in, h_in = self.image_data.shape[:2]
            v = (v_in==1080)*1125 + (v_in==720)*750   + (v_in==600)*628  + (v_in==480)*525
            h = (h_in==1920)*2200 + (h_in==1280)*1650 + (h_in==800)*1056 + (h_in==640)*800 
            # image_blank = 255*np.ones((1125,2200,3))
            image_blank = np.zeros((v,h,3))
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


    def work(self, input_items, output_items):

        if (self.line_num >= self.image_height) and (self.mode==3 or self.mode==4):
            print('Work done. Now you should stop the flowgraph ;)')
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

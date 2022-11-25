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


def bin_serializer(num, N):
  '''
  Serialize non-negative number to binary list.

  Inputs: 
  - num: number to serialize. The value must be between 0 and 2^N - 1
  - N: amount of bits to represent the binary number

  Output:
  - binlist: binary list which represents the number in bits (LSB first)

  '''

  # Cast number as integer
  num = int(num)

  assert num<2**N and num>=0, "Number value must be between 0 and 2^N - 1"

  # Initialize lists
  num_bits = []
  # Cast number to binary
  binstring = bin(num)[2:]
  # Fill string with 0's to get length N
  binstring = '0'*(N-len(binstring))+binstring
  # Re-order string for LSB first
  binstring = binstring[::-1]
  # Cast the list to numpy uint8
  binlist  = list( map( int, binstring ) )

  return binlist

class binary_serializer(gr.interp_block):
    """
    Cast the input float to integer and outputs the binary representation of the number serialized (LSB first). 

    Parameter N is the number of bits to be used for the binary representation. 

    Make sure the input number 'num'
    satisfies num in {0, 2^N - 1}
    """

    def __init__(self, N):
        gr.interp_block.__init__(self,
                name="binary_serializer",
                in_sig=[np.float32],
                out_sig=[np.float32],
                interp=N)
        # self.set_relative_rate(N)
    
        self.N = N


    def work(self, input_items, output_items):

      in_float  = input_items[0]
      out_list = output_items[0]

      tmp_arr = np.array([])
      for in_item in in_float:
        tmp_arr = np.concatenate((tmp_arr,np.array(bin_serializer(in_item, self.N))))

      out_list[:self.N*len(in_float)] = tmp_arr


      return self.N*len(in_float)


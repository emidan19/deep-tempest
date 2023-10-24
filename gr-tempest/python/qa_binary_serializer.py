#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020
#   Emilio Martínez <emilio.martinez@fing.edu.uy>
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

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

import numpy as np
from gnuradio import gr, gr_unittest
from gnuradio import blocks
from binary_serializer import binary_serializer  # grc-generated hier_block

class qa_binary_serializer(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_t(self):

        # Define sequences to test the block

        test_num_float = np.arange(500,700).astype('float32')

        for value in test_num_float:

            # set up fg

            # Get value and cast it as binary string  
            binstring = bin(int(value))[2:]
            # Fill string with 0's to get length 10
            binstring = '0'*(10-len(binstring))+binstring
            # Re-order string for LSB first
            binstring = binstring[::-1]
            # Casting string to float array
            expected_result  = np.array(list(binstring)).astype('float32')

            src = blocks.vector_source_f([value],False,1)
            bin_serializer = binary_serializer(M=10,N=16,offset=0)
            dst = blocks.vector_sink_f(1)

            self.tb.connect(src,bin_serializer,dst)

            # run fg
            self.tb.run()

            # check data
            actual_result = dst.data() 
            self.assertFloatTuplesAlmostEqual(expected_result,actual_result)


if __name__ == '__main__':
    gr_unittest.run(qa_binary_serializer)

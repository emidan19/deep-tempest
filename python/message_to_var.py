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


import numpy
import pmt
from gnuradio import gr

class message_to_var(gr.sync_block):
    """
    docstring for block message_to_var
    """
    def __init__(self, callback):
        gr.sync_block.__init__(self,
            name="message_to_var",
            in_sig=None, out_sig=None)

        self.callback = callback
        self.message_port_register_in(pmt.intern("port0")) 
        self.set_msg_handler(pmt.intern("port0"), self.msg_handler_function)

    def work(self, input_items, output_items):
        return len(input_items[0])

    def msg_handler_function(self, msg):
        if (pmt.is_pair(msg)):
            key = pmt.car(msg) # If is a pair, return the car, otherwise raise wrong_type
            val = pmt.cdr(msg) # If is a pair, return the cdr, otherwise raise wrong_type
            #(pmt::eq(key, pmt::string_to_symbol("ratio"))
            if(pmt.eq(pmt.intern("refresh_rate"),key) or 
                pmt.eq(pmt.intern("Hvisible"),key) or 
                pmt.eq(pmt.intern("Hblank"),key) or 
                pmt.eq(pmt.intern("Vblank"),key) or 
                pmt.eq(pmt.intern("Vvisible"),key)):
                if self.callback:
                    self.callback(pmt.to_python(val))

    def stop(self):
        return True

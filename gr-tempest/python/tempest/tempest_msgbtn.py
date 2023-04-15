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

from PyQt5 import Qt
from gnuradio import gr
import pmt

class tempest_msgbtn(gr.sync_block, Qt.QPushButton):
    """
    docstring for block tempest_msgbtn
    this was developed by ghostop14
    https://github.com/ghostop14/gr-guiextra/blob/maint-3.8/python/msgpushbutton.py
    """
    def __init__(self, lbl, msgName, msgValue, relBackColor, relFontColor):
        gr.sync_block.__init__(self, name = "tempest_msgbtn", in_sig = None, out_sig = None)
        Qt.QPushButton.__init__(self,lbl)
        
        self.lbl = lbl
        self.msgName = msgName
        self.msgValue = msgValue

        styleStr = ""
        if (relBackColor != 'default'):
            styleStr = "background-color: " + relBackColor + "; "
            
        if (relFontColor):
            styleStr += "color: " + relFontColor + "; "
            
        self.setStyleSheet(styleStr)

        self.clicked[bool].connect(self.onBtnClicked)
        
        self.message_port_register_out(pmt.intern("pressed"))

    def onBtnClicked(self, pressed):
        if type(self.msgValue) == int:
            self.message_port_pub(pmt.intern("pressed"),pmt.cons( pmt.intern(self.msgName), pmt.from_long(self.msgValue) ))
        elif type(self.msgValue) == float:
            self.message_port_pub(pmt.intern("pressed"),pmt.cons( pmt.intern(self.msgName), pmt.from_float(self.msgValue) ))
        elif type(self.msgValue) == str:
            self.message_port_pub(pmt.intern("pressed"),pmt.cons( pmt.intern(self.msgName), pmt.intern(self.msgValue) ))
        elif type(self.msgValue) == bool:
            self.message_port_pub(pmt.intern("pressed"),pmt.cons( pmt.intern(self.msgName), pmt.from_bool(self.msgValue) ))


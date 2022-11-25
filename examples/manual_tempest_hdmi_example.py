#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Manual Tempest Hdmi Example
# GNU Radio version: 3.8.5.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import video_sdl
from gnuradio.qtgui import Range, RangeWidget
import tempest

from gnuradio import qtgui

class manual_tempest_hdmi_example(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Manual Tempest Hdmi Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Manual Tempest Hdmi Example")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "manual_tempest_hdmi_example")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.refresh_rate = refresh_rate = 70
        self.Vsize = Vsize = 806
        self.Hsize = Hsize = 1344
        self.samp_rate = samp_rate = int(40e6)
        self.px_rate = px_rate = Hsize*Vsize*refresh_rate
        self.lines_offset = lines_offset = int(Vsize/2)
        self.inverted = inverted = 1
        self.interpolatedHsize = interpolatedHsize = int(Hsize/float(px_rate)*samp_rate)
        self.horizontal_offset = horizontal_offset = 0
        self.harmonic = harmonic = 1
        self.Vdisplay = Vdisplay = 768
        self.Hdisplay = Hdisplay = 1024

        ##################################################
        # Blocks
        ##################################################
        self._Vsize_range = Range(0, int(2160*1.5), 1, 806, 200)
        self._Vsize_win = RangeWidget(self._Vsize_range, self.set_Vsize, 'Vertical resolution (total)', "counter", int)
        self.top_grid_layout.addWidget(self._Vsize_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._lines_offset_range = Range(0, Vsize, 1, int(Vsize/2), 200)
        self._lines_offset_win = RangeWidget(self._lines_offset_range, self.set_lines_offset, 'Vertical offset', "counter_slider", int)
        self.top_layout.addWidget(self._lines_offset_win)
        # Create the options list
        self._inverted_options = [0, 1]
        # Create the labels list
        self._inverted_labels = ['Yes', 'No']
        # Create the combo box
        # Create the radio buttons
        self._inverted_group_box = Qt.QGroupBox('Inverted colors?' + ": ")
        self._inverted_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._inverted_button_group = variable_chooser_button_group()
        self._inverted_group_box.setLayout(self._inverted_box)
        for i, _label in enumerate(self._inverted_labels):
            radio_button = Qt.QRadioButton(_label)
            self._inverted_box.addWidget(radio_button)
            self._inverted_button_group.addButton(radio_button, i)
        self._inverted_callback = lambda i: Qt.QMetaObject.invokeMethod(self._inverted_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._inverted_options.index(i)))
        self._inverted_callback(self.inverted)
        self._inverted_button_group.buttonClicked[int].connect(
            lambda i: self.set_inverted(self._inverted_options[i]))
        self.top_layout.addWidget(self._inverted_group_box)
        self._horizontal_offset_range = Range(0, interpolatedHsize, 1, 0, 200)
        self._horizontal_offset_win = RangeWidget(self._horizontal_offset_range, self.set_horizontal_offset, 'Horizontal offset', "counter_slider", int)
        self.top_layout.addWidget(self._horizontal_offset_win)
        self._Hsize_range = Range(0, int(4096*1.5), 1, 1344, 200)
        self._Hsize_win = RangeWidget(self._Hsize_range, self.set_Hsize, 'Horizontal resolution (total)', "counter", int)
        self.top_grid_layout.addWidget(self._Hsize_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.video_sdl_sink_0_0_0 = video_sdl.sink_s(0, interpolatedHsize, Vsize, 0, 1920, 1080)
        self.tempest_normalize_flow_0 = tempest.normalize_flow(10, 245, interpolatedHsize, 1e-2, 0.1)
        self.tempest_fine_sampling_synchronization_0 = tempest.fine_sampling_synchronization(interpolatedHsize, Vsize, 1, 20.0/interpolatedHsize, 1e-2/(Hsize*Vsize))
        self._refresh_rate_range = Range(0, 240, 1, 70, 200)
        self._refresh_rate_win = RangeWidget(self._refresh_rate_range, self.set_refresh_rate, 'Refresh Rate (Hz)', "counter", float)
        self.top_layout.addWidget(self._refresh_rate_win)
        self._harmonic_range = Range(1, 10, 1, 1, 200)
        self._harmonic_win = RangeWidget(self._harmonic_range, self.set_harmonic, 'Harmonic', "counter_slider", float)
        self.top_layout.addWidget(self._harmonic_win)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_float_to_short_0_0 = blocks.float_to_short(1, inverted)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/emidan19/Downloads/grabacion_HDMI', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, interpolatedHsize*lines_offset+horizontal_offset)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_mag_0, 0), (self.tempest_normalize_flow_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_float_to_short_0_0, 0), (self.video_sdl_sink_0_0_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.tempest_fine_sampling_synchronization_0, 0))
        self.connect((self.tempest_fine_sampling_synchronization_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.tempest_normalize_flow_0, 0), (self.blocks_float_to_short_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "manual_tempest_hdmi_example")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_refresh_rate(self):
        return self.refresh_rate

    def set_refresh_rate(self, refresh_rate):
        self.refresh_rate = refresh_rate
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate)

    def get_Vsize(self):
        return self.Vsize

    def set_Vsize(self, Vsize):
        self.Vsize = Vsize
        self.set_lines_offset(int(self.Vsize/2))
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate)
        self.tempest_fine_sampling_synchronization_0.set_Htotal_Vtotal(self.interpolatedHsize, self.Vsize)

    def get_Hsize(self):
        return self.Hsize

    def set_Hsize(self, Hsize):
        self.Hsize = Hsize
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_px_rate(self):
        return self.px_rate

    def set_px_rate(self, px_rate):
        self.px_rate = px_rate
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))

    def get_lines_offset(self):
        return self.lines_offset

    def set_lines_offset(self, lines_offset):
        self.lines_offset = lines_offset
        self.blocks_delay_0.set_dly(self.interpolatedHsize*self.lines_offset+self.horizontal_offset)

    def get_inverted(self):
        return self.inverted

    def set_inverted(self, inverted):
        self.inverted = inverted
        self._inverted_callback(self.inverted)
        self.blocks_float_to_short_0_0.set_scale(self.inverted)

    def get_interpolatedHsize(self):
        return self.interpolatedHsize

    def set_interpolatedHsize(self, interpolatedHsize):
        self.interpolatedHsize = interpolatedHsize
        self.blocks_delay_0.set_dly(self.interpolatedHsize*self.lines_offset+self.horizontal_offset)
        self.tempest_fine_sampling_synchronization_0.set_Htotal_Vtotal(self.interpolatedHsize, self.Vsize)

    def get_horizontal_offset(self):
        return self.horizontal_offset

    def set_horizontal_offset(self, horizontal_offset):
        self.horizontal_offset = horizontal_offset
        self.blocks_delay_0.set_dly(self.interpolatedHsize*self.lines_offset+self.horizontal_offset)

    def get_harmonic(self):
        return self.harmonic

    def set_harmonic(self, harmonic):
        self.harmonic = harmonic

    def get_Vdisplay(self):
        return self.Vdisplay

    def set_Vdisplay(self, Vdisplay):
        self.Vdisplay = Vdisplay

    def get_Hdisplay(self):
        return self.Hdisplay

    def set_Hdisplay(self, Hdisplay):
        self.Hdisplay = Hdisplay





def main(top_block_cls=manual_tempest_hdmi_example, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Manual  Simulated Tempest TMDS Example
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

import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from binary_serializer import binary_serializer  # grc-generated hier_block
from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import filter
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import video_sdl
from gnuradio.qtgui import Range, RangeWidget
import tempest

from gnuradio import qtgui

class manual_simulated_tempest_TMDS_example(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Manual  Simulated Tempest TMDS Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Manual  Simulated Tempest TMDS Example")
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

        self.settings = Qt.QSettings("GNU Radio", "manual_simulated_tempest_TMDS_example")

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
        self.refresh_rate = refresh_rate = 60
        self.Vsize = Vsize = 1000
        self.Hsize = Hsize = 1800
        self.usrp_rate = usrp_rate = int(50e6/100)
        self.px_rate = px_rate = Hsize*Vsize*refresh_rate/100
        self.inter = inter = 10
        self.samp_rate = samp_rate = int(px_rate*inter)
        self.rectangular_pulse = rectangular_pulse = [0.7/255]*inter
        self.noise = noise = 0
        self.lines_offset = lines_offset = int(Vsize/2)
        self.inverted = inverted = 1
        self.interpolatedHsize = interpolatedHsize = int(Hsize/float(px_rate)*usrp_rate)
        self.horizontal_offset = horizontal_offset = 0
        self.harmonic = harmonic = 1
        self.freq = freq = 0
        self.epsilon_channel = epsilon_channel = 0
        self.decim = decim = inter
        self.Vvisible = Vvisible = 900
        self.Vdisplay = Vdisplay = Vsize
        self.Hvisible = Hvisible = 1600
        self.Hdisplay = Hdisplay = Hsize

        ##################################################
        # Blocks
        ##################################################
        self._noise_range = Range(0, 2e-1, 1e-4, 0, 200)
        self._noise_win = RangeWidget(self._noise_range, self.set_noise, 'Noise Power', "counter_slider", float)
        self.top_grid_layout.addWidget(self._noise_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._lines_offset_range = Range(0, Vsize, 1, int(Vsize/2), 200)
        self._lines_offset_win = RangeWidget(self._lines_offset_range, self.set_lines_offset, 'Vertical offset', "counter_slider", int)
        self.top_grid_layout.addWidget(self._lines_offset_win, 3, 1, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
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
        self.top_grid_layout.addWidget(self._inverted_group_box, 2, 1, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._horizontal_offset_range = Range(0, interpolatedHsize, 1, 0, 200)
        self._horizontal_offset_win = RangeWidget(self._horizontal_offset_range, self.set_horizontal_offset, 'Horizontal offset', "counter_slider", int)
        self.top_grid_layout.addWidget(self._horizontal_offset_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._harmonic_range = Range(1, 10, 1, 1, 200)
        self._harmonic_win = RangeWidget(self._harmonic_range, self.set_harmonic, 'Harmonic', "counter_slider", float)
        self.top_grid_layout.addWidget(self._harmonic_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._freq_range = Range(-1, 1, 1e-5, 0, 200)
        self._freq_win = RangeWidget(self._freq_range, self.set_freq, 'Frequency Error (normalized)', "counter_slider", float)
        self.top_grid_layout.addWidget(self._freq_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._epsilon_channel_range = Range(-0.1, 0.1, 10e-6, 0, 200)
        self._epsilon_channel_win = RangeWidget(self._epsilon_channel_range, self.set_epsilon_channel, 'Sampling error', "counter_slider", float)
        self.top_grid_layout.addWidget(self._epsilon_channel_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.video_sdl_sink_0_0_0 = video_sdl.sink_s(0, interpolatedHsize, Vsize, 0, Hsize, Vsize)
        self.tempest_tempest_msgbtn_0 = _tempest_tempest_msgbtn_0_toggle_button = tempest.tempest_msgbtn('Take screenshot', 'pressed',True,"default","default")
        self.tempest_tempest_msgbtn_0 = _tempest_tempest_msgbtn_0_toggle_button
        self.top_layout.addWidget(_tempest_tempest_msgbtn_0_toggle_button)
        self.tempest_normalize_flow_0 = tempest.normalize_flow(10, 245, interpolatedHsize, 1e-2, 0.1)
        self.tempest_fine_sampling_synchronization_0 = tempest.fine_sampling_synchronization(interpolatedHsize, Vsize, 1, 100.0/interpolatedHsize, 1.0/(interpolatedHsize*Vsize))
        self.tempest_buttonToFileSink_0 = tempest.buttonToFileSink('/home/emidan19/Desktop/captura_simulada', interpolatedHsize, Hsize, Vsize)
        self.tempest_TMDS_image_source_0 = tempest.TMDS_image_source('/home/emidan19/Desktop/deep-tempest/drunet/notebooks/results_l1_l2/imagenes/originalRGB/train/Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.png', 1, True)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=usrp_rate,
                decimation=samp_rate,
                taps=None,
                fractional_bw=0.4)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            1
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise,
            frequency_offset=freq,
            epsilon=epsilon_channel+1,
            taps=[1],
            noise_seed=0,
            block_tags=False)
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_float_to_short_0 = blocks.float_to_short(1, inverted)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, interpolatedHsize*lines_offset+horizontal_offset)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.blocks_add_xx_0 = blocks.add_vff(1)
        self.binary_serializer_0_0_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.binary_serializer_0_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.binary_serializer_0 = binary_serializer(
            M=10,
            N=16,
            offset=0,
        )
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, px_rate*harmonic, 4, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.tempest_tempest_msgbtn_0, 'pressed'), (self.tempest_buttonToFileSink_0, 'en'))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.binary_serializer_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.binary_serializer_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.binary_serializer_0_0_0, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.tempest_normalize_flow_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.tempest_buttonToFileSink_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.blocks_float_to_short_0, 0), (self.video_sdl_sink_0_0_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.tempest_fine_sampling_synchronization_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 0), (self.binary_serializer_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 1), (self.binary_serializer_0_0, 0))
        self.connect((self.tempest_TMDS_image_source_0, 2), (self.binary_serializer_0_0_0, 0))
        self.connect((self.tempest_fine_sampling_synchronization_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.tempest_normalize_flow_0, 0), (self.blocks_float_to_short_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "manual_simulated_tempest_TMDS_example")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_refresh_rate(self):
        return self.refresh_rate

    def set_refresh_rate(self, refresh_rate):
        self.refresh_rate = refresh_rate
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/100)

    def get_Vsize(self):
        return self.Vsize

    def set_Vsize(self, Vsize):
        self.Vsize = Vsize
        self.set_Vdisplay(self.Vsize)
        self.set_lines_offset(int(self.Vsize/2))
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/100)
        self.tempest_fine_sampling_synchronization_0.set_Htotal_Vtotal(self.interpolatedHsize, self.Vsize)

    def get_Hsize(self):
        return self.Hsize

    def set_Hsize(self, Hsize):
        self.Hsize = Hsize
        self.set_Hdisplay(self.Hsize)
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.usrp_rate))
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate/100)

    def get_usrp_rate(self):
        return self.usrp_rate

    def set_usrp_rate(self, usrp_rate):
        self.usrp_rate = usrp_rate
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.usrp_rate))

    def get_px_rate(self):
        return self.px_rate

    def set_px_rate(self, px_rate):
        self.px_rate = px_rate
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.usrp_rate))
        self.set_samp_rate(int(self.px_rate*self.inter))
        self.analog_sig_source_x_0.set_frequency(self.px_rate*self.harmonic)

    def get_inter(self):
        return self.inter

    def set_inter(self, inter):
        self.inter = inter
        self.set_decim(self.inter)
        self.set_rectangular_pulse([0.7/255]*self.inter)
        self.set_samp_rate(int(self.px_rate*self.inter))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)

    def get_rectangular_pulse(self):
        return self.rectangular_pulse

    def set_rectangular_pulse(self, rectangular_pulse):
        self.rectangular_pulse = rectangular_pulse

    def get_noise(self):
        return self.noise

    def set_noise(self, noise):
        self.noise = noise
        self.channels_channel_model_0.set_noise_voltage(self.noise)

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
        self.blocks_float_to_short_0.set_scale(self.inverted)

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
        self.analog_sig_source_x_0.set_frequency(self.px_rate*self.harmonic)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.channels_channel_model_0.set_frequency_offset(self.freq)

    def get_epsilon_channel(self):
        return self.epsilon_channel

    def set_epsilon_channel(self, epsilon_channel):
        self.epsilon_channel = epsilon_channel
        self.channels_channel_model_0.set_timing_offset(self.epsilon_channel+1)

    def get_decim(self):
        return self.decim

    def set_decim(self, decim):
        self.decim = decim

    def get_Vvisible(self):
        return self.Vvisible

    def set_Vvisible(self, Vvisible):
        self.Vvisible = Vvisible

    def get_Vdisplay(self):
        return self.Vdisplay

    def set_Vdisplay(self, Vdisplay):
        self.Vdisplay = Vdisplay

    def get_Hvisible(self):
        return self.Hvisible

    def set_Hvisible(self, Hvisible):
        self.Hvisible = Hvisible

    def get_Hdisplay(self):
        return self.Hdisplay

    def set_Hdisplay(self, Hdisplay):
        self.Hdisplay = Hdisplay





def main(top_block_cls=manual_simulated_tempest_TMDS_example, options=None):

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

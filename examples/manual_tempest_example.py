#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Manual Tempest Example
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

from FFT_autocorrelation import FFT_autocorrelation  # grc-generated hier_block
from Keep_1_in_N_Frames import Keep_1_in_N_Frames  # grc-generated hier_block
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import gr
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio import video_sdl
from gnuradio.qtgui import Range, RangeWidget
from math import pi
import tempest

from gnuradio import qtgui

class manual_tempest_example(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Manual Tempest Example")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Manual Tempest Example")
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

        self.settings = Qt.QSettings("GNU Radio", "manual_tempest_example")

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
        self.samp_rate = samp_rate = int(50e6)
        self.px_rate = px_rate = Hsize*Vsize*refresh_rate
        self.Hvisible = Hvisible = 1920
        self.interpolatedHsize = interpolatedHsize = int(Hsize/float(px_rate)*samp_rate)
        self.Vvisible = Vvisible = 1080
        self.Hblank = Hblank = Hsize-Hvisible
        self.inverted = inverted = 1
        self.interpolatedHscreen = interpolatedHscreen = int(Hvisible/float(px_rate)*samp_rate)
        self.interpolatedHblank = interpolatedHblank = int(Hblank/float(px_rate)*samp_rate)
        self.harmonic = harmonic = 1
        self.fft_size = fft_size = 2**21
        self.Vblank = Vblank = Vsize-Vvisible
        self.DroppedFrames = DroppedFrames = 50
        self.DelaySyncDetector = DelaySyncDetector = interpolatedHsize*1

        ##################################################
        # Blocks
        ##################################################
        self.tab_m = Qt.QTabWidget()
        self.tab_m_widget_0 = Qt.QWidget()
        self.tab_m_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.tab_m_widget_0)
        self.tab_m_grid_layout_0 = Qt.QGridLayout()
        self.tab_m_layout_0.addLayout(self.tab_m_grid_layout_0)
        self.tab_m.addTab(self.tab_m_widget_0, 'Tempest Main Tab')
        self.tab_m_widget_1 = Qt.QWidget()
        self.tab_m_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.tab_m_widget_1)
        self.tab_m_grid_layout_1 = Qt.QGridLayout()
        self.tab_m_layout_1.addLayout(self.tab_m_grid_layout_1)
        self.tab_m.addTab(self.tab_m_widget_1, 'Autocorrelation Plot Tab')
        self.top_layout.addWidget(self.tab_m)
        self._Vsize_range = Range(0, int(2160*1.5), 1, 1000, 200)
        self._Vsize_win = RangeWidget(self._Vsize_range, self.set_Vsize, 'Vertical resolution (total)', "counter", int)
        self.tab_m_layout_0.addWidget(self._Vsize_win)
        self._refresh_rate_range = Range(0, 240, 1, 60, 200)
        self._refresh_rate_win = RangeWidget(self._refresh_rate_range, self.set_refresh_rate, 'Refresh Rate (Hz)', "counter", float)
        self.tab_m_layout_0.addWidget(self._refresh_rate_win)
        # Create the options list
        self._inverted_options = [0, 1, 2, 3, 4]
        # Create the labels list
        self._inverted_labels = ['Yes', 'No', '2', '3', '4']
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
        self.tab_m_layout_0.addWidget(self._inverted_group_box)
        self._harmonic_range = Range(1, 10, 1, 1, 200)
        self._harmonic_win = RangeWidget(self._harmonic_range, self.set_harmonic, 'Harmonic', "counter_slider", float)
        self.tab_m_layout_0.addWidget(self._harmonic_win)
        self._DroppedFrames_range = Range(1, 100, 1, 50, 200)
        self._DroppedFrames_win = RangeWidget(self._DroppedFrames_range, self.set_DroppedFrames, 'DroppedFrames', "counter_slider", float)
        self.tab_m_layout_0.addWidget(self._DroppedFrames_win)
        self._DelaySyncDetector_range = Range(0, Vsize*interpolatedHsize, interpolatedHsize*1, interpolatedHsize*1, 200)
        self._DelaySyncDetector_win = RangeWidget(self._DelaySyncDetector_range, self.set_DelaySyncDetector, 'DelaySyncDetector', "counter_slider", float)
        self.tab_m_layout_0.addWidget(self._DelaySyncDetector_win)
        self.video_sdl_sink_0_0_0 = video_sdl.sink_s(0, interpolatedHsize, Vsize, 0, 1920, 1080)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "recv_frame_size= 65536, num_recv_frames=128")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_center_freq((px_rate*9)*harmonic, 0)
        self.uhd_usrp_source_0.set_gain(50, 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        self.tempest_sync_detector_0_0 = tempest.sync_detector(interpolatedHscreen, Vsize-Vblank, interpolatedHblank, Vblank)
        self.tempest_normalize_flow_0 = tempest.normalize_flow(10, 245, interpolatedHsize, 1e-2, 0.1)
        self.tempest_fft_peak_fine_sampling_sync_0_0 = tempest.fft_peak_fine_sampling_sync(samp_rate, int(fft_size), refresh_rate, Vsize, interpolatedHsize, True)
        self.ratio_finder_toggle_on_and_off = _ratio_finder_toggle_on_and_off_toggle_button = tempest.tempest_msgbtn('ratio_finder_toggle_on_and_off', 'pressed',True,"default","default")
        self.ratio_finder_toggle_on_and_off = _ratio_finder_toggle_on_and_off_toggle_button
        self.top_layout.addWidget(_ratio_finder_toggle_on_and_off_toggle_button)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_f(
            int(fft_size/2), #size
            samp_rate, #samp_rate
            "", #name
            1 #number of inputs
        )
        self.qtgui_time_sink_x_1.set_update_time(2.0)
        self.qtgui_time_sink_x_1.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 19.0, 0, 0, "peak_1")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(True)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.pyqwidget(), Qt.QWidget)
        self.tab_m_layout_1.addWidget(self._qtgui_time_sink_x_1_win)
        self.blocks_float_to_short_0 = blocks.float_to_short(1, inverted)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, DelaySyncDetector)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.Keep_1_in_N_Frames_0 = Keep_1_in_N_Frames(
            fac_decimation=DroppedFrames,
            fft_size=interpolatedHsize*Vsize*4,
        )
        self._Hsize_range = Range(0, int(4096*1.5), 1, 1800, 200)
        self._Hsize_win = RangeWidget(self._Hsize_range, self.set_Hsize, 'Horizontal resolution (total)', "counter", int)
        self.tab_m_layout_0.addWidget(self._Hsize_win)
        self.FFT_autocorrelation_0_0 = FFT_autocorrelation(
            alpha=1.0,
            fft_size=int(fft_size),
        )


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ratio_finder_toggle_on_and_off, 'pressed'), (self.tempest_fft_peak_fine_sampling_sync_0_0, 'en'))
        self.msg_connect((self.tempest_fft_peak_fine_sampling_sync_0_0, 'en'), (self.FFT_autocorrelation_0_0, 'en'))
        self.msg_connect((self.tempest_fft_peak_fine_sampling_sync_0_0, 'en'), (self.tempest_sync_detector_0_0, 'en'))
        self.msg_connect((self.tempest_fft_peak_fine_sampling_sync_0_0, 'rate'), (self.uhd_usrp_source_0, 'command'))
        self.connect((self.FFT_autocorrelation_0_0, 0), (self.tempest_fft_peak_fine_sampling_sync_0_0, 0))
        self.connect((self.Keep_1_in_N_Frames_0, 0), (self.FFT_autocorrelation_0_0, 0))
        self.connect((self.Keep_1_in_N_Frames_0, 0), (self.tempest_sync_detector_0_0, 0))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.tempest_normalize_flow_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_float_to_short_0, 0), (self.video_sdl_sink_0_0_0, 0))
        self.connect((self.tempest_fft_peak_fine_sampling_sync_0_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.tempest_normalize_flow_0, 0), (self.blocks_float_to_short_0, 0))
        self.connect((self.tempest_sync_detector_0_0, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.Keep_1_in_N_Frames_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "manual_tempest_example")
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
        self.set_Vblank(self.Vsize-self.Vvisible)
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate)
        self.Keep_1_in_N_Frames_0.set_fft_size(self.interpolatedHsize*self.Vsize*4)

    def get_Hsize(self):
        return self.Hsize

    def set_Hsize(self, Hsize):
        self.Hsize = Hsize
        self.set_Hblank(self.Hsize-self.Hvisible)
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))
        self.set_px_rate(self.Hsize*self.Vsize*self.refresh_rate)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_interpolatedHblank(int(self.Hblank/float(self.px_rate)*self.samp_rate))
        self.set_interpolatedHscreen(int(self.Hvisible/float(self.px_rate)*self.samp_rate))
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_px_rate(self):
        return self.px_rate

    def set_px_rate(self, px_rate):
        self.px_rate = px_rate
        self.set_interpolatedHblank(int(self.Hblank/float(self.px_rate)*self.samp_rate))
        self.set_interpolatedHscreen(int(self.Hvisible/float(self.px_rate)*self.samp_rate))
        self.set_interpolatedHsize(int(self.Hsize/float(self.px_rate)*self.samp_rate))
        self.uhd_usrp_source_0.set_center_freq((self.px_rate*9)*self.harmonic, 0)

    def get_Hvisible(self):
        return self.Hvisible

    def set_Hvisible(self, Hvisible):
        self.Hvisible = Hvisible
        self.set_Hblank(self.Hsize-self.Hvisible)
        self.set_interpolatedHscreen(int(self.Hvisible/float(self.px_rate)*self.samp_rate))

    def get_interpolatedHsize(self):
        return self.interpolatedHsize

    def set_interpolatedHsize(self, interpolatedHsize):
        self.interpolatedHsize = interpolatedHsize
        self.set_DelaySyncDetector(self.interpolatedHsize*1)
        self.Keep_1_in_N_Frames_0.set_fft_size(self.interpolatedHsize*self.Vsize*4)

    def get_Vvisible(self):
        return self.Vvisible

    def set_Vvisible(self, Vvisible):
        self.Vvisible = Vvisible
        self.set_Vblank(self.Vsize-self.Vvisible)

    def get_Hblank(self):
        return self.Hblank

    def set_Hblank(self, Hblank):
        self.Hblank = Hblank
        self.set_interpolatedHblank(int(self.Hblank/float(self.px_rate)*self.samp_rate))

    def get_inverted(self):
        return self.inverted

    def set_inverted(self, inverted):
        self.inverted = inverted
        self._inverted_callback(self.inverted)
        self.blocks_float_to_short_0.set_scale(self.inverted)

    def get_interpolatedHscreen(self):
        return self.interpolatedHscreen

    def set_interpolatedHscreen(self, interpolatedHscreen):
        self.interpolatedHscreen = interpolatedHscreen

    def get_interpolatedHblank(self):
        return self.interpolatedHblank

    def set_interpolatedHblank(self, interpolatedHblank):
        self.interpolatedHblank = interpolatedHblank

    def get_harmonic(self):
        return self.harmonic

    def set_harmonic(self, harmonic):
        self.harmonic = harmonic
        self.uhd_usrp_source_0.set_center_freq((self.px_rate*9)*self.harmonic, 0)

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size
        self.FFT_autocorrelation_0_0.set_fft_size(int(self.fft_size))

    def get_Vblank(self):
        return self.Vblank

    def set_Vblank(self, Vblank):
        self.Vblank = Vblank

    def get_DroppedFrames(self):
        return self.DroppedFrames

    def set_DroppedFrames(self, DroppedFrames):
        self.DroppedFrames = DroppedFrames
        self.Keep_1_in_N_Frames_0.set_fac_decimation(self.DroppedFrames)

    def get_DelaySyncDetector(self):
        return self.DelaySyncDetector

    def set_DelaySyncDetector(self, DelaySyncDetector):
        self.DelaySyncDetector = DelaySyncDetector
        self.blocks_delay_0_0.set_dly(self.DelaySyncDetector)





def main(top_block_cls=manual_tempest_example, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

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

import sys
import tkinter as tk

import matplotlib.backend_bases
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tabulate import tabulate

import matplotlib.pyplot as plt
import numpy as np
# from scipy import signal as sp
import scipy as sp

import pyemgpipeline


# import pandas as pd
# from PIL import Image, ImageTk


class LcaFileUtils:
    @staticmethod
    def ask_open_txt_file_name():
        file_name = tk.filedialog.askopenfilename(defaultextension=".txt",
                                                  filetypes=[("Text files", "*.txt"),
                                                             ("All files", "*.*")])
        return file_name

    @staticmethod
    def ask_save_txt_file_name():
        file_name = tk.filedialog.asksaveasfilename(defaultextension=".txt",
                                                    filetypes=[("Text files", "*.txt"),
                                                               ("All files", "*.*")])
        return file_name

    @staticmethod
    def read(file_name):
        time_data = []
        hammer_raw = []
        emg_raw = []
        reading = False
        with open(file_name, 'r') as file:
            for line in file:
                if reading:
                    try:
                        values = [float(value.replace(',', '.')) for value in line.split('\t')]
                        time_data.append(values[0])
                        hammer_raw.append(values[1])
                        emg_raw.append(values[2])
                    except ValueError:
                        break
                elif line.startswith('0\t'):
                    reading = True
        return time_data, hammer_raw, emg_raw


# static methods for signal handling
class LcaSignalUtils:
    @staticmethod
    def find_peaks(xdata, ydata_post, ydata_src, xlim, height):
        # Find indices in range
        if xlim is None:
            xrange = xdata
            yrange_post = ydata_post
            yrange_src = ydata_src
        else:
            irange = np.where((xdata >= xlim[0]) & (xdata <= xlim[1]))[0]
            # Find x's and y's in range
            xrange = np.array(xdata)[irange]
            yrange_post = np.array(ydata_post)[irange]
            yrange_src = np.array(ydata_src)[irange]

        # Find only peaks with a minimum height
        ipeaks, _ = sp.signal.find_peaks(yrange_post, height=height)

        xpeaks = [round(xrange[i], 3) for i in ipeaks]
        ypeaks_post = [round(yrange_post[i], 3) for i in ipeaks]
        ypeaks_src = [round(yrange_src[i], 3) for i in ipeaks]
        return xpeaks, ypeaks_post, ypeaks_src

    @staticmethod
    def find_hammer_peaks(time_data, hammer, xlim):
        height = 0.1
        peak_times_hammer, peak_values_hammer, _ = LcaSignalUtils.find_peaks(time_data, hammer, hammer, xlim, height)
        return peak_times_hammer, peak_values_hammer

    @staticmethod
    def find_emg_peaks(time, emg, latency_postprocess_name, xlim):
        emg_postprocessed = LcaSignalUtils.postprocess_emg_filtered(emg, latency_postprocess_name)
        if latency_postprocess_name == 'gradient':
            height = 0.005
        else:
            height = 0.06
        peak_times_emg, peak_values_emg_post, peak_values_emg_src = LcaSignalUtils.find_peaks(time, emg_postprocessed,
                                                                                              emg,
                                                                                              xlim, height)
        return peak_times_emg, peak_values_emg_post, peak_values_emg_src

    ####################################################################################################################

    # DEFAULT_HAMMER_FILTER_NAME = '-'
    # DEFAULT_EMG_FILTER_NAME = 'rolling_rms'
    DEFAULT_EMG_FILTER_NAME = '-'
    DEFAULT_LATENCY_POSTPROCESS_NAME = 'max'

    ####################################################################################################################

    SAMPLING_FREQ = 1000  # Hz
    NYQUIST_FREQ = SAMPLING_FREQ / 2

    @staticmethod
    def normalize_frequency(freq):
        return freq / LcaSignalUtils.NYQUIST_FREQ

    @staticmethod
    def build_digital_freqs(btype):
        if btype == 'lowpass':
            freqs = LcaSignalUtils.LOW_CUTOFF_FREQ
        elif btype == 'highpass':
            freqs = LcaSignalUtils.HIGH_CUTOFF_FREQ
        elif btype == 'bandpass':
            freqs = (LcaSignalUtils.LOW_CUTOFF_FREQ, LcaSignalUtils.HIGH_CUTOFF_FREQ)
        return freqs

    def build_digital_normalized_freqs(btype):
        if btype == 'lowpass':
            normalized_freqs = LcaSignalUtils.normalize_frequency(LcaSignalUtils.LOW_CUTOFF_FREQ)
        elif btype == 'highpass':
            normalized_freqs = LcaSignalUtils.normalize_frequency(LcaSignalUtils.HIGH_CUTOFF_FREQ)
        elif btype == 'bandpass':
            normalized_freqs = (
                LcaSignalUtils.normalize_frequency(LcaSignalUtils.LOW_CUTOFF_FREQ),
                LcaSignalUtils.normalize_frequency(LcaSignalUtils.HIGH_CUTOFF_FREQ))
        return normalized_freqs

    ####################################################################################################################

    # @staticmethod
    # def filter_hammer(hammer, hammer_filter_name):
    #     if hammer_filter_name == '-':
    #         return hammer
    #     elif hammer_filter_name == 'default':
    #         return LcaSignalUtils.filter_hammer(hammer, LcaSignalUtils.DEFAULT_HAMMER_FILTER_NAME)
    #     else:
    #         return None

    ####################################################################################################################

    @staticmethod
    def rolling_rms_filter(signal, half_window_size):
        window_size = 2 * half_window_size + 1
        window = np.ones(window_size) / float(window_size)

        return np.sqrt(
            sp.signal.fftconvolve(
                np.power(signal, 2),
                window,
                'same'))

    @staticmethod
    def filter_rolling_rms(signal):
        return LcaSignalUtils.rolling_rms_filter(signal, 2)
        # return LcaSignalUtils.rolling_rms_filter(signal, 3)
        # return LcaSignalUtils.rolling_rms_filter(rolling_rms_filter(signal, 3), 1)

    ####################################################################################################################

    FILTER_ORDER = 4

    HIGH_CUTOFF_FREQ = 60  # Hz
    LOW_CUTOFF_FREQ = 30  # Hz

    STOP_ATTENUATION = 40  # dB

    @staticmethod
    def build_digital_ba_butter_filter(btype):
        b, a = sp.signal.butter(LcaSignalUtils.FILTER_ORDER, LcaSignalUtils.build_digital_normalized_freqs(btype),
                                btype=btype, analog=False, output='ba')
        return b, a

    # @staticmethod
    # def build_digital_sos_butter_filter(btype):
    #     sos = sp.signal.butter(LcaSignalUtils.FILTER_ORDER, LcaSignalUtils.build_digital_normalized_freqs(btype),
    #                            btype=btype, analog=False, output='sos')
    #     return sos

    @staticmethod
    def filter_butter_lowpass_filtfilt(signal):
        b, a = LcaSignalUtils.build_digital_ba_butter_filter('lowpass')
        filtered = sp.signal.filtfilt(b, a, signal)
        return filtered

    @staticmethod
    def filter_butter_highpass_filtfilt(signal):
        b, a = LcaSignalUtils.build_digital_ba_butter_filter('highpass')
        filtered = sp.signal.filtfilt(b, a, signal)
        return filtered

    @staticmethod
    def filter_butter_bandpass_filtfilt(signal):
        b, a = LcaSignalUtils.build_digital_ba_butter_filter('lowpass')
        filtered = sp.signal.filtfilt(b, a, signal)
        return filtered

    @staticmethod
    def filter_butter_lowpass_ifilter(signal):
        b, a = LcaSignalUtils.build_digital_ba_butter_filter('lowpass')
        filtered = sp.signal.ifilter(b, a, signal)
        return filtered

    ####################################################################################################################

    @staticmethod
    def build_digital_ba_cheby2_filter(btype):
        b, a = sp.signal.cheby2(LcaSignalUtils.FILTER_ORDER, LcaSignalUtils.STOP_ATTENUATION,
                                LcaSignalUtils.build_digital_normalized_freqs(btype),
                                btype=btype, analog=False, output='ba')
        return b, a

    @staticmethod
    def filter_cheby2_lowpass_filtfilt(signal):
        b, a = LcaSignalUtils.build_digital_ba_cheby2_filter('lowpass')
        filtered = sp.signal.filtfilt(b, a, signal)
        return filtered

    ####################################################################################################################

    @staticmethod
    def filter_savgol(signal):
        # return sp.signal.savgol_filter(signal, window_length=45, polyorder=4)
        return sp.signal.savgol_filter(signal, window_length=11, polyorder=3)

    ####################################################################################################################

    @staticmethod
    def filter_conventional(signal):
        lowpass = 6
        sfreq = 2000
        high_band = 10
        low_band = 35
        amplification_factor = 2

        # normalise cut-off frequencies to sampling frequency
        high_band = high_band / (sfreq / 2)
        low_band = low_band / (sfreq / 2)

        # create bandpass filter for EMG
        b1, a1 = sp.signal.butter(4, [high_band, low_band], btype='bandpass')

        # process EMG signal: filter EMG
        emg_filtered = sp.signal.filtfilt(b1, a1, signal)

        # process EMG signal: rectify
        emg_rectified = abs(emg_filtered)

        # create lowpass filter and apply to rectified signal to get EMG envelope
        lowpass = lowpass / (sfreq / 2)
        b2, a2 = sp.signal.butter(4, lowpass, btype='lowpass')
        emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

        emg_amplified = emg_envelope * amplification_factor

        return emg_amplified

    ####################################################################################################################

    NOTCH_FREQ = 50.0  # Frequency to be removed (notch frequency)
    NOTCH_BANDWIDTH = 2.0  # Bandwidth of the notch in Hz
    NOTCH_ATTENUATION = 20  # Desired attenuation at 50 Hz in dB

    @staticmethod
    def filter_notch_filtfilt(signal):
        Q = LcaSignalUtils.NOTCH_FREQ / LcaSignalUtils.NOTCH_BANDWIDTH
        # Design the notch filter
        b, a = sp.signal.iirnotch(LcaSignalUtils.NOTCH_FREQ, Q, LcaSignalUtils.SAMPLING_FREQ)
        # Adjust the gain to achieve the desired attenuation
        gain = 10 ** (-LcaSignalUtils.NOTCH_ATTENUATION / 20)
        b *= gain

        # filtered_emg = sp.signal.lfilter(b, a, signal)
        filtered_emg = sp.signal.filtfilt(b, a, signal)

        return filtered_emg

    @staticmethod
    def filter_pyemgpipeline(signal):
        # Ensure the input is a NumPy array
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        emg_trial = pyemgpipeline.wrappers.EMGMeasurement(signal, hz=1000)

        # emg_trial.remove_dc_offset()

        # emg_trial.apply_bandpass_filter(lowcut=20, highcut=500, order=4)

        # emg_trial.apply_full_wave_rectification()

        # TypeError: EMGMeasurement.apply_linear_envelope() got an unexpected keyword argument 'lowcut'
        emg_trial.apply_linear_envelope(lowcut=6, order=2)

        return emg_trial

    @staticmethod
    def filter(signal, filter_name):
        if filter_name == '-':
            return signal
        elif filter_name == 'default':
            return LcaSignalUtils.filter(signal, LcaSignalUtils.DEFAULT_EMG_FILTER_NAME)
        elif filter_name == 'savgol':
            return LcaSignalUtils.filter_savgol(signal)
        elif filter_name == 'rolling_rms':
            return LcaSignalUtils.filter_rolling_rms(signal)
        elif filter_name == 'butter_lowpass_filtfilt':
            return LcaSignalUtils.filter_butter_lowpass_filtfilt(signal)
        elif filter_name == 'butter_highpass_filtfilt':
            return LcaSignalUtils.filter_butter_highpass_filtfilt(signal)
        elif filter_name == 'butter_bandpass_filtfilt':
            return LcaSignalUtils.filter_butter_bandpass_filtfilt(signal)
        elif filter_name == 'butter_lowpass_ifilter':
            return LcaSignalUtils.filter_butter_lowpass_ifilter(signal)
        elif filter_name == 'cheby2_lowpass_filtfilt':
            return LcaSignalUtils.filter_cheby2_lowpass_filtfilt(signal)
        elif filter_name == 'notch_filtfilt':
            return LcaSignalUtils.filter_notch_filtfilt(signal)
        elif filter_name == 'conventional':
            return LcaSignalUtils.filter_conventional(signal)
        elif filter_name == 'pyemgpipeline':
            return LcaSignalUtils.filter_pyemgpipeline(signal)
        else:
            return None

    @staticmethod
    def postprocess_emg_filtered(emg_filtered, postprocess_emg_name):
        if postprocess_emg_name == 'max':
            return emg_filtered
        if postprocess_emg_name == 'min':
            return np.multiply(emg_filtered, -1.0)
        if postprocess_emg_name == 'max+min':
            return np.abs(emg_filtered)
        if postprocess_emg_name == 'gradient':
            return np.gradient(emg_filtered)


class LcaData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.name = file_name.split('/')[-1].replace('.txt', '')

        self.time_data = None

        self.hammer_raw = None
        # self.hammer_filtered = None
        self.peak_times_hammer = None
        self.peak_values_hammer = None

        self.emg_raw = None
        self.emg_preprocessed = None
        self.emg_filtered = None
        self.peak_times_emg = None
        self.peak_values_emg = None

        self.time_data, self.hammer_raw, self.emg_raw = LcaFileUtils.read(file_name)

        self.emg_preprocessed = self.emg_raw

        # self.set_hammer_filter(LcaSignalUtils.DEFAULT_HAMMER_FILTER_NAME)
        self.peak_times_hammer, self.peak_values_hammer = LcaSignalUtils.find_hammer_peaks(
            self.time_data, self.hammer_raw, None)

        self.emg_filter_name = LcaSignalUtils.DEFAULT_EMG_FILTER_NAME
        self.latency_postprocess_name = LcaSignalUtils.DEFAULT_LATENCY_POSTPROCESS_NAME
        self.set_emg_filter(self.emg_filter_name)

    # def set_hammer_filter(self, hammer_filter_name):
    #     self.hammer_filtered = LcaSignalUtils.filter_hammer(self.hammer_raw, hammer_filter_name)
    #     self.peak_times_hammer, self.peak_values_hammer = LcaSignalUtils.find_hammer_peaks(
    #         self.time_data, self.hammer_filtered, None)

    def set_emg_filter(self, emg_filter_name):
        self.emg_filter_name = emg_filter_name
        self.emg_filtered = LcaSignalUtils.filter(self.emg_preprocessed, emg_filter_name)

        self.set_latency_postprocess(self.latency_postprocess_name)

    def set_latency_postprocess(self, latency_postprocess_name):
        self.latency_postprocess_name = latency_postprocess_name
        self.peak_times_emg, _, self.peak_values_emg \
            = LcaSignalUtils.find_emg_peaks(self.time_data, self.emg_filtered, self.latency_postprocess_name, None)

    def reset_emg_preprocessed(self):
        self.emg_preprocessed = self.emg_raw
        self.set_emg_filter(self.emg_filter_name)

    def apply_current_emg_filter(self):
        self.emg_preprocessed = self.emg_filtered
        self.set_emg_filter('-')

    def calc_stats_text(self, xlim):
        peak_times_hammer, peak_values_hammer = LcaSignalUtils.find_hammer_peaks(
            self.time_data, self.hammer_raw, xlim)
        peak_times_emg, peak_values_emg_post, peak_values_emg_src = LcaSignalUtils.find_emg_peaks(
            self.time_data, self.emg_filtered, self.latency_postprocess_name, xlim)

        def calc_latency_row(peak_time_hammer, peak_value_hammer):
            latency_row = (peak_time_hammer, peak_value_hammer, None, None, None)
            max_peak_values_emg_post = -10000
            for i, peak_time_emg in enumerate(peak_times_emg):
                # only in range 0 to 500 ms
                if peak_time_hammer < peak_time_emg and peak_time_emg <= peak_time_hammer + 0.5:
                    if peak_values_emg_post[i] > max_peak_values_emg_post:
                        latency_row = (peak_time_hammer,
                                       peak_value_hammer,
                                       peak_time_emg,
                                       peak_values_emg_src[i],
                                       round((peak_time_emg - peak_time_hammer) * 1000, 0))
                        max_peak_values_emg_post = peak_values_emg_post[i]
            return latency_row

        latency_table = [calc_latency_row(peak_time_hammer, peak_values_hammer[i])
                         for i, peak_time_hammer in enumerate(peak_times_hammer)]

        stats_text = tabulate({
            "Hammer peaks (s)": list(map(lambda row: row[0], latency_table)),
            "Value (V)": list(map(lambda row: row[1], latency_table)),
            "EMG peaks (s)": list(map(lambda row: row[2], latency_table)),
            "Value (mV)": list(map(lambda row: row[3], latency_table)),
            "Latency (ms)": list(map(lambda row: row[4], latency_table))
        }, headers='keys', tablefmt='pretty', showindex=False)

        valid_latencies = list(
            filter(lambda latency: latency is not None, list(map(lambda latency_row: latency_row[4], latency_table))))
        mean_latency = np.mean(valid_latencies)
        if not np.isnan(mean_latency):
            stats_text += f"\nMean latency: {mean_latency:.1f} ms"

        return stats_text


class LcaPlot:
    def __init__(self, lca_data):
        self.lca_data = lca_data

        # self.figure, self.base_axes = plt.subplots(figsize=(10, 3.7))
        # self.hammer_axes = self.base_axes.twinx()
        # self.emg_axes = self.hammer_axes.twinx()

        self.figure, self.hammer_axes = plt.subplots(figsize=(10, 3.7))
        self.emg_axes = self.hammer_axes.twinx()

        self.hammer_axes.set_title(lca_data.name)
        self.hammer_axes.grid(True, linestyle='--', alpha=0.7)
        self.hammer_axes.set_xlabel("Time(s)")
        self.hammer_axes.set_ylabel("Hammer signal (V)", color='red')
        self.hammer_axes.tick_params(axis='y', labelcolor='red')

        self.emg_axes.set_ylabel("EMG signal (mV)", color='blue')
        self.emg_axes.tick_params(axis='y', labelcolor='blue')

        self.hammer_raw_plot, = self.hammer_axes.plot(self.lca_data.time_data, self.lca_data.hammer_raw,
                                                      label="Hammer", color='red', linewidth=1, alpha=0.7)
        # self.hammer_filtered_plot, = self.hammer_axes.plot(self.lca_data.time_data, self.lca_data.hammer_filtered,
        #                                                    label="Hammer", color='red', linewidth=1, alpha=0.7)
        self.hammer_peaks_plot, = self.hammer_axes.plot(self.lca_data.peak_times_hammer,
                                                        self.lca_data.peak_values_hammer,
                                                        'ro', markerfacecolor='none', markeredgecolor='red',
                                                        label="Hammer peaks", alpha=0.7)

        self.emg_raw_plot, = self.emg_axes.plot(self.lca_data.time_data, self.lca_data.emg_raw,
                                                label="EMG raw", color='blue', linewidth=1,
                                                linestyle='dotted', alpha=0.3)
        self.emg_preprocessed_plot, = self.emg_axes.plot(self.lca_data.time_data, self.lca_data.emg_preprocessed,
                                                         label="EMG applied filters", color='blue', linewidth=1,
                                                         linestyle='dashed', alpha=0.3)
        self.emg_filtered_plot, = self.emg_axes.plot(self.lca_data.time_data, self.lca_data.emg_filtered,
                                                     label="EMG filtered", color='blue', linewidth=1.2, alpha=0.8)
        self.emg_peak_plot, = self.emg_axes.plot(self.lca_data.peak_times_emg, self.lca_data.peak_values_emg,
                                                 'bo', markerfacecolor='none', markeredgecolor='blue',
                                                 label="EMG peaks", alpha=0.7)

        self.initial_xlim = self.emg_axes.get_xlim()
        ylim_hammer = self.hammer_axes.get_ylim()
        ylim_emg = self.emg_axes.get_ylim()
        self.initial_ylim = (min(ylim_hammer[0] / 10, ylim_emg[0]), max(ylim_hammer[1] / 10, ylim_emg[1]))

        # self.time_range = matplotlib.widgets.RangeSlider(self.base_axes, "Selection", self.initial_xlim[0], self.initial_xlim[1])
        # # self.time_range.set_min(self.initial_xlim[0])
        # # self.time_range.set_max(self.initial_xlim[1])
        # self.time_range.set_val((self.initial_xlim[0], self.initial_xlim[1]))

        self.reset_zoom()

        # self.freq_axes= self.figure.add_axes([0.20, 0.1, 0.60, 0.03])

    def reset_zoom(self):
        # self.base_axes.set_xlim(self.initial_xlim)
        self.hammer_axes.set_xlim(self.initial_xlim)
        self.emg_axes.set_xlim(self.initial_xlim)

        # self.base_axes.set_ylim(self.initial_ylim[0], self.initial_ylim[1])
        self.hammer_axes.set_ylim(self.initial_ylim[0] * 10, self.initial_ylim[1] * 10)
        self.emg_axes.set_ylim(self.initial_ylim)

    def set_emg_filter(self, emg_filter_name):
        self.lca_data.set_emg_filter(emg_filter_name)

        self.emg_filtered_plot.set_ydata(self.lca_data.emg_filtered)
        self.emg_peak_plot.set_xdata(self.lca_data.peak_times_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_values_emg)

    def set_latency_postprocess(self, latency_postprocess_name):
        self.lca_data.set_latency_postprocess(latency_postprocess_name)

        self.emg_peak_plot.set_xdata(self.lca_data.peak_times_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_values_emg)

    def reset_emg_preprocessed(self):
        self.lca_data.reset_emg_preprocessed()

        self.emg_preprocessed_plot.set_ydata(self.lca_data.emg_preprocessed)
        self.emg_filtered_plot.set_ydata(self.lca_data.emg_filtered)
        self.emg_peak_plot.set_xdata(self.lca_data.peak_times_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_values_emg)

    def apply_current_emg_filter(self):
        self.lca_data.apply_current_emg_filter()

        self.emg_preprocessed_plot.set_ydata(self.lca_data.emg_preprocessed)
        self.emg_filtered_plot.set_ydata(self.lca_data.emg_filtered)
        self.emg_peak_plot.set_xdata(self.lca_data.peak_times_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_values_emg)


class LcaPlotWindow:

    def __init__(self, lca_data):
        self.lca_data = lca_data
        self.lca_plot = LcaPlot(lca_data)

        # Cannot create variables before root_window is created
        self.root_window = tk.Tk()
        self.canvas = None

        self.show_hammer_raw_plot = tk.IntVar(master=self.root_window, value=1)
        # self.show_hammer_filtered_plot = tk.IntVar(master=self.root_window, value=0)
        self.show_hammer_peaks_plot = tk.IntVar(master=self.root_window, value=1)
        self.show_emg_raw_plot = tk.IntVar(master=self.root_window, value=1)
        self.show_emg_preprocessed_plot = tk.IntVar(master=self.root_window, value=1)
        self.show_emg_filtered_plot = tk.IntVar(master=self.root_window, value=1)
        self.show_emg_peaks_plot = tk.IntVar(master=self.root_window, value=1)

        self.emg_filter_name = tk.StringVar(master=self.root_window, value=LcaSignalUtils.DEFAULT_EMG_FILTER_NAME)

        self.latency_postprocess_name = tk.StringVar(master=self.root_window,
                                                     value=LcaSignalUtils.DEFAULT_LATENCY_POSTPROCESS_NAME)

        self.create_root_window(self.root_window)

        self.refresh_plot()

    def create_root_window(self, root_window):
        root_window.title("EMG latency calculation app")
        root_window.iconbitmap(r"lca.ico")

        menubar = self.create_menubar(root_window)
        root_window.config(menu=menubar)

        # frame = tk.Frame(window)
        # frame.pack()
        frame = tk.Frame(root_window)
        frame.pack(fill=tk.BOTH, expand=True)

        # scrollbar_x = tk.Scrollbar(frame, orient="horizontal", command=self.lca_plot.hammer_axes.set_xlim)
        # scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        #
        # scrollbar_y = tk.Scrollbar(frame, orient="vertical", command=self.lca_plot.hammer_axes.set_ylim)
        # scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        # bottom_frame = tk.Frame(window)
        # bottom_frame.pack(side=tk.BOTTOM)

        self.canvas = self.create_canvas(frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = self.create_toolbar(self.canvas, frame)

        return root_window

    def create_menubar(self, parent):
        menubar = tk.Menu(parent)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = self.create_view_menu(menubar)
        menubar.add_cascade(label="View", menu=view_menu)

        emg_preprocess_menu = self.create_emg_preprocess_menu(menubar)
        menubar.add_cascade(label="User-applied filters", menu=emg_preprocess_menu)

        emg_filter_menu = self.create_emg_filter_menu(menubar)
        menubar.add_cascade(label="Current filter", menu=emg_filter_menu)

        latency_menu = self.create_latency_menu(menubar)
        menubar.add_cascade(label="Latency", menu=latency_menu)

        return menubar

    def create_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=False)

        file_menu.add_command(label="Open...", command=self.file_open_onclick)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.file_close_onclick)

        return file_menu

    def create_view_menu(self, menubar):
        view_menu = tk.Menu(menubar, tearoff=False)

        # self.view_menu.add_checkbutton(label="Hammer (raw)", variable=self.show_hammer_raw_plot,
        #                                command=self.show_any_plot_onchanged)
        # self.view_menu.add_checkbutton(label="Hammer (filtered)", variable=self.show_hammer_filtered_plot,
        #                                command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="Hammer", variable=self.show_hammer_raw_plot,
                                  command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="Hammer peaks", variable=self.show_hammer_peaks_plot,
                                  command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="EMG (raw)", variable=self.show_emg_raw_plot,
                                  command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="EMG (signal)", variable=self.show_emg_preprocessed_plot,
                                  command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="EMG (filtered)", variable=self.show_emg_filtered_plot,
                                  command=self.show_any_plot_onchanged)
        view_menu.add_checkbutton(label="EMG peaks", variable=self.show_emg_peaks_plot,
                                  command=self.show_any_plot_onchanged)

        view_menu.add_separator()

        view_menu.add_command(label="Reset zoom", command=self.view_reset_zoom_onclick)

        return view_menu

    def create_emg_filter_menu(self, menubar):
        emg_filter_menu = tk.Menu(menubar, tearoff=False)

        emg_filter_menu.add_radiobutton(label="No filter", variable=self.emg_filter_name,
                                        value='-', command=self.emg_filter_name_onchanged)
        # emg_filter_menuemg_filter_menu.add_radiobutton(label="Default filter (rolling RMS)", variable=self.emg_filter_name,
        #                             value='default', command=self.emg_filter_name_onchanged)

        emg_filter_menu.add_separator()

        emg_filter_menu.add_radiobutton(label="Butterworth low-pass + filtfilt filter", variable=self.emg_filter_name,
                                        value='butter_lowpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Butterworth high-pass + filtfilt filter", variable=self.emg_filter_name,
                                        value='butter_highpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Butterworth band-pass + filtfilt filter", variable=self.emg_filter_name,
                                        value='butter_bandpass_filtfilt', command=self.emg_filter_name_onchanged)

        emg_filter_menu.add_radiobutton(label="Butterworth low-pass (causal) filter", variable=self.emg_filter_name,
                                        value='butter_lowpass_ifilter', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Chebyshev (type2) low-pass + filtfilt filter",
                                        variable=self.emg_filter_name,
                                        value='cheby2_lowpass_filtfilt', command=self.emg_filter_name_onchanged)

        emg_filter_menu.add_radiobutton(label="Notch + filtfilt filter", variable=self.emg_filter_name,
                                        value='notch_filtfilt', command=self.emg_filter_name_onchanged)

        # emg_filter_menu.add_radiobutton(label="Rolling RMS filter", variable=self.emg_filter_name,
        #                                 value='rolling_rms', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Savitzky–Golay filter", variable=self.emg_filter_name,
                                        value='savgol', command=self.emg_filter_name_onchanged)

        # emg_filter_menu.add_radiobutton(label="Conventional EMG processing", variable=self.emg_filter_name,
        #                                 value='conventional', command=self.emg_filter_name_onchanged)
        # emg_filter_menu.add_radiobutton(label="pyemgpipeline EMG processing", variable=self.emg_filter_name,
        #                             value='pyemgpipeline', command=self.emg_filter_name_onchanged)

        # emg_filter_menu.add_separator()
        #
        # emg_filter_menu.add_radiobutton(label="Gradient", variable=self.emg_filter_name,
        #                             value='gradient', command=self.emg_filter_name_onchanged)

        return emg_filter_menu

    def create_emg_preprocess_menu(self, menubar):
        emg_preprocess_menu = tk.Menu(menubar, tearoff=False)

        emg_preprocess_menu.add_command(label="Apply current filter",
                                        command=self.emg_preprocess_apply_current_filter_onclick)

        emg_preprocess_menu.add_command(label="Reset to raw signal", command=self.emg_preprocess_reset_onclick)

        return emg_preprocess_menu

    def create_latency_menu(self, menubar):
        latency_menu = tk.Menu(menubar, tearoff=False)

        latency_menu.add_radiobutton(label="Use only maximums", variable=self.latency_postprocess_name,
                                     value='max', command=self.latency_postprocess_name_onchanged)
        latency_menu.add_radiobutton(label="Use only minimums", variable=self.latency_postprocess_name,
                                     value='min', command=self.latency_postprocess_name_onchanged)
        latency_menu.add_radiobutton(label="Use maximums & minimums", variable=self.latency_postprocess_name,
                                     value='max+min', command=self.latency_postprocess_name_onchanged)
        latency_menu.add_radiobutton(label="Use gradient", variable=self.latency_postprocess_name,
                                     value='gradient', command=self.latency_postprocess_name_onchanged)

        latency_menu.add_separator()

        latency_menu.add_command(label="Show statistics", command=self.latency_show_stats_onclick)

        return latency_menu

    def create_canvas(self, parent):
        canvas = FigureCanvasTkAgg(self.lca_plot.figure, master=parent)
        canvas.mpl_connect('scroll_event', self.canvas_onmousescroll)

        return canvas

    def create_toolbar(self, canvas, parent):
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        return toolbar

    def run(self):
        self.root_window.mainloop()

    def file_open_onclick(self):
        file_name = LcaFileUtils.ask_open_txt_file_name()
        lca_data = LcaData(file_name)
        lca_plot_window = LcaPlotWindow(lca_data)
        lca_plot_window.run()

    def file_close_onclick(self):
        self.root_window.destroy()

    def show_any_plot_onchanged(self):
        self.refresh_plot()

    def canvas_onmousescroll(self, event):
        # get current x and y limits
        xlim = self.lca_plot.emg_axes.get_xlim()
        ylim = self.lca_plot.emg_axes.get_ylim()

        event_key = event.key or ''

        if event_key.find('alt') >= 0:
            # pan
            if event_key.find('shift') >= 0:
                # vertical panning
                x_pan_factor = 0
                y_pan_factor = 0.1
            else:
                # horizontal panning
                x_pan_factor = 0.1
                y_pan_factor = 0

            if event_key.find('control') >= 0:
                # fast horizontal panning
                x_pan_factor *= 5
                y_pan_factor *= 5

            if event.button == 'up':
                x_pan = -x_pan_factor * (xlim[1] - xlim[0])
                y_pan = -y_pan_factor * (ylim[1] - ylim[0])
            else:
                x_pan = x_pan_factor * (xlim[1] - xlim[0])
                y_pan = y_pan_factor * (ylim[1] - ylim[0])

            new_xlim = (xlim[0] + x_pan, xlim[1] + x_pan)
            new_ylim = (ylim[0] + y_pan, ylim[1] + y_pan)

        else:
            # zoom
            if event_key.find('control') >= 0:
                # full zoom
                x_scale_factor = 1.5
                y_scale_factor = 1.5
            elif event_key.find('shift') >= 0:
                # vertical zoom
                x_scale_factor = 1.0
                y_scale_factor = 1.5
            else:
                # horizontal zoom
                x_scale_factor = 1.5
                y_scale_factor = 1.0

            if event.button == 'up':
                x_scale = 1 / x_scale_factor
                y_scale = 1 / y_scale_factor
            else:
                x_scale = x_scale_factor
                y_scale = y_scale_factor

            new_xlim = ([event.xdata - (event.xdata - xlim[0]) * x_scale,
                         event.xdata + (xlim[1] - event.xdata) * x_scale])
            new_ylim = ([event.ydata - (event.ydata - ylim[0]) * y_scale,
                         event.ydata + (ylim[1] - event.ydata) * y_scale])

        # set new limits

        self.lca_plot.hammer_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.lca_plot.hammer_axes.set_ylim(new_ylim[0] * 10, new_ylim[1] * 10)

        self.lca_plot.emg_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.lca_plot.emg_axes.set_ylim(new_ylim[0], new_ylim[1])

        self.canvas.draw_idle()

    def refresh_plot(self):
        self.lca_plot.hammer_raw_plot.set_visible(self.show_hammer_raw_plot.get())
        # self.lca_plot.hammer_filtered_plot.set_visible(self.show_hammer_filtered_plot.get())
        self.lca_plot.hammer_peaks_plot.set_visible(self.show_hammer_peaks_plot.get())
        self.lca_plot.hammer_axes.legend()

        self.lca_plot.emg_raw_plot.set_visible(self.show_emg_raw_plot.get())
        self.lca_plot.emg_preprocessed_plot.set_visible(self.show_emg_preprocessed_plot.get())
        self.lca_plot.emg_filtered_plot.set_visible(self.show_emg_filtered_plot.get())
        self.lca_plot.emg_peak_plot.set_visible(self.show_emg_peaks_plot.get())
        self.lca_plot.emg_axes.legend()

        self.canvas.draw_idle()

    def view_reset_zoom_onclick(self):
        self.lca_plot.reset_zoom()
        self.canvas.draw_idle()

    def emg_filter_name_onchanged(self):
        self.lca_plot.set_emg_filter(self.emg_filter_name.get())
        self.canvas.draw_idle()

    def emg_preprocess_reset_onclick(self):
        self.lca_plot.reset_emg_preprocessed()
        self.canvas.draw_idle()

    def emg_preprocess_apply_current_filter_onclick(self):
        self.lca_plot.apply_current_emg_filter()
        self.emg_filter_name.set('-')
        self.canvas.draw_idle()

    def latency_postprocess_name_onchanged(self):
        self.lca_plot.set_latency_postprocess(self.latency_postprocess_name.get())
        self.canvas.draw_idle()

    def latency_show_stats_onclick(self):
        xlim = self.lca_plot.emg_axes.get_xlim()
        stats_text = self.lca_data.calc_stats_text(xlim)
        lca_stats_window = LcaStatsWindow(stats_text)
        lca_stats_window.run()


class LcaStatsWindow:
    def __init__(self, stats_text):
        self.stats_text = stats_text

        self.root_window = tk.Tk()

        self.create_root_window(self.root_window)

    def create_root_window(self, root_window):
        root_window.title("EMG latency statistics")
        root_window.iconbitmap(r"lca.ico")

        menubar = self.create_menubar(root_window)
        root_window.config(menu=menubar)

        frame = tk.Frame(root_window)
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(frame, font=('Courier', 10), wrap=tk.NONE)
        text.insert(tk.END, self.stats_text)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

    def create_menubar(self, parent):
        menubar = tk.Menu(parent)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="File", menu=file_menu)

        return menubar

    def create_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=False)

        file_menu.add_command(label="Save as...", command=self.file_saveas_onclick)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.file_close_onclick)

        return file_menu

    def file_saveas_onclick(self):
        file_name = LcaFileUtils.ask_save_txt_file_name()
        if file_name:
            with open(file_name, "w") as file:
                file.write(self.stats_text)

    def file_close_onclick(self):
        self.root_window.destroy()

    def run(self):
        self.root_window.mainloop()


def main() -> int:
    file_name = LcaFileUtils.ask_open_txt_file_name()
    if file_name == '':
        return 1
    else:
        lca_data = LcaData(file_name)
        lca_plot_window = LcaPlotWindow(lca_data)
        lca_plot_window.run()
        return 0


if __name__ == '__main__':
    sys.exit(main())

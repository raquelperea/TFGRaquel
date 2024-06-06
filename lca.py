import sys
import os.path

import numpy as np
# import scipy as sp
import scipy.signal as signal

import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from tabulate import tabulate

try:
    import pyi_splash
except:
    pass


# import pyemgpipeline
# import pandas as pd
# from PIL import Image, ImageTk

class FileUtils:
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

    @staticmethod
    def resolve_icon_file_name(ico_file_name):
        # https://stackoverflow.com/questions/9929479/embed-icon-in-python-script
        try:
            if not hasattr(sys, "frozen"):
                ico_file_name = os.path.join(os.path.dirname(__file__), ico_file_name)
            else:
                ico_file_name = os.path.join(sys.prefix, ico_file_name)
        except:
            pass
        return ico_file_name


# static methods for signal handling
class SignalUtils:
    @staticmethod
    def cut_by_irange(data, irange):
        if irange is None:
            return data
        else:
            return np.array(data)[irange]

    @staticmethod
    def get_irange_from_xlim(xdata, xlim):
        # Find indices in range
        if xlim is None:
            return None
        else:
            return np.where((xdata >= xlim[0]) & (xdata <= xlim[1]))[0]

    @staticmethod
    def find_peaks(xdata, ydata, ydata_src, height):
        # Find only peaks with a minimum height
        ipeaks, _ = signal.find_peaks(ydata, height=height)

        xpeaks = [round(xdata[i], 3) for i in ipeaks]
        ypeaks = [round(ydata[i], 3) for i in ipeaks]
        ypeaks_src = [round(ydata_src[i], 3) for i in ipeaks]
        return xpeaks, ypeaks, ypeaks_src

    @staticmethod
    def rolling_rms_filter(signal, window_size):
        if window_size % 2 == 0:
            window_size += 1  # window_size must be odd

        window = np.ones(window_size) / float(window_size)

        return np.sqrt(
            signal.fftconvolve(
                np.power(signal, 2),
                window, mode='same'))


class PlotUtils:
    @staticmethod
    def apply_scroll_event(event, xlim, ylim):
        try:
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
                return new_xlim, new_ylim

        except:
            return xlim, ylim


class LcaData:
    def __init__(self, file_name):
        # Filter configuration

        self.sampling_freq = 1000  # Hz
        self.nyquist_freq = self.sampling_freq / 2

        self.highpass_cutoff_freq = 30  # Hz
        self.lowpass_cutoff_freq = 60  # Hz
        self.bandpass_cutoff_freqs = (30, 60)  # Hz

        self.butter_order = 2

        self.cheby2_order = 20
        self.cheby2_stop_attenuation = 40  # dB

        self.notch_freq = 50.0  # Frequency to be removed (notch frequency)
        # self.notch_bandwidth = 2.0  # Bandwidth of the notch in Hz
        self.notch_quality = 30.0

        self.savgol_windowsize = 11
        self.savgol_order = 3

        self.rolling_windowsize = 5

        # self.notch_attenuation = 20  # Desired attenuation at 50 Hz in dB

        ###

        self.emg_filter_name = '-'  # no filter

        self.emg_postprocess_name = 'max'

        ###

        self.file_name = file_name
        self.name = file_name.split('/')[-1].replace('.txt', '')

        self.time_data = None

        self.hammer_raw = None
        # self.hammer_filtered = None

        self.hammer_peaks_height = 0.1
        self.peak_times_hammer = None
        self.peak_values_hammer = None

        self.emg_raw = None
        self.emg_filtered = None
        self.emg_postprocessed = None

        self.emg_peaks_height = 0.01
        self.peak_times_emg = None
        self.peak_values_emg = None

        self.time_data, self.hammer_raw, self.emg_raw = FileUtils.read(file_name)

        self.refresh_hammer_peaks()

        self.refresh_emg_filtered()
        self.refresh_emg_peaks()

    def refresh_hammer_peaks(self):
        self.peak_times_hammer, self.peak_values_hammer, _ = \
            SignalUtils.find_peaks(self.time_data, self.hammer_raw, self.hammer_raw, self.hammer_peaks_height)

    def set_emg_filter(self, emg_filter_name):
        self.emg_filter_name = emg_filter_name

        self.refresh_emg_filtered()
        self.refresh_emg_peaks()

    def set_emg_postprocess(self, emg_postprocess_name):
        self.emg_postprocess_name = emg_postprocess_name
        self.refresh_emg_peaks()

    def calc_stats_text(self, xlim):
        irange = SignalUtils.get_irange_from_xlim(self.time_data, xlim)
        time_data = SignalUtils.cut_by_irange(self.time_data, irange)
        hammer_raw = SignalUtils.cut_by_irange(self.hammer_raw, irange)
        emg_filtered = SignalUtils.cut_by_irange(self.emg_filtered, irange)
        emg_postprocessed = SignalUtils.cut_by_irange(self.emg_postprocessed, irange)

        peak_times_hammer, _, peak_values_hammer = \
            SignalUtils.find_peaks(time_data, hammer_raw, hammer_raw, self.hammer_peaks_height)

        peak_times_emg, peak_values_emg_post, peak_values_emg_src = \
            SignalUtils.find_peaks(time_data, emg_postprocessed, emg_filtered,
                                   self.emg_peaks_height)

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

    def normalize_freq(self, freq):
        return freq / self.nyquist_freq

    def normalize_freqs(self, freqs):
        return (self.normalize_freq(freqs[0]) / self.normalize_freq(freqs[1]))

    def refresh_emg_filtered(self):
        if self.emg_filter_name == '-':
            self.emg_filtered = self.emg_raw
        elif self.emg_filter_name == 'savgol':
            self.emg_filtered = signal.savgol_filter(self.emg_raw,
                                                     window_length=self.savgol_windowsize,
                                                     polyorder=self.savgol_order)
        elif self.emg_filter_name == 'rolling_rms':
            self.emg_filtered = SignalUtils.rolling_rms_filter(self.emg_raw,
                                                               self.rolling_windowsize)
        elif self.emg_filter_name == 'butter_lowpass_filtfilt':
            b, a = signal.butter(self.butter_order,
                                 self.normalize_freq(self.lowpass_cutoff_freq),
                                 btype='lowpass', analog=False, output='ba')
            self.emg_filtered = signal.filtfilt(b, a, self.emg_raw)
        elif self.emg_filter_name == 'butter_highpass_filtfilt':
            b, a = signal.butter(self.butter_order,
                                 self.normalize_freq(self.highpass_cutoff_freq),
                                 btype='highpass', analog=False, output='ba')
            self.emg_filtered = signal.filtfilt(b, a, self.emg_raw)
        elif self.emg_filter_name == 'butter_bandpass_filtfilt':
            b, a = signal.butter(self.butter_order,
                                 self.normalize_freqs(self.bandpass_cutoff_freqs),
                                 btype='bandpass', analog=False, output='ba')
            self.emg_filtered = signal.filtfilt(b, a, self.emg_raw)
        elif self.emg_filter_name == 'butter_lowpass_lfilter':
            b, a = signal.butter(self.butter_order,
                                 self.normalize_freq(self.lowpass_cutoff_freq),
                                 btype='lowpass', analog=False, output='ba')
            self.emg_filtered = signal.lfilter(b, a, self.emg_raw)
        elif self.emg_filter_name == 'cheby2_lowpass_filtfilt':
            b, a = signal.cheby2(self.cheby2_order, self.cheby2_stop_attenuation,
                                 self.normalize_freq(self.lowpass_cutoff_freq),
                                 btype='lowpass', analog=False, output='ba')
            self.emg_filtered = signal.filtfilt(b, a, self.emg_raw)
        elif self.emg_filter_name == 'notch_filtfilt':
            b, a = signal.iirnotch(self.normalize_freq(self.notch_freq), self.notch_quality)
            self.emg_filtered = signal.filtfilt(b, a, self.emg_raw)
        elif self.emg_filter_name == 'notch_lfilter':
            b, a = signal.iirnotch(self.normalize_freq(self.notch_freq), self.notch_quality)
            self.emg_filtered = signal.lfilter(b, a, self.emg_raw)
        else:
            raise "Unknown filter name"

    def refresh_emg_peaks(self):
        if self.emg_postprocess_name == 'max':
            self.emg_postprocessed = self.emg_filtered
        elif self.emg_postprocess_name == 'min':
            self.emg_postprocessed = np.multiply(self.emg_filtered, -1.0)
        elif self.emg_postprocess_name == 'max+min':
            self.emg_postprocessed = np.abs(self.emg_filtered)
        elif self.emg_postprocess_name == 'gradient':
            self.emg_postprocessed = np.gradient(self.emg_filtered)
        else:
            raise "Unknown postprocess name"

        if self.emg_postprocess_name == 'gradient':
            self.emg_peaks_height = 0.005
        else:
            self.emg_peaks_height = 0.06

        self.peak_times_emg, _, self.peak_values_emg \
            = SignalUtils.find_peaks(self.time_data, self.emg_postprocessed, self.emg_filtered, self.emg_peaks_height)


class LcaPlot:
    def __init__(self, lca_data):
        self.lca_data = lca_data

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
                                                label="EMG raw", color='blue',
                                                linewidth=1, linestyle='dashed', alpha=0.3)
        self.emg_filtered_plot, = self.emg_axes.plot(self.lca_data.time_data, self.lca_data.emg_filtered,
                                                     label="EMG filtered", color='blue',
                                                     linewidth=1, alpha=0.8)
        self.emg_postprocessed_plot, = self.emg_axes.plot(self.lca_data.time_data, self.lca_data.emg_postprocessed,
                                                          label="EMG gradient", color='blue',
                                                          linestyle='dotted', linewidth=0.8, alpha=0.5)
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
        self.emg_postprocessed_plot.set_ydata(self.lca_data.emg_postprocessed)
        self.emg_peak_plot.set_xdata(self.lca_data.peak_times_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_values_emg)

    def set_emg_postprocess(self, emg_postprocess_name):
        self.lca_data.set_emg_postprocess(emg_postprocess_name)

        self.emg_postprocessed_plot.set_ydata(self.lca_data.emg_postprocessed)
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
        self.show_emg_filtered_plot = tk.IntVar(master=self.root_window, value=1)
        self.show_emg_peaks_plot = tk.IntVar(master=self.root_window, value=1)

        self.emg_filter_name = tk.StringVar(master=self.root_window, value=lca_data.emg_filter_name)

        self.emg_postprocess_name = tk.StringVar(master=self.root_window, value=lca_data.emg_postprocess_name)

        self.create_root_window(self.root_window)

        self.refresh_plot()

    def create_root_window(self, root_window):
        root_window.title("EMG latency calculation app")
        root_window.iconbitmap(FileUtils.resolve_icon_file_name("lca.ico"))

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

        emg_filter_menu = self.create_emg_filter_menu(menubar)
        menubar.add_cascade(label="Filter", menu=emg_filter_menu)

        analyze_menu = self.create_analyze_menu(menubar)
        menubar.add_cascade(label="Analyze", menu=analyze_menu)

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
        # emg_filter_menuemg_filter_menu.add_radiobutton(label="Default (rolling RMS)", variable=self.emg_filter_name,
        #                             value='default', command=self.emg_filter_name_onchanged)

        emg_filter_menu.add_separator()

        emg_filter_menu.add_radiobutton(label="Butterworth low-pass + filtfilt", variable=self.emg_filter_name,
                                        value='butter_lowpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Butterworth high-pass + filtfilt", variable=self.emg_filter_name,
                                        value='butter_highpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Butterworth band-pass + filtfilt", variable=self.emg_filter_name,
                                        value='butter_bandpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Butterworth low-pass + lfilter (causal)", variable=self.emg_filter_name,
                                        value='butter_lowpass_lfilter', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Chebyshev (type2) low-pass + filtfilt", variable=self.emg_filter_name,
                                        value='cheby2_lowpass_filtfilt', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Notch + filtfilt", variable=self.emg_filter_name,
                                        value='notch_filtfilt', command=self.emg_filter_name_onchanged)
        # emg_filter_menu.add_radiobutton(label="Notch + lfilter", variable=self.emg_filter_name,
        #                                 value='notch_lfilter', command=self.emg_filter_name_onchanged)
        # emg_filter_menu.add_radiobutton(label="Rolling RMS", variable=self.emg_filter_name,
        #                                 value='rolling_rms', command=self.emg_filter_name_onchanged)
        emg_filter_menu.add_radiobutton(label="Savitzky–Golay", variable=self.emg_filter_name,
                                        value='savgol', command=self.emg_filter_name_onchanged)

        return emg_filter_menu

    def create_analyze_menu(self, menubar):
        analyze_menu = tk.Menu(menubar, tearoff=False)

        analyze_menu.add_command(label="Show frequency spectrum", command=self.analyze_show_fft_onclick)

        analyze_menu.add_separator()

        analyze_menu.add_radiobutton(label="Use only maximums", variable=self.emg_postprocess_name,
                                     value='max', command=self.emg_postprocess_name_onchanged)
        analyze_menu.add_radiobutton(label="Use only minimums", variable=self.emg_postprocess_name,
                                     value='min', command=self.emg_postprocess_name_onchanged)
        analyze_menu.add_radiobutton(label="Use maximums & minimums", variable=self.emg_postprocess_name,
                                     value='max+min', command=self.emg_postprocess_name_onchanged)
        analyze_menu.add_radiobutton(label="Use gradient", variable=self.emg_postprocess_name,
                                     value='gradient', command=self.emg_postprocess_name_onchanged)

        analyze_menu.add_separator()

        analyze_menu.add_command(label="Show latency statistics", command=self.analyze_show_stats_onclick)

        return analyze_menu

    def create_canvas(self, parent):
        canvas = FigureCanvasTkAgg(self.lca_plot.figure, master=parent)
        canvas.mpl_connect('scroll_event', self.canvas_onmousescroll)

        return canvas

    def create_toolbar(self, canvas, parent):
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        return toolbar

    def file_open_onclick(self):
        file_name = FileUtils.ask_open_txt_file_name()
        lca_data = LcaData(file_name)
        lca_plot_window = LcaPlotWindow(lca_data)

    def file_close_onclick(self):
        self.root_window.destroy()
        self.root_window.quit()

    def show_any_plot_onchanged(self):
        self.refresh_plot()

    def canvas_onmousescroll(self, event):
        xlim = self.lca_plot.emg_axes.get_xlim()
        ylim = self.lca_plot.emg_axes.get_ylim()
        new_xlim, new_ylim = PlotUtils.apply_scroll_event(event, xlim, ylim)

        self.lca_plot.hammer_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.lca_plot.hammer_axes.set_ylim(new_ylim[0] * 10, new_ylim[1] * 10)

        self.lca_plot.emg_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.lca_plot.emg_axes.set_ylim(new_ylim[0], new_ylim[1])

        self.canvas.draw_idle()

    def refresh_plot(self):
        self.lca_plot.hammer_raw_plot.set_visible(self.show_hammer_raw_plot.get())
        # self.lca_plot.hammer_filtered_plot.set_visible(self.show_hammer_filtered_plot.get())
        self.lca_plot.hammer_peaks_plot.set_visible(self.show_hammer_peaks_plot.get())
        self.lca_plot.hammer_axes.legend(loc='upper left')

        self.lca_plot.emg_raw_plot.set_visible(self.show_emg_raw_plot.get())
        self.lca_plot.emg_filtered_plot.set_visible(self.show_emg_filtered_plot.get())
        self.lca_plot.emg_peak_plot.set_visible(self.show_emg_peaks_plot.get())
        show_emg_postprocessed_plot = self.show_emg_filtered_plot.get() and self.emg_postprocess_name.get() == 'gradient'
        if show_emg_postprocessed_plot:
            self.lca_plot.emg_postprocessed_plot.set_label('EMG gradient')
        else:
            self.lca_plot.emg_postprocessed_plot.set_label('_')
        self.lca_plot.emg_postprocessed_plot.set_visible(show_emg_postprocessed_plot)

        self.lca_plot.emg_axes.legend(loc='upper right')

        self.canvas.draw_idle()

    def view_reset_zoom_onclick(self):
        self.lca_plot.reset_zoom()
        self.canvas.draw_idle()

    def emg_filter_name_onchanged(self):
        self.lca_plot.set_emg_filter(self.emg_filter_name.get())
        self.canvas.draw_idle()

    def emg_postprocess_name_onchanged(self):
        self.lca_plot.set_emg_postprocess(self.emg_postprocess_name.get())
        self.refresh_plot()

    def analyze_show_fft_onclick(self):
        fft_window = FftWindow(self.root_window, self.lca_data, self.lca_plot.emg_axes.get_xlim())
        fft_window.root_window.focus()
        fft_window.root_window.grab_set()

    def analyze_show_stats_onclick(self):
        xlim = self.lca_plot.emg_axes.get_xlim()
        stats_text = self.lca_data.calc_stats_text(xlim)
        stats_window = StatsWindow(self.root_window, stats_text)
        stats_window.root_window.focus()
        stats_window.root_window.grab_set()


class StatsWindow:
    def __init__(self, parent, stats_text):
        self.stats_text = stats_text

        self.root_window = tk.Toplevel(parent)

        self.create_root_window(self.root_window)

    def create_root_window(self, root_window):
        root_window.title("EMG latency statistics")
        root_window.iconbitmap(FileUtils.resolve_icon_file_name("lca.ico"))

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
        file_name = FileUtils.ask_save_txt_file_name()
        if file_name:
            with open(file_name, "w") as file:
                file.write(self.stats_text)

    def file_close_onclick(self):
        self.root_window.destroy()


class FftPlot:
    def __init__(self, lca_data, xlim):
        self.lca_data = lca_data
        self.xlim = xlim

        self.figure, self.fft_axes = plt.subplots(figsize=(10, 3.7))

        self.fft_axes.set_title(lca_data.name)
        self.fft_axes.grid(True, linestyle='--', alpha=0.7)
        self.fft_axes.set_xlabel("Freq(Hz)")
        # scaling = 'spectrum'
        # self.fft_axes.set_ylabel("Power(mV²)", color='green')
        scaling = 'density'
        self.fft_axes.set_ylabel("Density(mV²/Hz)", color='green')
        self.fft_axes.tick_params(axis='y', labelcolor='green')

        irange = SignalUtils.get_irange_from_xlim(self.lca_data.time_data, xlim)
        emg_raw = SignalUtils.cut_by_irange(self.lca_data.emg_raw, irange)
        emg_filtered = SignalUtils.cut_by_irange(self.lca_data.emg_filtered, irange)

        fft_freq, fft_emg_raw = signal.welch(emg_raw, nperseg=1024,
                                             fs=lca_data.sampling_freq, window="hann", scaling=scaling)
        fft_freq, fft_emg_filtered = signal.welch(emg_filtered, nperseg=1024,
                                                  fs=lca_data.sampling_freq, window="hann", scaling=scaling)

        self.fft_raw_plot, = self.fft_axes.plot(fft_freq, fft_emg_raw,
                                                label="EMG raw", color='green',
                                                linewidth=1, linestyle='dashed', alpha=0.3)
        self.fft_filtered_plot, = self.fft_axes.plot(fft_freq, fft_emg_filtered,
                                                     label="EMG filtered", color='green',
                                                     linewidth=1, alpha=0.8)

        self.fft_axes.set_xlim(0, 200)
        # self.fft_axes.set_ylim(0, 0.01)

        self.fft_axes.legend(loc='upper right')


class FftWindow:
    def __init__(self, parent, lca_data, xlim):
        self.lca_data = lca_data
        self.xlim = xlim

        self.fft_plot = FftPlot(lca_data, xlim)

        self.root_window = tk.Toplevel(parent)
        self.canvas = None

        self.create_root_window(self.root_window)

    def create_root_window(self, root_window):
        root_window.title("EMG frequency analysis")
        root_window.iconbitmap(FileUtils.resolve_icon_file_name("lca.ico"))

        menubar = self.create_menubar(root_window)
        root_window.config(menu=menubar)

        frame = tk.Frame(root_window)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = self.create_canvas(frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        return root_window

    def create_menubar(self, parent):
        menubar = tk.Menu(parent)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="File", menu=file_menu)

        return menubar

    def create_canvas(self, parent):
        canvas = FigureCanvasTkAgg(self.fft_plot.figure, master=parent)
        canvas.mpl_connect('scroll_event', self.canvas_onmousescroll)

        return canvas

    def create_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=False)

        file_menu.add_command(label="Close", command=self.file_close_onclick)

        return file_menu

    def file_close_onclick(self):
        self.root_window.destroy()

    def canvas_onmousescroll(self, event):
        xlim = self.fft_plot.fft_axes.get_xlim()
        ylim = self.fft_plot.fft_axes.get_ylim()
        new_xlim, new_ylim = PlotUtils.apply_scroll_event(event, xlim, ylim)

        self.fft_plot.fft_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.fft_plot.fft_axes.set_ylim(new_ylim[0], new_ylim[1])

        self.canvas.draw_idle()


def main() -> int:
    # closes splash, if started using pyinstaller
    try:
        pyi_splash.close()
    except:
        pass

    # reads file_name from 1st arg, if existent
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = ''

    # otherwise, ask user
    if not os.path.exists(file_name):
        file_name = FileUtils.ask_open_txt_file_name()

    def root_window_onclose():
        lca_plot_window.root_window.quit()
        lca_plot_window.root_window.destroy()
        sys.exit()

    # open file_name
    if not os.path.exists(file_name):
        return 1
    else:
        lca_data = LcaData(file_name)
        lca_plot_window = LcaPlotWindow(lca_data)
        lca_plot_window.root_window.protocol("wm_delete_window", root_window_onclose)
        lca_plot_window.root_window.mainloop()
        return 0


if __name__ == '__main__':
    sys.exit(main())

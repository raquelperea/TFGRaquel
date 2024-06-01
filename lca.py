import sys
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tabulate import tabulate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

# from PIL import Image, ImageTk
import pyemgpipeline


class LcaFile:
    @staticmethod
    def ask_file_name():
        file_name = tk.filedialog.askopenfilename(defaultextension=".txt",
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        return file_name

    @staticmethod
    def read(file_name):
        time = []
        hammer_raw = []
        emg_raw = []
        reading = False
        with open(file_name, 'r') as file:
            for line in file:
                if reading:
                    try:
                        values = [float(value.replace(',', '.')) for value in line.split('\t')]
                        time.append(values[0])
                        hammer_raw.append(values[1])
                        emg_raw.append(values[2])
                    except ValueError:
                        break
                elif line.startswith('0\t'):
                    reading = True
        return time, hammer_raw, emg_raw


class LcaUtils:
    @staticmethod
    def find_peaks(xdata, ydata, xlim, ythreshold):
        # Find indices in range
        if xlim is None:
            irange = range(0, len(xdata))
        else:
            # irange = np.where((xdata >= xlim[0]) & (xdata <= xlim[1]))[0]
            irange = range(max(0, xlim[0]), min(len(xdata), xlim[1] + 1))

        # Find x's and y's in range
        xrange = np.array(xdata)[irange]
        yrange = np.array(ydata)[irange]

        # Find peaks above threshold
        ipeaks, _ = sp.find_peaks(yrange, threshold=ythreshold)  # ?? was 'height'

        xpeaks = [round(xrange[i], 3) for i in ipeaks]
        ypeaks = [round(yrange[i], 3) for i in ipeaks]
        return xpeaks, ypeaks

    @staticmethod
    def find_hammer_peaks(time, hammer_filtered, xlim):
        return LcaUtils.find_peaks(time, hammer_filtered, xlim, 0.1)

    @staticmethod
    def find_emg_peaks(time, emg_filtered, xlim):
        return LcaUtils.find_peaks(time, emg_filtered, xlim, 0.06)

    @staticmethod
    def filter_hammer(hammer):
        # threshold_hammer = 0.02  # Umbral para establecer valores casi cero a cero
        # hammer_filtered = [0 if abs(x) < threshold_hammer else x for x in hammer]
        offset = np.mean(hammer)  # Calculate the mean value of the signal
        offset_removed_signal = hammer - offset

        return offset_removed_signal

    @staticmethod
    def rolling_rms_filter(emg, half_window_size):
        window_size = 2 * half_window_size + 1
        window = np.ones(window_size) / float(window_size)

        return np.sqrt(
            sp.fftconvolve(
                np.power(emg, 2),
                window,
                'same'))

    @staticmethod
    def filter_emg_rolling_rms(emg):
        # return LcaUtils.rolling_rms_filter(emg, 10)
        return LcaUtils.rolling_rms_filter(emg, 3)
        # return LcaUtils.rolling_rms_filter(rolling_rms_filter(emg, 3), 1)
        # return LcaUtils.rolling_rms_filter(emg, 2)

    @staticmethod
    def filter_emg_gradient(emg):
        return np.gradient(emg)
        # return emg * np.gradient(emg) * 10

    @staticmethod
    def filter_emg_savgol(emg):
        return sp.savgol_filter(emg, window_length=45, polyorder=4)

    @staticmethod
    def filter_chebyshev_type2(emg):
        cutoff_freq = 10  # Hz
        sampling_freq = 1000  # Hz
        stop_attenuation_db = 40  # dB
        filter_order = 4

        nyquist_freq = 0.5 * sampling_freq
        normalized_cutoff_freq = cutoff_freq / nyquist_freq

        # Design Chebyshev Type II high-pass filter
        b, a = sp.cheby2(filter_order, stop_attenuation_db, normalized_cutoff_freq, btype='highpass', analog=False)
        filtered = sp.filtfilt(b, a, emg)
        return filtered

    @staticmethod
    def filter_emg_conventional(emg):
        low_pass = 6
        sfreq = 2000
        high_band = 10
        low_band = 35
        amplification_factor = 2

        offset = np.mean(emg)  # Calculate the mean value of the signal
        offset_removed_signal = emg - offset

        # normalise cut-off frequencies to sampling frequency
        high_band = high_band / (sfreq / 2)
        low_band = low_band / (sfreq / 2)

        # create bandpass filter for EMG
        b1, a1 = sp.butter(4, [high_band, low_band], btype='bandpass')

        # process EMG signal: filter EMG
        emg_filtered = sp.filtfilt(b1, a1, offset_removed_signal)

        # process EMG signal: rectify
        emg_rectified = abs(emg_filtered)

        # create lowpass filter and apply to rectified signal to get EMG envelope
        low_pass = low_pass / (sfreq / 2)
        b2, a2 = sp.butter(4, low_pass, btype='lowpass')
        emg_envelope = sp.filtfilt(b2, a2, emg_rectified)

        emg_amplified = emg_envelope * amplification_factor

        return emg_amplified

    @staticmethod
    def filter_emg_notch(emg):
        fs = 1000.0  # Sampling frequency
        f0 = 50.0  # Frequency to be removed (notch frequency)
        bw = 2.0  # Bandwidth of the notch in Hz
        attenuation_dB = 20.0  # Desired attenuation at 50 Hz in dB
        Q = f0 / bw
        # Design the notch filter
        b, a = sp.iirnotch(f0, Q, fs)
        # Adjust the gain to achieve the desired attenuation
        gain = 10 ** (-attenuation_dB / 20)
        b *= gain

        filtered_emg = sp.lfilter(b, a, emg)

        return filtered_emg

    @staticmethod
    def filter_emg_pyemgpipeline(emg):
        # Ensure the input is a NumPy array
        if not isinstance(emg, np.ndarray):
            emg = np.array(emg)

        emg_trial = pyemgpipeline.wrappers.EMGMeasurement(emg, hz=1000)

        emg_trial.remove_dc_offset()

        emg_trial.apply_bandpass_filter(lowcut=20, highcut=500, order=4)

        emg_trial.apply_full_wave_rectification()

        emg_trial.apply_linear_envelope(lowcut=6, order=2)

        return emg_trial

    default_emg_filter_name = 'rolling_rms'

    @staticmethod
    def filter_emg(emg_raw, emg_filter_name):
        if emg_filter_name == 'raw':
            return emg_raw

        elif emg_filter_name == 'default':
            return LcaUtils.filter_emg(emg_raw, LcaUtils.default_emg_filter_name)

        elif emg_filter_name == 'savgol':
            return LcaUtils.filter_emg_savgol(emg_raw)

        elif emg_filter_name == 'rolling_rms':
            return LcaUtils.filter_emg_rolling_rms(emg_raw)

        elif emg_filter_name == 'high_pass':
            return LcaUtils.filter_emg_high_pass(emg_raw)

        elif emg_filter_name == 'low_pass':
            return LcaUtils.filter_emg_low_pass(emg_raw)

        elif emg_filter_name == 'band_pass':
            return LcaUtils.filter_emg_band_pass(emg_raw)

        elif emg_filter_name == 'chebyshev_type2':
            return LcaUtils.filter_chebyshev_type2(emg_raw)

        elif emg_filter_name == 'notch':
            return LcaUtils.filter_emg_notch(emg_raw)

        elif emg_filter_name == 'conventional':
            return LcaUtils.filter_emg_conventional(emg_raw)

        elif emg_filter_name == 'pyemgpipeline':
            return LcaUtils.filter_emg_pyemgpipeline(emg_raw)

        elif emg_filter_name == 'gradient':
            return LcaUtils.filter_emg_gradient(emg_raw)


class LcaData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.name = file_name.split('/')[-1].replace('.txt', '')

        self.time, self.hammer_raw, self.emg_raw = LcaFile.read(file_name)

        self.hammer_filtered = None
        self.hammer_time_emg = None
        self.hammer_value_emg = None

        self.apply_hammer_filter()

        self.emg_filtered = None
        self.peak_time_emg = None
        self.peak_value_emg = None

        self.apply_emg_filter('default')

    def apply_hammer_filter(self):
        self.hammer_filtered = LcaUtils.filter_hammer(self.hammer_raw)
        self.peak_time_hammer, self.peak_value_hammer = LcaUtils.find_hammer_peaks(self.time, self.hammer_filtered,
                                                                                   None)

    def apply_emg_filter(self, emg_filter_name):
        self.emg_filtered = LcaUtils.filter_emg(self.emg_raw, emg_filter_name)
        self.peak_time_emg, self.peak_value_emg = LcaUtils.find_emg_peaks(self.time, self.emg_filtered,
                                                                          None)


class LcaPlot:
    def __init__(self, lca_data):
        self.lca_data = lca_data

        self.figure, self.hammer_axes = plt.subplots(figsize=(10, 3.7))

        self.hammer_axes.set_title(lca_data.name)
        self.hammer_axes.grid(True, linestyle='--', alpha=0.7)
        self.hammer_axes.set_xlabel("Time(s)")
        self.hammer_axes.set_ylabel("Hammer signal (V)", color='red')
        self.hammer_axes.tick_params(axis='y', labelcolor='red')

        self.emg_axes = self.hammer_axes.twinx()
        self.emg_axes.set_ylabel("EMG signal (mV)", color='blue')
        self.emg_axes.tick_params(axis='y', labelcolor='blue')

        # self.hammer_raw_plot = None
        self.hammer_filtered_plot, = self.hammer_axes.plot(self.lca_data.time, self.lca_data.hammer_filtered,
                                                           label="Hammer", color='red', linewidth=1, alpha=0.7)
        self.hammer_peaks_plot, = self.hammer_axes.plot(self.lca_data.peak_time_hammer, self.lca_data.peak_value_hammer,
                                                        'ro', markerfacecolor='none', markeredgecolor='red',
                                                        label="Hammer peak", alpha=0.7)
        self.emg_raw_plot, = self.emg_axes.plot(self.lca_data.time, self.lca_data.emg_raw,
                                                label="EMG", color='blue', linewidth=1, alpha=0.4)
        self.emg_filtered_plot, = self.emg_axes.plot(self.lca_data.time, self.lca_data.emg_filtered,
                                                     label="EMG filtered", color='blue', linewidth=1, alpha=0.7)
        self.emg_peak_plot, = self.emg_axes.plot(self.lca_data.peak_time_emg, self.lca_data.peak_value_emg,
                                                 'bo', markerfacecolor='none', markeredgecolor='blue',
                                                 label="EMG peak", alpha=0.7)

        self.reset_zoom()

    def reset_zoom(self):
        ylim_hammer = self.hammer_axes.get_ylim()
        ylim_emg = self.emg_axes.get_ylim()
        ylim = (min(ylim_hammer[0] / 10, ylim_emg[0]), max(ylim_hammer[1] / 10, ylim_emg[1]))
        self.hammer_axes.set_ylim(ylim[0] * 10, ylim[1] * 10)
        self.emg_axes.set_ylim(ylim)

    def apply_emg_filter(self, emg_filter_name):
        self.lca_data.apply_emg_filter(emg_filter_name)

        self.emg_filtered_plot.set_ydata(self.lca_data.emg_filtered)
        self.emg_peak_plot.set_xdata(self.lca_data.peak_time_emg)
        self.emg_peak_plot.set_ydata(self.lca_data.peak_value_emg)


class LcaWindow:

    def __init__(self, lca_data):
        self.lca_data = lca_data
        self.lca_plot = LcaPlot(lca_data)

        # Cannot create variables before root_window is created
        self.root_window = tk.Tk()

        self.show_hammer_raw_plot = tk.IntVar(value=0)
        self.show_hammer_filtered_plot = tk.IntVar(value=1)
        self.show_hammer_peaks_plot = tk.IntVar(value=1)
        self.show_emg_raw_plot = tk.IntVar(value=1)
        self.show_emg_filtered_plot = tk.IntVar(value=1)
        self.show_emg_peaks_plot = tk.IntVar(value=1)

        self.emg_filter_name = tk.StringVar(value='default')

        self.create_root_window(self.root_window)

    def create_root_window(self, root_window):
        root_window.title("EMG latency calculation app")
        logo = tk.PhotoImage(file="LOGO.png")
        root_window.img = logo
        # window.iconphoto(False, window.img)

        # top_frame = tk.Frame(window)
        # top_frame.pack()
        top_frame = tk.Frame(root_window)
        top_frame.pack(side="top", fill="both", expand=True)

        # scrollbar_x = tk.Scrollbar(top_frame, orient="horizontal", command=self.lca_plot.hammer_axes.set_xlim)
        # scrollbar_x.pack(side="bottom", fill="x")
        #
        # scrollbar_y = tk.Scrollbar(top_frame, orient="vertical", command=self.lca_plot.hammer_axes.set_ylim)
        # scrollbar_y.pack(side="right", fill="y")

        # bottom_frame = tk.Frame(window)
        # bottom_frame.pack(side=tk.BOTTOM)

        menubar = self.create_menubar(root_window)
        root_window.config(menu=menubar)

        self.canvas = self.create_canvas(top_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)

        toolbar = self.create_toolbar(self.canvas, top_frame)

        return root_window

    def create_menubar(self, parent):
        menubar = tk.Menu(parent)

        file_menu = self.create_file_menu(menubar)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = self.create_view_menu(menubar)
        menubar.add_cascade(label="View", menu=view_menu)

        filter_menu = self.create_filter_menu(menubar)
        menubar.add_cascade(label="Filter", menu=filter_menu)

        latency_menu = self.create_latency_menu(menubar)
        menubar.add_cascade(label="Latency", menu=latency_menu)

        return menubar

    def create_file_menu(self, menubar):
        file_menu = tk.Menu(menubar, tearoff=False)

        file_menu.add_command(label="Open", command=self.file_open_onclick)
        file_menu.add_command(label="Close", command=self.file_close_onclick)

        return file_menu

    def create_view_menu(self, menubar):
        view_menu = tk.Menu(menubar, tearoff=False)

        # self.view_menu.add_checkbutton(label="Hammer (raw)", variable=self.show_hammer_raw_plot,
        #                                command=self.show_plot_onchanged)
        # self.view_menu.add_checkbutton(label="Hammer (filtered)", variable=self.show_hammer_filtered_plot,
        #                                command=self.show_plot_onchanged)
        view_menu.add_checkbutton(label="Hammer", variable=self.show_hammer_filtered_plot,
                                  command=self.show_plot_onchanged)
        view_menu.add_checkbutton(label="Hammer peaks", variable=self.show_hammer_peaks_plot,
                                  command=self.show_plot_onchanged)
        view_menu.add_checkbutton(label="EMG (raw)", variable=self.show_emg_raw_plot,
                                  command=self.show_plot_onchanged)
        view_menu.add_checkbutton(label="EMG (filtered)", variable=self.show_emg_filtered_plot,
                                  command=self.show_plot_onchanged)
        view_menu.add_checkbutton(label="EMG peaks", variable=self.show_emg_peaks_plot,
                                  command=self.show_plot_onchanged)

        view_menu.add_separator()

        view_menu.add_command(label="Reset zoom", command=self.view_reset_zoom_onclick)

        return view_menu

    def create_filter_menu(self, menubar):
        filter_menu = tk.Menu(menubar, tearoff=False)

        filter_menu.add_radiobutton(label="Raw signal", variable=self.emg_filter_name,
                                    value='raw', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Default filter (rolling RMS)", variable=self.emg_filter_name,
                                    value='default', command=self.emg_filter_name_onchanged)

        filter_menu.add_separator()

        filter_menu.add_radiobutton(label="Rolling RMS filter", variable=self.emg_filter_name,
                                    value='rolling_rms', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Savitzky–Golay filter", variable=self.emg_filter_name,
                                    value='savgol', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="High pass filter", variable=self.emg_filter_name,
                                    value='high_pass', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Low pass filter", variable=self.emg_filter_name,
                                    value='low_pass', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Band pass filter", variable=self.emg_filter_name,
                                    value='band_pass', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Chebyshev type2 filter", variable=self.emg_filter_name,
                                    value='chebyshev_type2', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="Notch filter", variable=self.emg_filter_name,
                                    value='notch', command=self.emg_filter_name_onchanged)

        filter_menu.add_separator()

        filter_menu.add_radiobutton(label="Conventional EMG processing", variable=self.emg_filter_name,
                                    value='conventional', command=self.emg_filter_name_onchanged)
        filter_menu.add_radiobutton(label="pyemgpipeline EMG processing", variable=self.emg_filter_name,
                                    value='pyemgpipeline', command=self.emg_filter_name_onchanged)

        filter_menu.add_separator()

        filter_menu.add_radiobutton(label="Gradient", variable=self.emg_filter_name,
                                    value='gradient', command=self.emg_filter_name_onchanged)

        return filter_menu

    def create_latency_menu(self, menubar):
        latency_menu = tk.Menu(menubar, tearoff=False)

        latency_menu.add_separator()

        latency_menu.add_command(label="Show statistics", command=self.latency_show_stats_onclick)

        return latency_menu

    def create_canvas(self, parent):
        canvas = FigureCanvasTkAgg(self.lca_plot.figure, master=parent)
        canvas.mpl_connect('scroll_event', self.canvas_onscroll)

        return canvas

    def create_toolbar(self, canvas, parent):
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        return toolbar

    def run(self):
        self.root_window.mainloop()

    def file_open_onclick(self):
        file_name = LcaFile.ask_file_name()
        lca_data = LcaData(file_name)
        lca_window = LcaWindow(lca_data)
        lca_window.run()

    def file_close_onclick(self):
        self.root_window.destroy()

    def show_plot_onchanged(self):
        self.refresh_plot()

    def canvas_onscroll(self, event):
        scale_x = 1.5
        scale_y = 1.0

        # get the current x and y limits
        xlim = self.lca_plot.emg_axes.get_xlim()
        ylim = self.lca_plot.emg_axes.get_ylim()

        if event.button == 'up':
            scale_x = 1 / scale_x
            scale_y = 1 / scale_y
        elif event.button == 'down':
            pass
        else:
            return

        new_xlim = ([event.xdata - (event.xdata - xlim[0]) * scale_x,
                     event.xdata + (xlim[1] - event.xdata) * scale_x])

        new_ylim = ([event.ydata - (event.ydata - ylim[0]) * scale_y,
                     event.ydata + (ylim[1] - event.ydata) * scale_y])

        self.lca_plot.hammer_axes.set_xlim(new_xlim[0], new_xlim[1])
        self.lca_plot.emg_axes.set_xlim(new_xlim[0], new_xlim[1])

        self.lca_plot.hammer_axes.set_ylim(new_ylim[0] * 10, new_ylim[1] * 10)
        self.lca_plot.emg_axes.set_ylim(new_ylim[0], new_ylim[1])

        self.canvas.draw_idle()

    def refresh_plot(self):
        # self.lca_plot.hammer_raw_plot.set_visible(show_hammer_raw_plot.get())
        self.lca_plot.hammer_filtered_plot.set_visible(self.show_hammer_filtered_plot.get())
        self.lca_plot.hammer_peaks_plot.set_visible(self.show_hammer_peaks_plot.get())
        self.lca_plot.hammer_axes.legend()

        self.lca_plot.emg_raw_plot.set_visible(self.show_emg_raw_plot.get())
        self.lca_plot.emg_filtered_plot.set_visible(self.show_emg_filtered_plot.get())
        self.lca_plot.emg_peak_plot.set_visible(self.show_emg_peaks_plot.get())
        self.lca_plot.emg_axes.legend()

        self.canvas.draw_idle()

    def view_reset_zoom_onclick(self):
        self.lca_plot.reset_zoom()

    def emg_filter_name_onchanged(self):
        self.lca_plot.apply_emg_filter(self.emg_filter_name.get())
        self.canvas.draw_idle()

    def latency_show_stats_onclick(self):
        pass

def main()->int:
    file_name = LcaFile.ask_file_name()
    if file_name == '':
        return 1
    else:
        lca_data = LcaData(file_name)
        lca_window = LcaWindow(lca_data)
        lca_window.run()
        return 0

if __name__ == '__main__':
    sys.exit(main())
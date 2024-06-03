import tkinter as tk
from tkinter import filedialog # , messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tabulate import tabulate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

# from PIL import Image, ImageTk
# import pyemgpipeline as pep

archivo_global = None

time = None
hammer_original = None
emg_original = None

figure = None
ax_hammer = None
ax_emg = None
xlim = None
hammer_original_data = None
emg_original_data = None
hammer_filtered_data = None
emg_filtered_data = None
hammer_peak_data = None
emg_peak_data = None
canvas = None  # Definir canvas como variable global


def abrir_archivo():
    global archivo_global
    if archivo_global is None:
        archivo_global = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Archivos de texto", "*.txt"),
                                                                                        ("Todos los archivos", "*.*")])
        generar_grafica()
    return archivo_global


def leer_datos(archivo):
    time = []
    hammer = []
    emg = []
    procesando_datos = False
    with open(archivo, 'r') as file:
        for linea in file:
            if procesando_datos:
                try:
                    valores = [float(valor.replace(',', '.')) for valor in linea.split('\t')]
                    time.append(valores[0])
                    hammer.append(valores[1])
                    emg.append(valores[2])
                except ValueError:
                    pass
            elif linea.startswith('0\t'):
                procesando_datos = True
    return time, hammer, emg


def filter_hammer_default(hammer):
    # threshold_hammer = 0.02  # Umbral para establecer valores casi cero a cero
    # hammer_filtered = [0 if abs(x) < threshold_hammer else x for x in hammer]
    offset = np.mean(hammer)  # Calculate the mean value of the signal
    offset_removed_signal = hammer - offset

    return offset_removed_signal


def filter_emg_rms(emg):
    rms_emg = pd.DataFrame(abs(emg) ** 2).rolling(5).mean() ** 0.5

    # Calculate the RMS

    # rms_emg = np.sqrt(np.mean(emg ** 2))

    return rms_emg


def filter_emg_high_pass(emg):
    sampling_freq = 1000
    # cutoff_freq = 20
    cutoff_freq = 20
    order = 2

    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = sp.butter(order, normalized_cutoff_freq, btype='highpass', analog=False)
    filtered_signal = sp.filtfilt(b, a, emg)
    return filtered_signal


def filter_emg_low_pass(emg):
    cutoff_freq = 35  # Hz
    sampling_freq = 1000  # Hz
    filter_order = 2

    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq

    offset = np.mean(emg)  # Calculate the mean value of the signal
    offset_removed_signal = emg - offset

    # Butterworth low-pass filter
    b, a = sp.butter(filter_order, normalized_cutoff_freq, btype='low', analog=False)
    filtered_signal = sp.filtfilt(b, a, offset_removed_signal)

    return filtered_signal


def filter_emg_band_pass(emg):
    # Example usage:
    low_cutoff_freq = 20  # Hz
    high_cutoff_freq = 35  # Hz
    sampling_freq = 1000  # Hz
    filter_order = 4

    nyquist_freq = 0.5 * sampling_freq
    normalized_low_cutoff_freq = low_cutoff_freq / nyquist_freq
    normalized_high_cutoff_freq = high_cutoff_freq / nyquist_freq

    # Design Butterworth band-pass filter
    b, a = sp.butter(filter_order, [normalized_low_cutoff_freq, normalized_high_cutoff_freq], btype='band',
                     analog=False)
    filtered_signal = sp.filtfilt(b, a, emg)

    return filtered_signal


def rolling_rms_filter(emg, half_window_size):
    window_size = 2 * half_window_size + 1
    window = np.ones(window_size) / float(window_size)

    return np.sqrt(
        sp.fftconvolve(
            np.power(emg, 2),
            window,
            'same'))


def filter_emg_rolling_rms(emg):
    # return rolling_rms_filter(emg, 10)
    return rolling_rms_filter(emg, 3)
    # return rolling_rms_filter(rolling_rms_filter(emg, 3), 1)
    # return rolling_rms_filter(emg, 2)


def filter_emg_prueba(emg):
    # return np.abs(emg)
    # return np.gradient(emg)
    return emg * np.gradient(emg) * 10


def filter_emg_savgol(emg):
    filtered_emg_savgol = sp.savgol_filter(emg, 45, 4)

    return filtered_emg_savgol


def filter_chebyshev_type2(emg):
    cutoff_freq = 10  # Hz
    sampling_freq = 1000  # Hz
    stop_attenuation_db = 40  # dB
    filter_order = 4

    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq

    # Design Chebyshev Type II high-pass filter
    b, a = sp.cheby2(filter_order, stop_attenuation_db, normalized_cutoff_freq, btype='highpass', analog=False)
    filtered_signal = sp.filtfilt(b, a, emg)
    return filtered_signal


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


def proces_emg_bandpas(emg):
    # Ensure the input is a NumPy array
    if not isinstance(emg, np.ndarray):
        emg = np.array(emg)

    emg_trial = pep.wrappers.EMGMeasurement(emg, hz=1000)

    # Remover offset DC
    emg_trial.remove_dc_offset()

    # Aplicar filtro pasabanda
    emg_trial.apply_bandpass_filter(lowcut=20, highcut=500, order=4)

    # Rectificación de onda completa
    emg_trial.apply_full_wave_rectification()

    # Calcular la envolvente lineal
    emg_trial.apply_linear_envelope(lowcut=6, order=2)

    return emg_trial


def generar_grafica():
    global canvas, figure, ax_hammer, ax_emg, xlim
    global time, hammer_original, emg_original
    global hammer_original_data, emg_original_data, hammer_filtered_data, emg_filtered_data, hammer_peak_data, emg_peak_data

    if not archivo_global:
        return

    time, hammer_original, emg_original = leer_datos(archivo_global)
    # hammer_original = [value / 10 for value in hammer_original]

    hammer_filtered = filter_hammer_default(hammer_original)
    emg_filtered = filter_emg_default(emg_original)

    name_graph = archivo_global.split('/')[-1]  # Obtener la parte después de la última barra
    name_graph = name_graph.replace('.txt', '')

    # # Dividir los valores de hammer entre 10

    if figure is None:
        figure, ax_hammer = plt.subplots(figsize=(10, 3.7))
        ax_hammer.set_title(name_graph)
        # loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        # ax_hammer.xaxis.set_major_locator(loc)
        # ax_hammer.xaxis.set_major_locator(plticker.MultipleLocator(0.1))

        ax_hammer.grid(True, linestyle='--', alpha=0.7)
        ax_hammer.set_xlabel('Time(s)')
        ax_hammer.set_ylabel('Hammer signal (V)', color='red')
        ax_hammer.tick_params(axis='y', labelcolor='red')

        ax_emg = ax_hammer.twinx()
        ax_emg.set_ylabel('EMG signal (mV)', color='blue')
        ax_emg.tick_params(axis='y', labelcolor='blue')

    # hammer_original_data, = ax.plot(time, hammer_original, label='Hammer (dV)', color='red', linewidth=1, alpha=0.4)
    hammer_filtered_data, = ax_hammer.plot(time, hammer_filtered, label='Hammer', color='red',
                                           linewidth=1, alpha=0.7)
    xlim = ax_hammer.get_xlim()
    peak_time_hammer, peak_value_hammer = calcular_maximos_hammer(xlim)
    hammer_peak_data, = ax_hammer.plot(peak_time_hammer, peak_value_hammer, 'ro', label='Hammer peak', alpha=0.7,
                                       markerfacecolor='none', markeredgecolor='red')

    emg_original_data, = ax_emg.plot(time, emg_original, label='EMG', color='blue', linewidth=1, alpha=0.4)
    emg_filtered_data, = ax_emg.plot(time, emg_filtered, label='EMG filtered', color='blue',
                                     linewidth=1, alpha=0.7)
    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    emg_peak_data, = ax_emg.plot(peak_time_emg, peak_value_emg, 'bo', label='EMG peak', alpha=0.7,
                                 markerfacecolor='none',
                                 markeredgecolor='blue')
    # peak_value_hammer = [value / 10 for value in peak_value_hammer]

    ylim_emg = ax_emg.get_ylim()
    ylim_hammer = ax_hammer.get_ylim()
    ylim = (min(ylim_emg[0], ylim_hammer[0] / 10), max(ylim_emg[1], ylim_hammer[1] / 10))
    ax_emg.set_ylim(ylim)
    ax_hammer.set_ylim(ylim[0] * 10, ylim[1] * 10)

    # ax_hammer.set_xticks(np.arange(min(time), max(time), 0.01))

    # Crear el contenedor principal para la gráfica y las barras de desplazamiento
    topFrame = tk.Frame(ventana)
    topFrame.pack(side="top", fill="both", expand=True)

    # Agregar barras de desplazamiento
    scrollbar_x = tk.Scrollbar(topFrame, orient="horizontal", command=ax_hammer.set_xlim)
    scrollbar_y = tk.Scrollbar(topFrame, orient="vertical", command=ax_hammer.set_ylim)

    scrollbar_x.pack(side="bottom", fill="x")
    scrollbar_y.pack(side="right", fill="y")

    # Crear el lienzo de la figura y vincularlo a la interfaz gráfica de tkinter
    canvas = FigureCanvasTkAgg(figure, master=topFrame)

    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side="top", fill="both", expand=True)

    # Crear la barra de herramientas
    toolbar = NavigationToolbar2Tk(canvas, topFrame)
    toolbar.update()

    actualizar_leyenda()

    # Función para aumentar la gráfica
    def zoom_fun(event):
        base_scale = 1.5
        # get the current x and y limits
        cur_xlim = ax_hammer.get_xlim()
        cur_ylim = ax_hammer.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        # set new limits
        ax_hammer.set_xlim([xdata - (xdata - cur_xlim[0]) * scale_factor,
                            xdata + (cur_xlim[1] - xdata) * scale_factor])
        ax_emg.set_xlim([xdata - (xdata - cur_xlim[0]) * scale_factor,
                         xdata + (cur_xlim[1] - xdata) * scale_factor])
        # ax.set_ylim([ydata - (ydata - cur_ylim[0]) * scale_factor,
        #              ydata + (cur_ylim[1] - ydata) * scale_factor])
        canvas.draw_idle()

    # attach the call back
    canvas.mpl_connect('scroll_event', zoom_fun)


def calcular_maximos_hammer(xlim):
    global time  # , hammer, emg
    global hammer_filtered_data

    hammer_filtered = hammer_filtered_data.get_ydata()

    # Calculamos el rango
    indices_en_rango = np.where((time >= xlim[0]) & (time <= xlim[1]))[0]

    # Filtrar los datos dentro del rango de la gráfica ampliada
    time_en_rango = np.array(time)[indices_en_rango]
    hammer_en_rango = np.array(hammer_filtered)[indices_en_rango]

    # Encontrar los máximos del Hammer por encima de 0.1
    peak_hammer, _ = sp.find_peaks(hammer_en_rango, height=0.1)
    peak_time_hammer = [round(time_en_rango[i], 3) for i in peak_hammer]
    peak_value_hammer = [round(hammer_en_rango[i], 3) for i in peak_hammer]

    return peak_time_hammer, peak_value_hammer


def calcular_maximos_emg(xlim):
    global time  # , hammer, emg
    global emg_filtered_data

    emg_filtered = emg_filtered_data.get_ydata()

    # Calculamos el rango
    indices_en_rango = np.where((time >= xlim[0]) & (time <= xlim[1]))[0]

    # Filtrar los datos dentro del rango de la gráfica ampliada
    emg_en_rango = np.array(emg_filtered)[indices_en_rango]
    time_en_rango = np.array(time)[indices_en_rango]

    # Encontrar los máximos del EMG por encima de 0.05
    peak_emg, _ = sp.find_peaks(emg_en_rango, height=0.06)
    peak_time_emg = [round(time_en_rango[i], 3) for i in peak_emg]
    peak_value_emg = [round(emg_en_rango[i], 3) for i in peak_emg]

    return peak_time_emg, peak_value_emg


def calcular_delay_row(time, value, peak_time_emg, peak_value_emg):
    delay_row = (time, value, None, None, None)
    max_peak_value_emg = 0
    for i, peak_time in enumerate(peak_time_emg):
        if time < peak_time and peak_time <= time + 0.5:
            if peak_value_emg[i] > max_peak_value_emg:
                delay_row = (time, value, peak_time, peak_value_emg[i], round(peak_time - time, 3))
                max_peak_value_emg = peak_value_emg[i]

    return delay_row


def calcular_delay_rows(peak_time_hammer, peak_value_hammer, peak_time_emg, peak_value_emg):
    delay_rows = [calcular_delay_row(time, peak_value_hammer[i], peak_time_emg, peak_value_emg) for i, time in
                  enumerate(peak_time_hammer)]
    return delay_rows


def mostrar_resultados_calculos():
    global ax_hammer
    xlim = ax_hammer.get_xlim()

    peak_time_hammer, peak_value_hammer = calcular_maximos_hammer(xlim)
    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    delay_rows = calcular_delay_rows(peak_time_hammer, peak_value_hammer, peak_time_emg, peak_value_emg)

    valid_delays = list(filter(lambda delay: delay is not None, list(map(lambda delay_row: delay_row[4], delay_rows))))
    mean_delay = np.mean(valid_delays)

    # Crear una tabla con los resultados
    resultados_table = tabulate({
        'T hammer(s)': list(map(lambda delay_row: delay_row[0], delay_rows)),
        'Hammer peaks(V)': list(map(lambda delay_row: delay_row[1], delay_rows)),
        'T EMG(s)': list(map(lambda delay_row: delay_row[2], delay_rows)),
        'EMG peaks(mV)': list(map(lambda delay_row: delay_row[3], delay_rows)),
        'Delay(s)': list(map(lambda delay_row: delay_row[4], delay_rows))
    }, headers='keys', tablefmt='pretty', showindex=False)

    # Agregar la media de los delays a la tabla
    resultados_table += f"\nMean delay: {mean_delay * 1000:.3f} ms"

    # Crear una ventana para mostrar los resultados
    ventana_resultados = tk.Toplevel()
    ventana_resultados.title("Statistics Results")

    # Crear un frame para contener la tabla, el scrollbar y el botón de guardar
    frame_resultados = tk.Frame(ventana_resultados)
    frame_resultados.pack(fill=tk.BOTH, expand=True)

    # Crear un widget de texto para mostrar la tabla
    text_resultados = tk.Text(frame_resultados, font=('Courier', 10), wrap=tk.NONE)
    text_resultados.insert(tk.END, resultados_table)
    text_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Crear un scrollbar y vincularlo al widget de texto
    scrollbar = tk.Scrollbar(frame_resultados, command=text_resultados.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_resultados.config(yscrollcommand=scrollbar.set)

    # Crear un botón para guardar los resultados en un archivo
    def guardar_resultados():
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(resultados_table)

    boton_guardar = tk.Button(ventana_resultados, text="Save statistics as", command=guardar_resultados)
    boton_guardar.pack(pady=10)


def actualizar_leyenda():
    global canvas, ax_hammer, ax_emg, hammer_original_data, emg_original_data, hammer_filtered_data, emg_filtered_data, hammer_peak_data, emg_peak_data

    # hammer_original_data.set_visible(show_original_hammer.get())
    hammer_filtered_data.set_visible(show_filtered_hammer.get())
    hammer_peak_data.set_visible(show_hammer_peaks.get())
    ax_hammer.legend()

    emg_original_data.set_visible(show_original_emg.get())
    emg_filtered_data.set_visible(show_filtered_emg.get())
    emg_peak_data.set_visible(show_emg_peaks.get())
    ax_emg.legend()

    canvas.draw_idle()


def filter_emg_default(emg):
    # return filter_emg_high_pass(emg)
    # return filter_emg_rolling_rms(emg)
    return filter_emg_prueba(emg)


def actualizar_emg_filter():
    global canvas, xlim, emg_filtered_data, emg_peak_data
    global emg_original

    if emg_filter.get() == 'original':
        emg_filtered_data.set_ydata(emg_original)

    elif emg_filter.get() == 'default':
        emg_filtered = filter_emg_default(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'savgol':
        emg_filtered = filter_emg_savgol(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'rolling_rms':
        emg_filtered = filter_emg_rolling_rms(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'prueba':
        emg_filtered = filter_emg_prueba(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'high_pass':
        emg_filtered = filter_emg_high_pass(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'low_pass':
        emg_filtered = filter_emg_low_pass(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'band_pass':
        emg_filtered = filter_emg_band_pass(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'Chebyshev2':
        emg_filtered = filter_chebyshev_type2(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'conventional':
        emg_filtered = filter_emg_conventional(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'notch':
        emg_filtered = filter_emg_notch(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    elif emg_filter.get() == 'process':
        emg_filtered = proces_emg_bandpas(emg_original)
        emg_filtered_data.set_ydata(emg_filtered)

    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    emg_peak_data.set_xdata(peak_time_emg)
    emg_peak_data.set_ydata(peak_value_emg)

    # ax.legend()
    canvas.draw_idle()


def close_gui(ventana):
    ventana.destroy()


ventana = tk.Tk()
ventana.title("EMG latency calculation app")
topFrame = tk.Frame(ventana)
topFrame.pack()
bottomFrame = tk.Frame(ventana)
bottomFrame.pack(side=tk.BOTTOM)
menu = tk.Menu(ventana)
ventana.config(menu=menu)
subMenu_file = tk.Menu(menu, tearoff=False)
menu.add_cascade(label='File', menu=subMenu_file)
subMenu_file.add_command(label='Open', command=abrir_archivo)
subMenu_file.add_command(label='Close', command=lambda: close_gui(ventana))
subMenu_view = tk.Menu(menu, tearoff=False)
menu.add_cascade(label='View', menu=subMenu_view)

show_original_hammer = tk.IntVar(value=0)
show_filtered_hammer = tk.IntVar(value=1)
show_hammer_peaks = tk.IntVar(value=1)
show_original_emg = tk.IntVar(value=1)
show_filtered_emg = tk.IntVar(value=1)
show_emg_peaks = tk.IntVar(value=1)

# subMenu_view.add_checkbutton(label="Hammer (original)", variable=show_original_hammer, command=actualizar_leyenda)
# subMenu_view.add_checkbutton(label="Hammer (filtered)", variable=show_filtered_hammer, command=actualizar_leyenda)
subMenu_view.add_checkbutton(label="Hammer", variable=show_filtered_hammer, command=actualizar_leyenda)
subMenu_view.add_checkbutton(label="Hammer peaks", variable=show_hammer_peaks, command=actualizar_leyenda)
subMenu_view.add_checkbutton(label="EMG (original)", variable=show_original_emg, command=actualizar_leyenda)
subMenu_view.add_checkbutton(label="EMG (filtered)", variable=show_filtered_emg, command=actualizar_leyenda)
subMenu_view.add_checkbutton(label="EMG peaks", variable=show_emg_peaks, command=actualizar_leyenda)
subMenu_view.add_separator()
subMenu_view.add_command(label='Statistics', command=mostrar_resultados_calculos)

subMenu_filter = tk.Menu(menu, tearoff=False)
menu.add_cascade(label='Filter', menu=subMenu_filter)

emg_filter = tk.StringVar(value='default')
subMenu_filter.add_radiobutton(label='Original signal', variable=emg_filter, value='original',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Default (Rolling RMS) filter', variable=emg_filter, value='default',
                               command=actualizar_emg_filter)
subMenu_filter.add_separator()
# subMenu_filter.add_radiobutton(label='RMS filter', variable=emg_filter, value='RMS',
#                                command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Rolling RMS filter', variable=emg_filter, value='rolling_rms',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Savitzky–Golay filter', variable=emg_filter, value='savgol',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='High pass filter', variable=emg_filter, value='high_pass',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Low pass filter', variable=emg_filter, value='low_pass',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Band pass filter', variable=emg_filter, value='band_pass',
                               command=actualizar_emg_filter)

subMenu_filter.add_radiobutton(label='Chebyshev type2 filter', variable=emg_filter, value='Chebyshev2',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Conventional EMG processing', variable=emg_filter, value='conventional',
                               command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Notch filter', variable=emg_filter, value='notch', command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Processed emg', variable=emg_filter, value='process',
                               command=actualizar_emg_filter)

logo = tk.PhotoImage(file="lca.png")
ventana.iconphoto(True, logo)

ventana.mainloop()

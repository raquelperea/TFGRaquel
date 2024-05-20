## GRÁFICAS SUPERPUESTAS PREPROCESADAS (No se actualiza hasta que no haces algo)
# For the executable: pyinstaller --onefile prueba1.py
# Y después click del ratón derecho y abrir en explorer


import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tabulate import tabulate

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

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
    threshold_hammer = 0.02  # Umbral para establecer valores casi cero a cero
    hammer_filtered = [0 if abs(x) < threshold_hammer else x for x in hammer]

    return hammer_filtered


def filter_emg_default(emg):
    return filter_emg_high_pass(emg)

def filter_emg_high_pass(emg):
    sampling_freq = 1000
    # cutoff_freq = 20
    cutoff_freq = 10
    order = 4

    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = sp.butter(order, normalized_cutoff_freq, btype='highpass', analog=False)
    print(b,a)
    filtered_signal = sp.filtfilt(b, a, emg)
    return filtered_signal

def filter_emg_low_pass(emg):
    cutoff_freq = 25 # Hz
    sampling_freq = 1000  # Hz
    filter_order = 4

    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq

    #Butterworth low-pass filter
    b, a = sp.butter(filter_order, normalized_cutoff_freq, btype='low', analog=False)
    filtered_signal = sp.filtfilt(b, a, emg)

    return filtered_signal

def filter_emg_band_pass(emg):
    # Example usage:
    low_cutoff_freq = 10  # Hz
    high_cutoff_freq = 25  # Hz
    sampling_freq = 1000  # Hz
    filter_order = 4

    nyquist_freq = 0.5 * sampling_freq
    normalized_low_cutoff_freq = low_cutoff_freq / nyquist_freq
    normalized_high_cutoff_freq = high_cutoff_freq / nyquist_freq

    # Design Butterworth band-pass filter
    b, a = sp.butter(filter_order, [normalized_low_cutoff_freq, normalized_high_cutoff_freq], btype='band', analog=False)
    filtered_signal = sp.filtfilt(b, a, emg)

    return filtered_signal





def filter_emg_savgol(emg):
    filtered_emg_savgol = sp.savgol_filter(emg, 51, 4)

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

import scipy.signal as sp


# def filter_emg_kalman(emg):
#
#     R = 0.1**2 # estimate of measurement variance, change to see effect
#
#     # intial guesses
#     xhat[0] = 0.0
#     P[0] = 1.0
#
#     for k in range(1,n_iter):
#         # time update
#         xhatminus[k] = xhat[k-1]
#         Pminus[k] = P[k-1]+Q
#
#         # measurement update
#         K[k] = Pminus[k]/( Pminus[k]+R )
#         xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
#         P[k] = (1-K[k])*Pminus[k]


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

    # # Dividir los valores de hammer entre 10

    if figure is None:
        figure, ax_hammer = plt.subplots(figsize=(10, 3.7))
        ax_hammer.set_title('EMG and hammer graph')
        ax_hammer.grid(True, linestyle='--', alpha=0.7)
        ax_hammer.set_xlabel('Time(s)')
        ax_hammer.set_ylabel('Hammer signal (V)', color = 'red')
        ax_hammer.tick_params(axis='y', labelcolor='red')

        ax_emg = ax_hammer.twinx()
        ax_emg.set_ylabel('EMG signal (mV)', color = 'blue')
        ax_emg.tick_params(axis='y', labelcolor='blue')


    # hammer_original_data, = ax.plot(time, hammer_original, label='Hammer (dV)', color='red', linewidth=1, alpha=0.4)
    hammer_filtered_data, = ax_hammer.plot(time, hammer_filtered, label='Hammer filtered', color='red',
                                           linewidth=1, alpha=0.7)
    xlim = ax_hammer.get_xlim()
    peak_time_hammer, peak_value_hammer = calcular_maximos_hammer(xlim)
    hammer_peak_data, = ax_hammer.plot(peak_time_hammer, peak_value_hammer, 'ro', label='Hammer peak', alpha=0.7,
                                       markerfacecolor='none', markeredgecolor='red')

    emg_original_data, = ax_emg.plot(time, emg_original, label='EMG', color='blue', linewidth=1, alpha=0.4)
    emg_filtered_data, = ax_emg.plot(time, emg_filtered, label='EMG filtered', color='blue',
                                        linewidth=1, alpha=0.7)
    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    emg_peak_data, = ax_emg.plot(peak_time_emg, peak_value_emg, 'bo', label='EMG peak', alpha=0.7, markerfacecolor='none',
                                    markeredgecolor='blue')
    # peak_value_hammer = [value / 10 for value in peak_value_hammer]

    ylim_emg = ax_emg.get_ylim()
    ylim_hammer = ax_hammer.get_ylim()
    ylim = (min(ylim_emg[0], ylim_hammer[0] / 10), max(ylim_emg[1], ylim_hammer[1] / 10))
    ax_emg.set_ylim(ylim)
    ax_hammer.set_ylim(ylim[0] * 10, ylim[1] * 10)

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
    peak_hammer, _ = sp.find_peaks(hammer_en_rango, height =0.1)
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
    peak_emg, _ = sp.find_peaks(emg_en_rango, height=0.05)
    peak_time_emg = [round(time_en_rango[i], 3) for i in peak_emg]
    peak_value_emg = [round(emg_en_rango[i], 3) for i in peak_emg]

    return peak_time_emg, peak_value_emg


def calcular_minimos_emg(xlim):
    global time  # , hammer, emg
    global emg_filtered_data

    emg_filtered = emg_filtered_data.get_ydata()
    emg_filtered_inverted = - emg_filtered

    # Calculamos el rango
    indices_en_rango = np.where((time >= xlim[0]) & (time <= xlim[1]))[0]

    # Filtrar los datos dentro del rango de la gráfica ampliada
    emg_en_rango = np.array(emg_filtered_inverted)[indices_en_rango]
    time_en_rango = np.array(time)[indices_en_rango]

    # Encontrar los máximos del EMG por encima de 0.05
    valley_emg, _ = sp.find_peaks(emg_en_rango, height=0.05)
    valley_time_emg = [round(time_en_rango[i], 3) for i in valley_emg]
    valley_value_emg = [round(emg_en_rango[i], 3) for i in valley_emg]

    return valley_time_emg, valley_value_emg

def juntar_peaks_valleys():
    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    valley_time_emg, valley_value_emg = calcular_minimos_emg(xlim)
    extremes_time_emg = peak_time_emg + valley_time_emg
    extremes_value_emg = peak_value_emg + valley_value_emg
    return extremes_time_emg, extremes_value_emg


def calcular_delay_row(time, value, extremes_time_emg, extremes_value_emg):
    delay_row = (time, value, None, None, None)
    max_peak_value_emg = 0
    for i, peak_time in enumerate(extremes_time_emg):
        if time < peak_time and peak_time <= time+0.5:
            if extremes_value_emg[i] > max_peak_value_emg:
                delay_row = (time, value, peak_time, extremes_value_emg[i], peak_time - time)
                max_peak_value_emg = extremes_value_emg[i]

    return delay_row


def calcular_delay_rows(peak_time_hammer, peak_value_hammer, extremes_time_emg, extremes_value_emg):
    delay_rows = [calcular_delay_row(time, peak_value_hammer[i], extremes_time_emg, extremes_value_emg) for i, time in enumerate(peak_time_hammer)]
    return delay_rows


def mostrar_resultados_calculos():
    global ax_hammer
    xlim = ax_hammer.get_xlim()

    peak_time_hammer, peak_value_hammer = calcular_maximos_hammer(xlim)

    extremes_time_emg, extremes_value_emg = juntar_peaks_valleys()

    # delays = [round(t_emg - t_hammer, 3) for t_emg, t_hammer in zip(peak_time_emg, peak_time_hammer)]
    delay_rows = calcular_delay_rows(peak_time_hammer, peak_value_hammer, extremes_time_emg, extremes_value_emg)

    valid_delays = list(filter(lambda delay: delay is not None, list(map(lambda delay_row:delay_row[4], delay_rows))))
    mean_delay = np.mean(valid_delays)

    # Crear una tabla con los resultados
    resultados_table = tabulate({
        # 'T EMG': peak_time_emg,
        # 'Max EMG': peak_value_emg,
        # 'T Hammer': peak_time_hammer,
        # 'Max Hammer': peak_value_hammer,
        # 'Delay': delay_rows  # Agregar la columna de Delay
        'T hammer': list(map(lambda delay_row:delay_row[0], delay_rows)),
        'Hammer peaks': list(map(lambda delay_row:delay_row[1], delay_rows)),
        'T EMG': list(map(lambda delay_row:delay_row[2], delay_rows)),
        'EMG peaks': list(map(lambda delay_row:delay_row[3], delay_rows)),
        'Delay': list(map(lambda delay_row:delay_row[4], delay_rows))
    }, headers='keys', tablefmt='pretty', showindex=False)

    # Agregar la media de los delays a la tabla
    resultados_table += f"\nMean delay: {mean_delay:.3f}s"

    # Crear una ventana para mostrar los resultados
    ventana_resultados = tk.Toplevel()
    ventana_resultados.title("Resultados de cálculos")

    # Crear un frame para contener la tabla y el scrollbar
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


# def actualizar_leyenda_completa():
#     global canvas, ax
#     ax.legend()
#     ax.set_title('Prueba grafica preprocesado')
#     ax.grid(True, linestyle='--', alpha=0.7)
#     canvas.draw_idle()


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

    peak_time_emg, peak_value_emg = calcular_maximos_emg(xlim)
    emg_peak_data.set_xdata(peak_time_emg)
    emg_peak_data.set_ydata(peak_value_emg)

    # ax.legend()
    canvas.draw_idle()


def close_gui(ventana):
    ventana.destroy()



ventana = tk.Tk()
ventana.title("GUI")
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
subMenu_filter.add_radiobutton(label='Original signal', variable=emg_filter, value='original', command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Default (high pass) filter', variable=emg_filter, value='default',
                               command=actualizar_emg_filter)
subMenu_filter.add_separator()
subMenu_filter.add_radiobutton(label='Savitzky–Golay filter', variable=emg_filter, value='savgol', command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='High pass filter', variable=emg_filter, value='high_pass', command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Low pass filter', variable=emg_filter, value='low_pass', command=actualizar_emg_filter)
subMenu_filter.add_radiobutton(label='Band pass filter', variable=emg_filter, value='band_pass', command=actualizar_emg_filter)

subMenu_filter.add_radiobutton(label='Chebyshev type2 filter', variable=emg_filter, value='Chebyshev2', command=actualizar_emg_filter)


ventana.mainloop()
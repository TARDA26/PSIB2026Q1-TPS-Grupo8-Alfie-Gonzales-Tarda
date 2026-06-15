"""
TPS - Procesamiento de Señales Biomédicas - ITBA 2025 Q2
EEG durante tareas aritméticas (EEGMAT - PhysioNet)

Parte 1: Carga del dataset y pre-procesamiento (filtros).

- Lee los EDF del dataset.
- Diseña el filtro pasa-banda Butterworth (0.5-35 Hz, ord. 4) + Notch
  (50 Hz, Q=30) y los aplica en cascada con fase cero (filtfilt).
- Genera el gráfico de Bode y la comparación cruda vs filtrada.

Las funciones de este módulo (load_edf, filtrar, FS, DATA_DIR) son
reutilizadas por las Partes 2, 3 y por el dashboard interactivo.
"""

import os
import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')  # para que no nos llene la consola con DeprecationWarnings

# Un toque más legible que el default
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10


# =============================================================================
# Constantes / paths
# =============================================================================
# Si moves la carpeta o cambias el dataset, lo único que hay que tocar es esto.
DATA_DIR = './muestras_EEG'
FS = 500       # Hz (lo dice el paper, es fijo para todo el dataset)
SEG_S = 60     # segundos por condición - alcanza para Welch y HRV


# =============================================================================
# 1) CARGA DE DATOS
# =============================================================================
def load_edf(path, seg_s=60, fs=500):
    """
    Lee un EDF y devuelve EEG (sin A2 ni ECG), nombres de canales y ECG por separado.
    Sólo tomamos los primeros seg_s segundos de cada señal para igualar duraciones
    (algunos archivos tienen un par de segundos extra).
    """
    f = pyedflib.EdfReader(path)
    labels = f.getSignalLabels()
    n_samples = int(seg_s * fs)

    # Filtramos canales: queremos los EEG, dejamos afuera ECG y la referencia A2.
    eeg_idx = [i for i, l in enumerate(labels)
               if 'EEG' in l and 'ECG' not in l and 'A2' not in l]
    ecg_idx = [i for i, l in enumerate(labels) if 'ECG' in l]

    eeg = np.array([f.readSignal(i)[:n_samples] for i in eeg_idx])
    ecg = f.readSignal(ecg_idx[0])[:n_samples] if ecg_idx else None
    # Sacamos el prefijo "EEG " de los nombres para que queden tipo "Fz", "Pz", etc.
    ch_names = [labels[i].replace('EEG ', '') for i in eeg_idx]

    f.close()
    return eeg, ch_names, ecg


def explorar_archivo(path):
    """Imprime info básica de un EDF (lo usamos para chusmear un sujeto)."""
    f = pyedflib.EdfReader(path)
    labels = f.getSignalLabels()
    fs_local = f.getSampleFrequencies()[0]
    n_samples = f.getNSamples()[0]
    f.close()

    print(f'=== {os.path.basename(path)} ===')
    print(f'  fs              : {fs_local} Hz')
    print(f'  duración total  : {n_samples / fs_local:.1f} s')
    print(f'  canales ({len(labels)}):')
    for i, l in enumerate(labels):
        print(f'    [{i:2d}] {l}')


# =============================================================================
# 2) FILTROS
# =============================================================================
# Banda EEG útil: 0.5 a 35 Hz (por debajo: deriva DC / sudor; por arriba: EMG).
# Notch en 50 Hz por la red eléctrica.
fL, fH = 0.5, 35.0
ORDER = 4
F_NOTCH = 50.0
Q_NOTCH = 30.0

# Pasa-banda Butterworth (suave en la banda, sin ripple)
b_bp, a_bp = signal.butter(ORDER, [fL / (FS / 2), fH / (FS / 2)], btype='bandpass')

# Notch IIR estándar
b_n, a_n = signal.iirnotch(F_NOTCH, Q_NOTCH, FS)

# Combinamos los dos en uno solo - así filtrar() hace una sola pasada filtfilt
# en vez de dos. La convolución de los coeficientes equivale a poner los filtros
# en cascada.
B_FILT = np.convolve(b_bp, b_n)
A_FILT = np.convolve(a_bp, a_n)


def filtrar(x):
    """Pasa-banda + notch en fase cero (filtfilt - aplica ida y vuelta)."""
    return signal.filtfilt(B_FILT, A_FILT, x)


def plot_respuesta_filtros():
    """Bode de los tres filtros - sirve para chequear que estén bien diseñados."""
    w_bp, h_bp = signal.freqz(b_bp, a_bp, fs=FS, worN=2048)
    w_n,  h_n  = signal.freqz(b_n,  a_n,  fs=FS, worN=2048)
    w_c,  h_c  = signal.freqz(B_FILT, A_FILT, fs=FS, worN=2048)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Diseño de Filtros Digitales', fontweight='bold')

    axes[0].plot(w_bp, 20 * np.log10(np.abs(h_bp)), color='steelblue', lw=2)
    axes[0].axvline(fL, color='r', ls='--', alpha=0.7, label=f'{fL} Hz')
    axes[0].axvline(fH, color='r', ls='--', alpha=0.7, label=f'{fH} Hz')
    axes[0].set(title='Butterworth pasa-banda (0.5-35 Hz, ord. 4)',
                xlabel='Frecuencia (Hz)', ylabel='|H(f)| (dB)',
                xlim=[0, 100], ylim=[-80, 5])
    axes[0].legend(); axes[0].grid(alpha=0.4)

    # Sumamos 1e-10 antes del log para evitar log(0) en la frecuencia notch
    axes[1].plot(w_n, 20 * np.log10(np.abs(h_n) + 1e-10), color='darkorange', lw=2)
    axes[1].axvline(50, color='r', ls='--', alpha=0.7, label='50 Hz')
    axes[1].set(title='Notch (50 Hz, Q=30)',
                xlabel='Frecuencia (Hz)', ylabel='|H(f)| (dB)',
                xlim=[0, 100], ylim=[-40, 5])
    axes[1].legend(); axes[1].grid(alpha=0.4)

    axes[2].plot(w_c, 20 * np.log10(np.abs(h_c) + 1e-10), color='forestgreen', lw=2)
    axes[2].set(title='Combinado (Butterworth + Notch)',
                xlabel='Frecuencia (Hz)', ylabel='|H(f)| (dB)',
                xlim=[0, 100], ylim=[-80, 5])
    axes[2].grid(alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_crudo_vs_filtrado(eeg_raw, ch_names, fz_i, pz_i):
    """Comparación visual cruda vs filtrada en Fz y Pz (primeros 10 s)."""
    raw_fz, raw_pz = eeg_raw[fz_i], eeg_raw[pz_i]
    filt_fz, filt_pz = filtrar(raw_fz), filtrar(raw_pz)

    t10 = np.linspace(0, 10, int(10 * FS))

    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    fig.suptitle('EEG crudo vs. filtrado - Subject00 baseline (primeros 10 s)',
                 fontweight='bold')

    axes[0, 0].plot(t10, raw_fz[:len(t10)], color='gray', lw=0.7)
    axes[0, 0].set(title='Fz - cruda', ylabel='µV')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t10, filt_fz[:len(t10)], color='steelblue', lw=0.8)
    axes[0, 1].set(title='Fz - filtrada (0.5-35 Hz + notch)')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t10, raw_pz[:len(t10)], color='gray', lw=0.7)
    axes[1, 0].set(title='Pz - cruda', ylabel='µV', xlabel='Tiempo (s)')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t10, filt_pz[:len(t10)], color='darkorange', lw=0.8)
    axes[1, 1].set(title='Pz - filtrada', xlabel='Tiempo (s)')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # El dataset ya viene pre-procesado con ICA, así que la diferencia es chica
    # pero igual queda como sanity check.
    print(f'RMS Fz cruda   : {np.sqrt(np.mean(raw_fz ** 2)):.2f} µV')
    print(f'RMS Fz filtrada: {np.sqrt(np.mean(filt_fz ** 2)):.2f} µV')


def plot_butterfly(eeg_raw, ch_names):
    """Butterfly de los 19 canales con un offset vertical para que no se pisen."""
    t10 = np.linspace(0, 10, int(10 * FS))
    offset = 100  # µV entre canal y canal
    colors = plt.cm.tab20(np.linspace(0, 1, len(ch_names)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Butterfly plot - 19 canales EEG (primeros 10 s)', fontweight='bold')

    for j in range(len(ch_names)):
        raw_ch = eeg_raw[j, :len(t10)]
        filt_ch = filtrar(eeg_raw[j])[:len(t10)]
        axes[0].plot(t10, raw_ch + j * offset, color=colors[j], lw=0.5, alpha=0.8)
        axes[1].plot(t10, filt_ch + j * offset, color=colors[j], lw=0.6, alpha=0.8)

    for ax, title in zip(axes, ['Cruda', 'Filtrada']):
        ax.set_title(title)
        ax.set_yticks([j * offset for j in range(len(ch_names))])
        ax.set_yticklabels(ch_names, fontsize=7)
        ax.set_xlabel('Tiempo (s)')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    # --- 1. Exploración de un EDF de ejemplo ---
    explorar_archivo(os.path.join(DATA_DIR, 'Subject00_1.edf'))

    # --- 2. Subject-info: separamos grupos G (buenos) y B (malos) ---
    df_info = pd.read_csv(os.path.join(DATA_DIR, 'subject-info.csv'))
    print(df_info)

    grupo_G = df_info[df_info['Count quality'] == 1]
    grupo_B = df_info[df_info['Count quality'] == 0]

    print(f'\nTotal sujetos    : {len(df_info)}')
    print(f'Grupo G (buenos) : N={len(grupo_G)}, '
          f'subtracciones {grupo_G["Number of subtractions"].mean():.1f} ± '
          f'{grupo_G["Number of subtractions"].std():.1f}')
    print(f'Grupo B (malos)  : N={len(grupo_B)}, '
          f'subtracciones {grupo_B["Number of subtractions"].mean():.1f} ± '
          f'{grupo_B["Number of subtractions"].std():.1f}')

    # --- 3. Filtros: Bode + comparación cruda vs filtrada ---
    plot_respuesta_filtros()

    eeg_raw, ch_names, _ = load_edf(os.path.join(DATA_DIR, 'Subject00_1.edf'),
                                     seg_s=60)
    fz_i = next(j for j, l in enumerate(ch_names) if 'Fz' in l)
    pz_i = next(j for j, l in enumerate(ch_names) if 'Pz' in l)

    plot_crudo_vs_filtrado(eeg_raw, ch_names, fz_i, pz_i)
    plot_butterfly(eeg_raw, ch_names)


if __name__ == '__main__':
    main()

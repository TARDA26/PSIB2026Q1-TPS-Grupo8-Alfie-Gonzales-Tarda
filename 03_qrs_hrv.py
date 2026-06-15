"""
TPS - Procesamiento de Señales Biomédicas - ITBA 2025 Q2
EEG durante tareas aritméticas (EEGMAT - PhysioNet)

Parte 3: ECG, detección QRS y HRV.

Los EDF traen además del EEG un canal ECG. Acá:
- Lo filtramos con un pasa-banda 1-40 Hz.
- Detectamos picos R con un Pan-Tompkins simplificado (derivar,
  cuadrar, suavizar, umbral adaptativo).
- A partir de los intervalos RR calculamos FC, SDNN, RMSSD y pNN50.
- Comparamos baseline vs. tarea y G vs. B con tests no paramétricos.

Reutiliza FS y DATA_DIR de la Parte 1.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import pyedflib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import mannwhitneyu

# Importamos la Parte 1 a mano (el módulo empieza con número)
_spec = importlib.util.spec_from_file_location(
    'parte1', os.path.join(os.path.dirname(__file__), '01_carga_y_filtrado.py'))
parte1 = importlib.util.module_from_spec(_spec)
sys.modules['parte1'] = parte1
_spec.loader.exec_module(parte1)

FS       = parte1.FS
DATA_DIR = parte1.DATA_DIR


# =============================================================================
# 1) ECG: filtrado, detección QRS y métricas HRV
# =============================================================================
def filtrar_ecg(x, fs=500):
    """
    Pasa-banda 1-40 Hz para ECG. Es la banda típica - saca deriva de
    línea de base y EMG / ruido de alta frecuencia.
    """
    b, a = signal.butter(3, [1.0 / (fs / 2), 40.0 / (fs / 2)],
                         btype='bandpass')
    return signal.filtfilt(b, a, x)


def detectar_qrs(ecg, fs=500, min_dist=0.3):
    """
    Detector de complejos QRS (Pan-Tompkins simplificado).

    Pasos:
      1. Derivar la señal (resalta las pendientes pronunciadas del QRS)
      2. Elevar al cuadrado (queda todo positivo y enfatiza los picos)
      3. Suavizar con ventana móvil de ~80 ms
      4. Buscar picos con umbral adaptativo (media + 0.5·sigma)

    min_dist = 0.3 s evita detectar dos picos del mismo QRS (asume
    FC máxima razonable de 200 bpm).
    """
    diff = np.diff(ecg)
    squared = diff ** 2
    win = int(0.08 * fs)  # 80 ms - cubre un QRS típico
    smooth = np.convolve(squared, np.ones(win) / win, mode='same')

    thresh = np.mean(smooth) + 0.5 * np.std(smooth)
    peaks, _ = signal.find_peaks(smooth, height=thresh,
                                  distance=int(min_dist * fs))
    return peaks


def calcular_hrv(r_peaks, fs=500):
    """
    Métricas temporales de HRV a partir de los picos R.

    - FC media (bpm)
    - RR medio (ms)
    - SDNN: desvío estándar de los RR -> variabilidad total
    - RMSSD: raíz del promedio de las diferencias RR consecutivas
             al cuadrado -> variabilidad de corto plazo (parasimpático)
    - pNN50: % de pares consecutivos que difieren más de 50 ms
    """
    if len(r_peaks) < 3:
        return {}

    rr = np.diff(r_peaks) / fs * 1000   # paso a ms
    # Filtro fisiológico: RR razonables van entre ~300 ms (200 bpm)
    # y ~1500 ms (40 bpm). Lo de afuera son artefactos.
    rr = rr[(rr > 300) & (rr < 1500)]
    if len(rr) < 3:
        return {}

    return {
        'mean_hr_bpm': 60000 / np.mean(rr),
        'mean_rr_ms':  np.mean(rr),
        'sdnn_ms':     np.std(rr),
        'rmssd_ms':    np.sqrt(np.mean(np.diff(rr) ** 2)),
        'pnn50_pct':   np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100,
        'n_beats':     len(rr) + 1,
    }


def load_ecg(path, fs=500, seg_s=60):
    """Lee y filtra el canal ECG de un EDF. Devuelve None si no hay ECG."""
    f = pyedflib.EdfReader(path)
    labels = f.getSignalLabels()
    idx = next((i for i, l in enumerate(labels) if 'ECG' in l), None)
    if idx is None:
        f.close()
        return None
    x = f.readSignal(idx)[:int(seg_s * fs)]
    f.close()
    return filtrar_ecg(x, fs)


def plot_qrs_subject(ecg_b, ecg_t, r_b, r_t, hrv_b, hrv_t):
    """ECG + picos detectados (15 s) y tachogramas + PSD del HRV."""
    n15 = int(15 * FS)
    t_b15 = np.linspace(0, 15, n15)
    t_t15 = np.linspace(0, 15, n15)
    r_b15 = r_b[r_b < n15]
    r_t15 = r_t[r_t < n15]

    rr_b = np.diff(r_b) / FS * 1000
    rr_t = np.diff(r_t) / FS * 1000
    rr_b = rr_b[(rr_b > 300) & (rr_b < 1500)]
    rr_t = rr_t[(rr_t > 300) & (rr_t < 1500)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle('Detección QRS y HRV - Subject00', fontweight='bold')

    axes[0, 0].plot(t_b15, ecg_b[:n15], color='steelblue', lw=0.8, label='ECG')
    axes[0, 0].plot(r_b15 / FS, ecg_b[r_b15], 'rv', ms=10,
                    label=f'{len(r_b)} picos R')
    axes[0, 0].set(title='ECG BASELINE - 15 s', ylabel='µV')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t_t15, ecg_t[:n15], color='darkorange', lw=0.8,
                    label='ECG')
    axes[0, 1].plot(r_t15 / FS, ecg_t[r_t15], 'rv', ms=10,
                    label=f'{len(r_t)} picos R')
    axes[0, 1].set(title='ECG TAREA - 15 s')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Tachograma
    axes[1, 0].plot(rr_b, color='steelblue', lw=1.5,
                    label=f'Baseline | SDNN={hrv_b["sdnn_ms"]:.1f} ms')
    axes[1, 0].plot(rr_t, color='darkorange', lw=1.5,
                    label=f'Tarea    | SDNN={hrv_t["sdnn_ms"]:.1f} ms')
    axes[1, 0].set(title='Tachograma (intervalos RR)',
                   xlabel='Latido #', ylabel='RR (ms)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # PSD del tachograma para HRV frecuencial (LF y HF)
    if len(rr_b) >= 8 and len(rr_t) >= 8:
        # El tachograma es irregular en tiempo; asumimos fs ~= 1/RR_medio
        # (sirve para visualizar las bandas LF/HF).
        np_seg = min(len(rr_b), len(rr_t), 8)
        fs_rr_b = 1 / (np.mean(rr_b) / 1000)
        fs_rr_t = 1 / (np.mean(rr_t) / 1000)
        fr_hrv_b, psd_hrv_b = signal.welch(rr_b, fs=fs_rr_b, nperseg=np_seg)
        fr_hrv_t, psd_hrv_t = signal.welch(rr_t, fs=fs_rr_t, nperseg=np_seg)

        axes[1, 1].plot(fr_hrv_b, psd_hrv_b, color='steelblue', lw=1.5,
                        label='Baseline')
        axes[1, 1].plot(fr_hrv_t, psd_hrv_t, color='darkorange', lw=1.5,
                        label='Tarea')
        # LF (simpático + parasimpático), HF (vagal)
        axes[1, 1].axvspan(0.04, 0.15, alpha=0.15, color='blue',
                           label='LF (0.04-0.15 Hz)')
        axes[1, 1].axvspan(0.15, 0.40, alpha=0.15, color='green',
                           label='HF (0.15-0.40 Hz)')
        axes[1, 1].set(title='PSD tachograma - HRV frecuencial',
                       xlabel='Frecuencia (Hz)', ylabel='PSD (ms²/Hz)')
        axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def hrv_todos_los_sujetos(df_info):
    """Loopea por todos los sujetos y arma un DF con las métricas HRV."""
    print('Calculando HRV para todos los sujetos...')
    hrv_results = []

    for _, row in df_info.iterrows():
        subj = row['Subject']
        grp = row['Count quality']
        for cond, suffix in [('baseline', '_1'), ('task', '_2')]:
            path = os.path.join(DATA_DIR, f'{subj}{suffix}.edf')
            if not os.path.exists(path):
                continue
            try:
                ecg = load_ecg(path)
                if ecg is None:
                    continue
                peaks = detectar_qrs(ecg)
                hrv = calcular_hrv(peaks)
                hrv.update({'subject': subj, 'group': grp, 'condition': cond})
                hrv_results.append(hrv)
            except Exception:
                # Si falla un sujeto, seguimos con los demás
                pass

    df_hrv = pd.DataFrame(hrv_results)
    print(f'Registros HRV calculados: {len(df_hrv)}')
    return df_hrv


def plot_hrv_boxplots_y_stats(df_hrv):
    """Boxplots de las 4 métricas HRV + tests entre grupos."""
    metrics = [
        ('mean_hr_bpm', 'FC media (bpm)'),
        ('sdnn_ms',     'SDNN (ms)'),
        ('rmssd_ms',    'RMSSD (ms)'),
        ('pnn50_pct',   'pNN50 (%)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.suptitle('HRV - Grupo G (verde) vs. B (rojo) | baseline vs. tarea',
                 fontweight='bold')

    for ax, (met, label) in zip(axes, metrics):
        data = [df_hrv[(df_hrv['group'] == g) &
                       (df_hrv['condition'] == c)][met].dropna()
                for g, c in [(1, 'baseline'), (1, 'task'),
                             (0, 'baseline'), (0, 'task')]]
        bp = ax.boxplot(data,
                        labels=['G-Base', 'G-Tarea', 'B-Base', 'B-Tarea'],
                        patch_artist=True)
        for patch, col in zip(
                bp['boxes'],
                ['lightgreen', 'darkgreen', 'lightsalmon', 'darkred']):
            patch.set_facecolor(col); patch.set_alpha(0.75)

        # Mann-Whitney G vs B en tarea
        if len(data[1]) > 2 and len(data[3]) > 2:
            _, p = mannwhitneyu(data[1], data[3], alternative='two-sided')
        else:
            p = 1.0
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'ns')
        ax.text(0.5, 0.97, f'G vs B (tarea): {sig}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8, color='purple', fontweight='bold')
        ax.set_title(label); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Tabla de resumen en consola
    print('\n=== RESUMEN HRV ===')
    for met, label in metrics:
        for cond in ['baseline', 'task']:
            Gv = df_hrv[(df_hrv['group'] == 1) &
                        (df_hrv['condition'] == cond)][met].dropna()
            Bv = df_hrv[(df_hrv['group'] == 0) &
                        (df_hrv['condition'] == cond)][met].dropna()
            if len(Gv) > 2 and len(Bv) > 2:
                _, p = mannwhitneyu(Gv, Bv, alternative='two-sided')
                sig = ('***' if p < 0.001 else '**' if p < 0.01
                       else '*' if p < 0.05 else 'ns')
                print(f'{label:18s} | {cond:8s} | '
                      f'G: {Gv.mean():.1f}±{Gv.std():.1f} | '
                      f'B: {Bv.mean():.1f}±{Bv.std():.1f} | {sig}')


# =============================================================================
# MAIN
# =============================================================================
def main():
    df_info = pd.read_csv(os.path.join(DATA_DIR, 'subject-info.csv'))

    # --- ECG y HRV en Subject00 (ejemplo visual) ---
    ecg_b = load_ecg(os.path.join(DATA_DIR, 'Subject00_1.edf'))
    ecg_t = load_ecg(os.path.join(DATA_DIR, 'Subject00_2.edf'))
    r_b = detectar_qrs(ecg_b)
    r_t = detectar_qrs(ecg_t)
    hrv_b = calcular_hrv(r_b)
    hrv_t = calcular_hrv(r_t)

    print('=== Subject00 ===')
    print(f'Baseline: {len(r_b)} latidos | FC: {hrv_b["mean_hr_bpm"]:.1f} bpm'
          f' | SDNN: {hrv_b["sdnn_ms"]:.1f} ms')
    print(f'Tarea:    {len(r_t)} latidos | FC: {hrv_t["mean_hr_bpm"]:.1f} bpm'
          f' | SDNN: {hrv_t["sdnn_ms"]:.1f} ms')

    plot_qrs_subject(ecg_b, ecg_t, r_b, r_t, hrv_b, hrv_t)

    # --- HRV para todos los sujetos ---
    df_hrv = hrv_todos_los_sujetos(df_info)
    plot_hrv_boxplots_y_stats(df_hrv)


if __name__ == '__main__':
    main()

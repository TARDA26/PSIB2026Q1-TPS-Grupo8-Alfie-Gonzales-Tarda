"""
TPS - Procesamiento de Señales Biomédicas - ITBA 2025 Q2
EEG durante tareas aritméticas (EEGMAT - PhysioNet)

Dashboard interactivo con Streamlit.

Tabs: Señal | Filtro | Grupos | HRV
(reflejan el pipeline real del informe: filtros, Welch + bandas,
comparación G vs B y HRV)

Para correrlo:
    streamlit run 04_dashboard.py
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import trapezoid   # reemplaza np.trapz (eliminado en numpy 2.0)


# =============================================================================
# Carga de los módulos vecinos (empiezan con dígito)
# =============================================================================
def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p1 = _load('parte1', '01_carga_y_filtrado.py')
p3 = _load('parte3', '03_qrs_hrv.py')

# Fijamos DATA_DIR como ruta absoluta basada en este archivo, así no
# depende del cwd desde donde se ejecute streamlit.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'muestras_EEG')
p1.DATA_DIR = DATA_DIR
p3.DATA_DIR = DATA_DIR
FS = p1.FS


# =============================================================================
# Config general
# =============================================================================
st.set_page_config(page_title='EEGMAT - TP Señales', layout='wide')
plt.rcParams['figure.dpi'] = 110


# =============================================================================
# Cache de datos pesados
# =============================================================================
@st.cache_data
def cargar_info():
    return pd.read_csv(os.path.join(DATA_DIR, 'subject-info.csv'))


@st.cache_data
def cargar_eeg(subject, condition):
    suffix = '_1' if condition == 'baseline' else '_2'
    path = os.path.join(DATA_DIR, f'{subject}{suffix}.edf')
    eeg, chs, _ = p1.load_edf(path, seg_s=60)
    return eeg, chs


@st.cache_data
def cargar_ecg(subject, condition):
    suffix = '_1' if condition == 'baseline' else '_2'
    path = os.path.join(DATA_DIR, f'{subject}{suffix}.edf')
    return p3.load_ecg(path)


# =============================================================================
# Sidebar - controles globales
# =============================================================================
df_info = cargar_info()
subjects = df_info['Subject'].tolist()

with st.sidebar:
    st.title('EEGMAT')
    subject = st.selectbox('Sujeto', subjects, index=0)
    condition = st.radio('Condición', ['baseline', 'task'], horizontal=True)

eeg, ch_names = cargar_eeg(subject, condition)


# =============================================================================
# Tabs
# =============================================================================
tab_sig, tab_filt, tab_grp, tab_hrv = st.tabs(
    ['Señal', 'Filtro', 'Grupos', 'HRV'])


# -----------------------------------------------------------------------------
# Tab 1: Señal - explorador básico
# -----------------------------------------------------------------------------
with tab_sig:
    ch = st.selectbox('Canal', ch_names,
                      index=ch_names.index('Fz') if 'Fz' in ch_names else 0)
    ci = ch_names.index(ch)
    x = p1.filtrar(eeg[ci])
    t = np.linspace(0, 60, len(x))

    fig, ax = plt.subplots(3, 1, figsize=(11, 7))

    ax[0].plot(t, x, lw=0.5, color='steelblue')
    ax[0].set(xlim=[0, 60], ylabel='µV', title=f'{ch} - filtrada')
    ax[0].grid(alpha=0.3)

    fr, psd = signal.welch(x, fs=FS, nperseg=1024, noverlap=512)
    ax[1].semilogy(fr, psd, color='darkorange')
    ax[1].set(xlim=[0.5, 35], xlabel='Hz', ylabel='PSD', title='Welch')
    ax[1].grid(alpha=0.3)

    f_s, t_s, Sxx = signal.spectrogram(x, fs=FS, nperseg=512, noverlap=400)
    mask = f_s <= 35
    ax[2].pcolormesh(t_s, f_s[mask], 10 * np.log10(Sxx[mask] + 1e-12),
                     cmap='RdYlBu_r', shading='auto', vmin=-20, vmax=30)
    ax[2].set(xlabel='Tiempo (s)', ylabel='Hz', title='Espectrograma')

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# Tab 2: Filtro - jugar con los parámetros
# -----------------------------------------------------------------------------
with tab_filt:
    c1, c2 = st.columns(2)
    with c1:
        fL = st.slider('fL (Hz)', 0.1, 5.0, 0.5, 0.1)
        fH = st.slider('fH (Hz)', 10.0, 60.0, 35.0, 1.0)
    with c2:
        order = st.slider('Orden Butterworth', 1, 8, 4)
        Q = st.slider('Q notch', 5, 60, 30)

    b_bp, a_bp = signal.butter(order, [fL / (FS / 2), fH / (FS / 2)],
                                btype='bandpass')
    b_n, a_n = signal.iirnotch(50.0, Q, FS)
    B = np.convolve(b_bp, b_n); A = np.convolve(a_bp, a_n)

    w, h = signal.freqz(B, A, fs=FS, worN=2048)

    ci_demo = ch_names.index('Fz') if 'Fz' in ch_names else 0
    x_demo = signal.filtfilt(B, A, eeg[ci_demo])
    t10 = np.linspace(0, 10, int(10 * FS))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(w, 20 * np.log10(np.abs(h) + 1e-10), color='forestgreen', lw=2)
    ax[0].axvspan(fL, fH, alpha=0.1, color='green')
    ax[0].set(xlim=[0, 80], ylim=[-80, 5], xlabel='Hz', ylabel='dB',
              title='Respuesta combinada')
    ax[0].grid(alpha=0.3)

    ax[1].plot(t10, x_demo[:len(t10)], color='steelblue', lw=0.7)
    ax[1].set(xlabel='Tiempo (s)', ylabel='µV', title='Fz filtrada (10 s)')
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# Tab 3: Grupos - comparación G vs B
# -----------------------------------------------------------------------------
# Precalculamos potencia relativa de TODAS las bandas en TODOS los canales para
# TODOS los sujetos una sola vez. Es ~30-60 s en la primera carga; después
# cambiar canal/banda es instantáneo.
@st.cache_data(show_spinner='Calculando potencia por bandas para los 36 sujetos...')
def potencia_todos():
    BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8),
             'Alpha': (8, 13), 'Beta': (13, 30)}
    out = []
    errores = []
    for _, row in df_info.iterrows():
        subj = row['Subject']; grp = row['Count quality']
        for cond, suffix in [('baseline', '_1'), ('task', '_2')]:
            path = os.path.join(DATA_DIR, f'{subj}{suffix}.edf')
            if not os.path.exists(path):
                errores.append(f'no existe: {path}')
                continue
            try:
                eeg_s, chs, _ = p1.load_edf(path, seg_s=60)
                # Filtramos los 19 canales de una y reutilizamos en cada banda
                eeg_f = [p1.filtrar(c) for c in eeg_s]
                for ci, ch in enumerate(chs):
                    fr, psd = signal.welch(eeg_f[ci], fs=FS,
                                           nperseg=1024, noverlap=512)
                    idx_tot = (fr >= 0.5) & (fr <= 35)
                    total = trapezoid(psd[idx_tot], fr[idx_tot])
                    if total <= 0:
                        continue
                    for band, (flo, fhi) in BANDS.items():
                        idx = (fr >= flo) & (fr <= fhi)
                        p_rel = trapezoid(psd[idx], fr[idx]) / total * 100
                        out.append({'group': grp, 'condition': cond,
                                    'channel': ch, 'band': band,
                                    'p_rel': p_rel})
            except Exception as e:
                errores.append(f'{subj}{suffix}: {e}')
    return pd.DataFrame(out), errores


with tab_grp:
    c1, c2 = st.columns(2)
    with c1:
        canal_g = st.selectbox('Canal', ['Fz', 'Cz', 'Pz', 'O1', 'O2',
                                          'F3', 'F4'], key='canal_g')
    with c2:
        banda_g = st.selectbox('Banda', ['Delta', 'Theta', 'Alpha', 'Beta'],
                                index=2)

    df_all, errores = potencia_todos()

    if len(df_all) == 0:
        st.error('No se pudo calcular ningún sujeto. '
                 'Verificá que la carpeta muestras_EEG/ esté al lado del script.')
        if errores:
            with st.expander('Detalle de errores'):
                for e in errores[:20]:
                    st.text(e)
    else:
        df_g = df_all[(df_all['channel'] == canal_g) &
                      (df_all['band'] == banda_g)]
        data = [df_g[(df_g['group'] == g) &
                     (df_g['condition'] == c)]['p_rel'].values
                for g, c in [(1, 'baseline'), (1, 'task'),
                             (0, 'baseline'), (0, 'task')]]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bp = ax.boxplot(data,
                        labels=['G-Base', 'G-Tarea', 'B-Base', 'B-Tarea'],
                        patch_artist=True, widths=0.55)
        for patch, color in zip(
                bp['boxes'],
                ['lightgreen', 'darkgreen', 'lightsalmon', 'darkred']):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set(ylabel='Potencia relativa (%)',
               title=f'{canal_g} - {banda_g}')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# Tab 4: HRV
# -----------------------------------------------------------------------------
with tab_hrv:
    ecg = cargar_ecg(subject, condition)
    if ecg is None:
        st.warning('Este sujeto no tiene canal ECG.')
    else:
        r = p3.detectar_qrs(ecg)
        hrv = p3.calcular_hrv(r)

        m1, m2, m3 = st.columns(3)
        m1.metric('FC (bpm)',   f'{hrv["mean_hr_bpm"]:.0f}')
        m2.metric('SDNN (ms)',  f'{hrv["sdnn_ms"]:.1f}')
        m3.metric('RMSSD (ms)', f'{hrv["rmssd_ms"]:.1f}')

        n15 = int(15 * FS)
        t15 = np.linspace(0, 15, n15)
        r15 = r[r < n15]
        rr = np.diff(r) / FS * 1000
        rr = rr[(rr > 300) & (rr < 1500)]

        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5))
        ax[0].plot(t15, ecg[:n15], color='steelblue', lw=0.8)
        ax[0].plot(r15 / FS, ecg[r15], 'rv', ms=8)
        ax[0].set(xlabel='Tiempo (s)', ylabel='µV',
                  title=f'ECG ({len(r)} latidos en 60 s)')
        ax[0].grid(alpha=0.3)

        ax[1].plot(rr, color='darkorange', lw=1.5, marker='o', ms=3)
        ax[1].set(xlabel='Latido #', ylabel='RR (ms)', title='Tachograma')
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

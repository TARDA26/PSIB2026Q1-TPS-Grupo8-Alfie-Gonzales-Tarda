"""
TPS - Procesamiento de Señales Biomédicas - ITBA 2025 Q2
EEG durante tareas aritméticas (EEGMAT - PhysioNet)

Parte 2: Análisis espectral.

- PSD por Welch + espectrograma (Subject00 de ejemplo).
- Potencia relativa por banda (delta/theta/alpha/beta) para todos los
  sujetos.
- Tests estadísticos: Wilcoxon (efecto de la tarea, pares baseline/tarea)
  y Mann-Whitney U (comparación Grupo G vs. Grupo B).
- Correlación de Spearman entre el score aritmético y la potencia
  relativa por banda durante la tarea.

Reutiliza load_edf, filtrar, FS y DATA_DIR de la Parte 1.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import trapezoid   # reemplaza np.trapz (eliminado en numpy 2.0)
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

# Carga del módulo de la Parte 1 (empieza con dígito, no entra con import)
_spec = importlib.util.spec_from_file_location(
    'parte1', os.path.join(os.path.dirname(__file__), '01_carga_y_filtrado.py'))
parte1 = importlib.util.module_from_spec(_spec)
sys.modules['parte1'] = parte1
_spec.loader.exec_module(parte1)

load_edf = parte1.load_edf
filtrar  = parte1.filtrar
FS       = parte1.FS
DATA_DIR = parte1.DATA_DIR


# =============================================================================
# Bandas EEG
# =============================================================================
# Límites clásicos. Beta lo cortamos en 30 Hz porque el filtro corta en 35.
BANDS = {
    'Delta (0.5-4 Hz)': (0.5,  4.0),
    'Theta (4-8 Hz)':   (4.0,  8.0),
    'Alpha (8-13 Hz)':  (8.0, 13.0),
    'Beta  (13-30 Hz)': (13.0, 30.0),
}
BAND_COLORS = ['navy', 'royalblue', 'forestgreen', 'firebrick']


def sig_label(p):
    """Marcador estándar de significancia."""
    return ('***' if p < 0.001 else '**' if p < 0.01
            else '*' if p < 0.05 else 'ns')


# =============================================================================
# 1) PSD WELCH + ESPECTROGRAMA
# =============================================================================
def welch_psd(x, fs=500, nperseg=1024):
    """
    PSD por Welch con ventana de Hann y 50% solapamiento.
    nperseg=1024 a fs=500 da una resolución de ~0.5 Hz.
    """
    return signal.welch(x, fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, window='hann',
                        scaling='density')


def band_power(psd, freqs, fmin, fmax):
    """Potencia absoluta en una banda (integral trapezoidal bajo la PSD)."""
    idx = (freqs >= fmin) & (freqs <= fmax)
    return trapezoid(psd[idx], freqs[idx])


def plot_psd_y_espectrograma(eeg_b_f, eeg_t_f, ch_names):
    """PSD + espectrograma para Fz, Pz y O1 (baseline vs tarea)."""
    chs_plot = [
        (next(j for j, l in enumerate(ch_names) if 'Fz' in l), 'Fz'),
        (next(j for j, l in enumerate(ch_names) if 'Pz' in l), 'Pz'),
        (next(j for j, l in enumerate(ch_names) if 'O1' in l), 'O1'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle('PSD Welch + espectrograma - Subject00 | '
                 'baseline (azul) vs. tarea (rojo)', fontweight='bold')

    for row, (ci, name) in enumerate(chs_plot):
        xb, xt = eeg_b_f[ci], eeg_t_f[ci]
        fr_b, psd_b = welch_psd(xb)
        fr_t, psd_t = welch_psd(xt)

        # --- col 0: PSD en log ---
        axes[row, 0].semilogy(fr_b, psd_b, color='steelblue',
                              lw=1.5, label='Baseline')
        axes[row, 0].semilogy(fr_t, psd_t, color='firebrick',
                              lw=1.5, alpha=0.85, label='Tarea')
        # Sombreado de cada banda
        for (flo, fhi), bc in zip(BANDS.values(), BAND_COLORS):
            axes[row, 0].axvspan(flo, fhi, alpha=0.07, color=bc)
        axes[row, 0].set(xlim=[0.5, 35], ylabel=f'{name}\nPSD (µV²/Hz)')
        if row == 0:
            axes[row, 0].set_title('PSD Welch')
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(alpha=0.3)

        # --- col 1 y 2: espectrogramas ---
        for col_i, (sig_x, title_suffix) in enumerate(
                [(xb, 'BASELINE'), (xt, 'TAREA')]):
            f_s, t_s, Sxx = signal.spectrogram(
                sig_x, fs=FS, nperseg=512, noverlap=400,
                window='hann', scaling='density')
            mask = f_s <= 35
            im = axes[row, col_i + 1].pcolormesh(
                t_s, f_s[mask], 10 * np.log10(Sxx[mask] + 1e-12),
                cmap='RdYlBu_r', shading='auto', vmin=-20, vmax=30)
            axes[row, col_i + 1].set(ylim=[0.5, 35], ylabel='Freq (Hz)')
            if row == 0:
                axes[row, col_i + 1].set_title(
                    f'Espectrograma - {title_suffix}')
            plt.colorbar(im, ax=axes[row, col_i + 1], label='dB')

    for ax in axes[-1, :]:
        ax.set_xlabel('Tiempo (s)')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 2) POTENCIA POR BANDAS - TODOS LOS SUJETOS
# =============================================================================
def calcular_potencias_todos(df_info):
    """
    Loopea por todos los sujetos y todas las condiciones, y arma un
    DataFrame con la potencia absoluta y relativa de cada banda por canal.

    Devuelve un DF largo (un row por sujeto/condición/canal/banda).
    """
    subjects = df_info['Subject'].tolist()
    groups   = df_info['Count quality'].tolist()

    print('Calculando potencia por bandas para todos los sujetos...')
    results = []

    for subj, grp in zip(subjects, groups):
        for cond, suffix in [('baseline', '_1'), ('task', '_2')]:
            path = os.path.join(DATA_DIR, f'{subj}{suffix}.edf')
            if not os.path.exists(path):
                continue
            try:
                eeg, chs, _ = load_edf(path, seg_s=60)
                eeg_f = np.array([filtrar(ch) for ch in eeg])
                for ci, ch in enumerate(chs):
                    fr, psd = welch_psd(eeg_f[ci])
                    total_p = band_power(psd, fr, 0.5, 35.0)
                    for band, (flo, fhi) in BANDS.items():
                        bp_abs = band_power(psd, fr, flo, fhi)
                        # Potencia relativa: % del total dentro de 0.5-35 Hz
                        bp_rel = bp_abs / total_p * 100 if total_p > 0 else 0
                        results.append({
                            'subject': subj, 'group': grp,
                            'condition': cond, 'channel': ch,
                            'band': band,
                            'power_abs': bp_abs, 'power_rel': bp_rel,
                        })
            except Exception as e:
                # Algunos archivos pueden tener corrupción menor, no rompemos
                # todo el loop por uno solo.
                print(f'  Error {subj}{suffix}: {e}')

    df_res = pd.DataFrame(results)
    print(f'Registros calculados: {len(df_res)}')
    return df_res


def plot_boxplots_bandas(df_res, channels_plot=('Fz', 'Pz', 'O1')):
    """Boxplots de potencia relativa por banda - G vs B, baseline vs tarea."""
    band_list = list(BANDS.keys())

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    fig.suptitle('Potencia relativa por banda\n'
                 'Grupo G (verde) vs. B (rojo) | baseline vs. tarea',
                 fontweight='bold')

    for col, ch in enumerate(channels_plot):
        df_ch = df_res[df_res['channel'] == ch]
        for row, band in enumerate(band_list):
            ax = axes[row, col]
            df_band = df_ch[df_ch['band'] == band]
            data = [
                df_band[(df_band['group'] == g) &
                        (df_band['condition'] == c)]['power_rel'].values
                for g, c in [(1, 'baseline'), (1, 'task'),
                             (0, 'baseline'), (0, 'task')]
            ]

            bp = ax.boxplot(
                data,
                labels=['G-Base', 'G-Tarea', 'B-Base', 'B-Tarea'],
                patch_artist=True, widths=0.6)
            for patch, color in zip(
                    bp['boxes'],
                    ['lightgreen', 'darkgreen', 'lightsalmon', 'darkred']):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            if col == 0:
                ax.set_ylabel('Potencia relativa (%)')
            ax.set_title(f'{ch} - {band.split(" ")[0]}')
            ax.grid(alpha=0.3, axis='y')

            # Test Mann-Whitney G vs B en condición tarea
            Gt, Bt = data[1], data[3]
            if len(Gt) > 3 and len(Bt) > 3:
                _, p = mannwhitneyu(Gt, Bt, alternative='two-sided')
                ax.text(0.5, 0.97,
                        f'G vs B (tarea): {sig_label(p)} (p={p:.3f})',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=7, color='purple', fontweight='bold')

    plt.tight_layout()
    plt.show()


def stats_efecto_tarea_y_grupos(df_res,
                                key_channels=('Fz', 'Cz', 'Pz', 'O1',
                                              'O2', 'F3', 'F4')):
    """
    Para cada canal "interesante" y cada banda:
      - Wilcoxon pareado baseline vs tarea (efecto de la tarea)
      - Mann-Whitney G vs B durante la tarea (diferencia entre grupos)
    """
    band_list = list(BANDS.keys())
    stat_results = []

    print(f'{"Canal":5s} | {"Banda":20s} | '
          f'{"Base->Tarea":12s} | {"G vs B tarea":12s}')
    print('-' * 60)

    for ch in key_channels:
        df_ch = df_res[df_res['channel'] == ch]
        for band in band_list:
            df_b = df_ch[df_ch['band'] == band]
            base_all = df_b[df_b['condition'] == 'baseline']['power_rel'].values
            task_all = df_b[df_b['condition'] == 'task']['power_rel'].values
            Gt = df_b[(df_b['group'] == 1) &
                      (df_b['condition'] == 'task')]['power_rel'].values
            Bt = df_b[(df_b['group'] == 0) &
                      (df_b['condition'] == 'task')]['power_rel'].values

            # Wilcoxon necesita pares - cortamos al mínimo común
            L = min(len(base_all), len(task_all))
            try:
                _, p_task = wilcoxon(base_all[:L], task_all[:L])
            except Exception:
                p_task = 1.0
            if len(Gt) > 3 and len(Bt) > 3:
                _, p_grp = mannwhitneyu(Gt, Bt, alternative='two-sided')
            else:
                p_grp = 1.0

            sig_t, sig_g = sig_label(p_task), sig_label(p_grp)
            # Sólo printeamos los significativos para no llenar la consola
            if sig_t != 'ns' or sig_g != 'ns':
                print(f'{ch:5s} | {band:20s} | '
                      f'{sig_t:12s} | {sig_g:12s}')

            stat_results.append({
                'channel': ch, 'band': band,
                'p_task': p_task, 'sig_task': sig_t,
                'p_group': p_grp, 'sig_group': sig_g,
                'mean_G_task': Gt.mean() if len(Gt) > 0 else np.nan,
                'mean_B_task': Bt.mean() if len(Bt) > 0 else np.nan,
            })

    return pd.DataFrame(stat_results)


# =============================================================================
# 3) CORRELACIONES SCORE vs. EEG
# =============================================================================
def correlaciones_score_eeg(df_res, df_info,
                            key_channels=('Fz', 'Cz', 'Pz', 'O1',
                                          'O2', 'F3', 'F4')):
    """
    Spearman entre el # de sustracciones de cada sujeto y la potencia
    relativa por banda (sólo en condición tarea).
    """
    band_list = list(BANDS.keys())
    corr_all = []

    for ch in key_channels:
        df_ch_task = df_res[(df_res['channel'] == ch) &
                            (df_res['condition'] == 'task')]
        for band in band_list:
            df_band = df_ch_task[df_ch_task['band'] == band]
            df_merged = df_band.merge(
                df_info[['Subject', 'Number of subtractions']],
                left_on='subject', right_on='Subject')
            x = df_merged['Number of subtractions'].values
            y = df_merged['power_rel'].values
            if len(x) > 5:
                r, p = spearmanr(x, y)
                corr_all.append({
                    'channel': ch, 'band': band,
                    'r': r, 'p': p, 'sig': sig_label(p),
                })

    return pd.DataFrame(corr_all)


def plot_correlaciones(df_res, df_info, df_corr,
                       key_channels=('Fz', 'Cz', 'Pz', 'O1',
                                     'O2', 'F3', 'F4')):
    """Scatter por banda en Fz + heatmap del r de Spearman por canal/banda."""
    band_list = list(BANDS.keys())
    df_task_fz = df_res[(df_res['channel'] == 'Fz') &
                        (df_res['condition'] == 'task')]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Correlación de Spearman: score aritmético vs. '
                 'potencia relativa (tarea)', fontweight='bold')

    # Scatter por banda en Fz, coloreado por grupo
    for i, (band, color) in enumerate(zip(band_list, BAND_COLORS)):
        ax = fig.add_subplot(2, 4, i + 1)
        df_band = df_task_fz[df_task_fz['band'] == band].merge(
            df_info[['Subject', 'Number of subtractions', 'Count quality']],
            left_on='subject', right_on='Subject')
        x = df_band['Number of subtractions'].values
        y = df_band['power_rel'].values
        grp_col = ['forestgreen' if g == 1 else 'firebrick'
                   for g in df_band['Count quality'].values]
        ax.scatter(x, y, c=grp_col, alpha=0.75,
                   edgecolors='gray', lw=0.5, s=60)
        if len(x) > 3:
            # Recta de regresión a ojo
            z = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color='black', lw=1.5, ls='--')
            r, p_val = spearmanr(x, y)
            ax.set_title(f'Fz - {band.split(" ")[0]}\n'
                         f'r={r:.2f}, p={p_val:.3f} {sig_label(p_val)}',
                         fontsize=9)
        ax.set(xlabel='Score (# sustracciones)',
               ylabel='Potencia relativa (%)')
        ax.grid(alpha=0.3)

    # Heatmap por canal y banda
    ax_hm = fig.add_subplot(2, 4, (5, 8))
    if len(df_corr) > 0:
        pivot = df_corr.pivot(index='band', columns='channel', values='r')
        pivot = pivot.reindex(
            columns=[c for c in key_channels if c in pivot.columns])
        im = ax_hm.imshow(pivot.values, cmap='RdBu_r',
                          vmin=-0.6, vmax=0.6, aspect='auto')
        ax_hm.set_xticks(range(pivot.shape[1]))
        ax_hm.set_xticklabels(pivot.columns.tolist(), fontsize=9)
        ax_hm.set_yticks(range(len(band_list)))
        ax_hm.set_yticklabels([b.split(' ')[0] for b in band_list],
                              fontsize=9)
        plt.colorbar(im, ax=ax_hm, label='r de Spearman')
        # Marcamos con asterisco las celdas significativas
        for row_i in range(pivot.shape[0]):
            for col_i in range(pivot.shape[1]):
                try:
                    p_val = df_corr[
                        (df_corr['band'] == band_list[row_i]) &
                        (df_corr['channel'] == pivot.columns[col_i])
                    ]['p'].values[0]
                    if p_val < 0.05:
                        ax_hm.text(col_i, row_i, '*',
                                   ha='center', va='center', fontsize=14,
                                   color='white'
                                   if abs(pivot.values[row_i, col_i]) > 0.3
                                   else 'black')
                except Exception:
                    pass
        ax_hm.set_title('Heatmap r de Spearman (* = p<0.05)', fontsize=10)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    df_info = pd.read_csv(os.path.join(DATA_DIR, 'subject-info.csv'))

    # --- PSD + espectrograma para Subject00 (ejemplo visual) ---
    eeg_b, ch_names, _ = load_edf(os.path.join(DATA_DIR, 'Subject00_1.edf'),
                                   seg_s=60)
    eeg_t, _,        _ = load_edf(os.path.join(DATA_DIR, 'Subject00_2.edf'),
                                   seg_s=60)
    eeg_b_f = np.array([filtrar(ch) for ch in eeg_b])
    eeg_t_f = np.array([filtrar(ch) for ch in eeg_t])

    plot_psd_y_espectrograma(eeg_b_f, eeg_t_f, ch_names)

    # --- Potencia por bandas en todos los sujetos ---
    df_res = calcular_potencias_todos(df_info)
    print(df_res.head(10))

    plot_boxplots_bandas(df_res)
    df_stats = stats_efecto_tarea_y_grupos(df_res)
    print('\nResultados estadísticos (efecto de tarea):')
    print(df_stats[df_stats['sig_task'] != 'ns'])

    # --- Correlaciones Spearman score vs. potencia ---
    df_corr = correlaciones_score_eeg(df_res, df_info)
    print('\nCorrelaciones Spearman:')
    print(df_corr)
    plot_correlaciones(df_res, df_info, df_corr)


if __name__ == '__main__':
    main()

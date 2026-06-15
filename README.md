# TPS - EEG durante tareas aritméticas (EEGMAT)

Trabajo Práctico de Procesamiento de Señales Biomédicas (16.63) - ITBA 2026 Q1.

Análisis de EEG y ECG de la base **EEGMAT** (Zyma et al., 2019, PhysioNet)
durante reposo y tarea aritmética mental, comparando sujetos con buena (Grupo G)
y mala (Grupo B) performance.

## Estructura

```
.
├── 01_carga_y_filtrado.py     # carga EDF + filtros Butterworth/Notch
├── 02_analisis_espectral.py   # Welch, bandas, stats, correlaciones
├── 03_qrs_hrv.py              # detección QRS + métricas HRV
├── 04_dashboard.py            # interfaz interactiva (Streamlit)
├── requirements.txt
└── muestras_EEG/              # dataset EEGMAT (incluido, ~176 MB)
```

## Setup

```bash
# 1. Clonar el repo
git clone <url> && cd <repo>

# 2. Instalar dependencias
pip install -r requirements.txt
```

El dataset (EEGMAT - Zyma et al. 2019) está incluido en `muestras_EEG/` para que
los scripts y el dashboard corran out-of-the-box. Fuente original (acceso libre):
https://physionet.org/content/eegmat/1.0.0/

## Cómo correr cada parte

```bash
python 01_carga_y_filtrado.py     # exploración del dataset + filtros
python 02_analisis_espectral.py   # PSD, bandas, stats y correlaciones
python 03_qrs_hrv.py              # QRS + HRV
streamlit run 04_dashboard.py     # interfaz gráfica interactiva
```

## Dashboard

Dashboard web minimalista con cuatro pestañas:

- **Señal:** sujeto y canal a elección. Muestra la señal filtrada, su PSD por
  Welch y el espectrograma.
- **Filtro:** sliders para los parámetros del Butterworth (fL, fH, orden) y
  del Notch (Q). Redibuja en vivo la respuesta combinada y un tramo filtrado.
- **Grupos:** boxplot G vs. B por canal y banda (Delta / Theta / Alpha / Beta),
  baseline vs. tarea.
- **HRV:** ECG con picos R detectados, tachograma de intervalos RR y métricas
  FC / SDNN / RMSSD.

Sidebar fija arriba a la izquierda para elegir sujeto y condición; los cambios
afectan a todas las tabs.

## Datos

- 36 sujetos, EEG de 19 canales (sistema 10-20) + ECG, fs = 500 Hz.
- Dos archivos `.edf` por sujeto: `_1` baseline / `_2` tarea aritmética.
- `subject-info.csv` con edad, género, score (# sustracciones) y grupo
  (`Count quality` ∈ {0,1}).

## Autores

Alfie, Gonzales Chaves y Tardá - Grupo 08 - ITBA 2026 Q1.

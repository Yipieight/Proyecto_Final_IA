# Asistente Robótico por Comandos de Voz

**Universidad Rafael Landívar — Inteligencia Artificial, Primer Semestre 2026**  
**Proyecto Final — Modalidad A: Robot Móvil**

Sistema completo de reconocimiento de voz en tiempo real que clasifica comandos en español y controla un robot móvil mediante comunicación UDP sobre WiFi. Todo el pipeline —preprocesamiento, entrenamiento y despliegue— fue implementado desde cero sin APIs externas ni modelos preentrenados.

---

## Resultados

| Modelo | Tarea | Val Accuracy | Parámetros |
|---|---|---|---|
| **VoiceCNN** | Comandos aislados (6 clases) | **100.0%** | 2,207,142 |
| **VoiceGRU** | Comandos compuestos (4 clases) | **98.83%** | 675,460 |

---

## Arquitectura del sistema

```
Micrófono (16 kHz)
    ↓
VAD — Voice Activity Detection (energía RMS adaptativa)
    ↓
Mel-Spectrogram (NumPy manual — sin librosa)
  Pre-énfasis → Hann window → FFT → Banco de filtros mel → Log → Resize/Pad → Z-score
    ↓
┌─────────────────────────────────────────────────┐
│ VoiceCNN (palabras aisladas)                    │
│   Conv(1→32) → Conv(32→64) → Conv(64→128)      │
│   → FC(8192→256) → FC(256→64) → FC(64→6)       │
│   Entrada: (1, 64, 64)   Salida: 6 clases       │
└─────────────────────────────────────────────────┘
       ó
┌─────────────────────────────────────────────────┐
│ VoiceGRU (frases compuestas de 2 palabras)      │
│   GRU(64→256, 2 capas) → FC(256→128) → FC(→4)  │
│   Entrada: (300, 64)     Salida: 4 clases       │
└─────────────────────────────────────────────────┘
    ↓
Predicción (confianza ≥ 80%)
    ↓
UDP (1 byte) → ESP32 (192.168.1.4:9999) → Motores
```

---

## Comandos reconocidos

### Palabras aisladas — VoiceCNN

| Comando | Acción | Byte UDP |
|---|---|---|
| ALTO | Para el robot | `0x00` |
| ADELANTE | Avanza recto | `0x01` |
| IZQUIERDA | Gira izquierda | `0x02` |
| DERECHA | Gira derecha | `0x03` |
| GIRO_IZQ | Pivote 90° izquierda | `0x04` |
| GIRO_DER | Pivote 90° derecha | `0x05` |

### Frases compuestas — VoiceGRU

| Dices | El robot hace | Bytes UDP |
|---|---|---|
| "adelante izquierda" | Avanza → Gira izq | `0x01` → `0x02` |
| "derecha adelante" | Gira der → Avanza | `0x03` → `0x01` |
| "giro izquierda adelante" | Pivota izq → Avanza | `0x04` → `0x01` |
| "giro derecha adelante" | Pivota der → Avanza | `0x05` → `0x01` |

---

## Estructura del repositorio

```
proyecto_de_IA_final/
│
├── voice_dataset.py            # Dataset PyTorch + Mel-Spectrogram NumPy manual
├── model_voice.py              # Arquitectura VoiceCNN
├── model_gru.py                # Arquitectura VoiceGRU + compute_mel_sequence
├── train_voice.py              # Entrenamiento CNN (split 85/15)
├── train_voice_kfold.py        # 5-Fold CV + reporte completo de métricas
├── train_gru.py                # Entrenamiento GRU sobre pares compuestos
├── generate_voice_dataset.py   # Síntesis TTS + 11 fases de aumentación
├── main_voice.py               # Pipeline en tiempo real: palabras aisladas (VAD/PTT)
├── main_compound.py            # Pipeline en tiempo real: frases compuestas (VAD/PTT)
├── main_compound_ptt2.py       # Pipeline determinista de respaldo (doble PTT)
├── diagnose_voice.py           # Matriz de confusión y métricas por clase
├── extract_embeddings.py       # Extracción de embeddings CNN (auxiliar)
├── utils.py                    # IP ESP32, puerto UDP, constantes
│
├── models/
│   ├── voice_model.pth         # VoiceCNN entrenada (val_acc 100%)
│   └── gru_model.pth           # VoiceGRU entrenada (val_acc 98.83%)
│
├── metrics/
│   ├── gru_report.json         # Métricas y curvas de entrenamiento GRU
│   ├── kfold_report.md         # Reporte 5-Fold CV de la CNN
│   ├── kfold_results.json      # Resultados numéricos del K-Fold
│   ├── voice_cm_train.csv      # Matriz de confusión (train)
│   └── voice_cm_val.csv        # Matriz de confusión (validación)
│
├── esp32_firmware/             # Firmware del microcontrolador (C++/Arduino)
├── estudio/                    # Notas de estudio para defensa
├── entrenamiento.ipynb         # Notebook Jupyter reproducible con métricas
│
├── Informe_Proyecto_Final_IA_URLandívar_2026.pdf
├── README.md                   # Este archivo
└── VOICE_CHEATSHEET.md         # Referencia rápida de comandos
```

> **Corpus de audio (902 MB):** No incluido en el repositorio por tamaño. Se regenera ejecutando `generate_voice_dataset.py` (requiere modelos TTS). Ver sección de instalación.

---

## Instalación

### Requisitos

- Python 3.11+
- `uv` como gestor de paquetes (recomendado) o `pip`
- macOS con Apple Silicon (MPS) o Linux con CPU/CUDA

```bash
# Clonar el repositorio
git clone https://github.com/Yipieight/proyecto_de_IA_final.git
cd proyecto_de_IA_final

# Instalar dependencias con uv
uv sync

# Alternativa con pip
pip install torch sounddevice soundfile scipy pynput numpy
```

### Dependencias principales

| Librería | Uso |
|---|---|
| `torch` | Entrenamiento e inferencia (VoiceCNN, VoiceGRU) |
| `sounddevice` | Captura de audio en tiempo real |
| `soundfile` | Lectura de archivos WAV |
| `scipy` | Resampleo de audio y zoom para resize |
| `numpy` | Mel-spectrogram completo desde cero |
| `pynput` | Control de teclado para modo PTT |

---

## Generación del dataset

El corpus se genera con síntesis TTS española (8 voces, 3 acentos) + 11 fases de aumentación de audio.

```bash
# Dataset completo (~32,000 muestras, ~20-30 min)
uv run python generate_voice_dataset.py

# Solo regenerar una clase
uv run python generate_voice_dataset.py --only ADELANTE
```

**Voces TTS utilizadas:**

| Motor | Voces | Acento |
|---|---|---|
| Piper TTS | davefx, sharvard | España |
| Piper TTS | daniela | Argentina |
| Piper TTS | ald, claude_mx | México |
| Kokoro ONNX | ef_dora, em_alex, em_santa | Español neutro |

**Fases de aumentación (11 fases):** limpio, ruido ambiente (SNR 5/10/20 dB), reverb, filtro de micrófono, reverb+ruido, mic+ruido, clipping, EQ aleatorio, doble ruido, eco de pasillo, lugar público.

---

## Entrenamiento

### VoiceCNN — Comandos aislados

```bash
# Entrenamiento estándar (split 85/15, recomendado)
uv run python train_voice.py --epochs 40

# K-Fold cross-validation con reporte completo de métricas
uv run python train_voice_kfold.py --epochs 40
```

### VoiceGRU — Comandos compuestos

```bash
# Entrenamiento GRU (genera pares compuestos desde el corpus CNN)
uv run python train_gru.py --epochs 40 --pairs 1000
```

### Diagnóstico del modelo CNN

```bash
uv run python diagnose_voice.py
```

---

## Ejecución en tiempo real

### Palabras aisladas (VoiceCNN)

```bash
# Listar micrófonos disponibles
uv run python main_voice.py --list-devices

# Modo VAD — detección automática de voz
uv run python main_voice.py --microphone 4

# Modo PTT — mantén ESPACIO para hablar (recomendado en presentaciones)
uv run python main_voice.py --ptt --microphone 4

# Prueba sin enviar al ESP32
uv run python main_voice.py --ptt --microphone 4 --verbose --dry-run

# Con auto-stop: el robot se mueve 1 segundo y para solo
uv run python main_voice.py --microphone 4 --delay 1.0
```

### Frases compuestas (VoiceGRU)

```bash
# Modo VAD (default) — di la frase completa de corrido
uv run python main_compound.py --microphone 4

# Modo PTT — mantén ESPACIO y di la frase completa
uv run python main_compound.py --microphone 4 --ptt

# Prueba sin ESP32
uv run python main_compound.py --microphone 4 --dry-run --verbose

# Ajustar duración de cada movimiento (default 1.0 s)
uv run python main_compound.py --microphone 4 --delay 1.5
```

### Flags principales

| Flag | Default | Descripción |
|---|---|---|
| `--microphone N` | auto | Índice del micrófono |
| `--ptt` | off | Push-to-Talk con ESPACIO |
| `--dry-run` | off | Muestra predicciones sin enviar al ESP32 |
| `--verbose` | off | Barras de probabilidad por clase |
| `--confidence X` | 0.80 | Confianza mínima para aceptar predicción |
| `--delay X` | 0.0 | Segundos activo por comando antes de STOP automático |
| `--threshold X` | 0.015 | Sensibilidad del VAD (energía RMS base) |

---

## Hardware

- **Microcontrolador:** ESP32 (recibe comandos UDP por WiFi)
- **IP del robot:** `192.168.1.4:9999` (configurable en `utils.py`)
- **PC de inferencia:** MacBook Pro M1/M2 (Apple MPS) o cualquier laptop con Python
- **Micrófono:** USB genérico o integrado (16 kHz mínimo)

El firmware del ESP32 está en `esp32_firmware/`. Recibe un byte UDP y lo traduce a señales PWM para los motores DC.

---

## Restricciones del proyecto

| Restricción | Cumplimiento |
|---|---|
| Sin modelos preentrenados de voz | VoiceCNN y VoiceGRU entrenadas desde cero |
| Sin APIs externas (Google STT, Whisper, etc.) | Sistema 100% offline durante evaluación |
| Sin datasets públicos | Corpus generado con TTS propio + aumentación |
| Mel-Spectrogram implementado manualmente | `voice_dataset.py` — NumPy + FFT sin librosa |
| Latencia < 500 ms | ~80 ms medidos (captura + mel + inferencia + UDP) |

---

## Dataset

El corpus de audio (902 MB, 32,112 archivos WAV) no está incluido en este repositorio por su tamaño. Para regenerarlo:

```bash
uv run python generate_voice_dataset.py
```

Esto requiere los modelos TTS en `voices/`. Los modelos Piper se descargan automáticamente desde HuggingFace. Los modelos Kokoro ONNX se descargan desde [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases).

---

*Universidad Rafael Landívar | Inteligencia Artificial | Primer Semestre 2026*

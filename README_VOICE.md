# Control por Voz — Módulo de Respaldo

> **Rama:** `feature/voice-control`  
> **Universidad Rafael Landívar — Inteligencia Artificial, Primer Semestre 2026**  
> **Proyecto Final: Navegación Autónoma y Reconocimiento de Señales**

Este módulo implementa un sistema de control del robot mediante comandos de voz como alternativa de respaldo al sistema principal de visión por cámara. El modelo de reconocimiento de voz fue diseñado, entrenado y evaluado completamente desde cero, sin pesos preentrenados ni datasets públicos.

---

## Comandos reconocidos

| Dices | El robot hace | Byte UDP |
|-------|--------------|----------|
| **"detener"** | Se detiene | `0x00` |
| **"adelante"** | Avanza recto | `0x01` |
| **"izquierda"** | Gira izquierda | `0x02` |
| **"derecha"** | Gira derecha | `0x03` |
| **"giro izquierda"** | Pivote 90° izquierda | `0x04` |
| **"giro derecha"** | Pivote 90° derecha | `0x05` |

---

## Arquitectura del sistema

```
Micrófono → VAD / PTT → Audio (float32, 16kHz)
                ↓
     Mel-Spectrogram (NumPy manual)
       Pre-énfasis → FFT → Filtro Mel triangular
       → Log → Resize 64×64 → Z-score
                ↓
          VoiceCNN (PyTorch)
       3 bloques Conv+BN+ReLU+MaxPool
       → FC(8192→256) → FC(256→64) → FC(64→6)
       2,207,142 parámetros entrenables
                ↓
     Predicción (confianza ≥ 80%)
                ↓
     Comando UDP (1 byte) → ESP32
```

### VoiceCNN — Detalle de capas

| Capa | Entrada | Salida | Parámetros |
|------|---------|--------|-----------|
| Conv2d(1→32) + BN + ReLU + MaxPool | (1, 64, 64) | (32, 32, 32) | 320 |
| Conv2d(32→64) + BN + ReLU + MaxPool | (32, 32, 32) | (64, 16, 16) | 18,560 |
| Conv2d(64→128) + BN + ReLU + MaxPool | (64, 16, 16) | (128, 8, 8) | 73,984 |
| FC(8192→256) + ReLU + Dropout(0.5) | 8192 | 256 | 2,097,408 |
| FC(256→64) + ReLU | 256 | 64 | 16,448 |
| FC(64→6) | 64 | 6 | 390 |

---

## Generación del dataset

El dataset fue generado 100% de forma sintética usando síntesis de voz (TTS) española en combinación con técnicas extensivas de aumentación de audio. No se utilizó ningún dataset público.

### Voces de síntesis (8 voces)

| Motor | Voces | Variedad |
|-------|-------|----------|
| **Piper TTS** | davefx, sharvard (ES-España) | Voces masculinas medium |
| **Piper TTS** | daniela (AR-Argentina) | Voz femenina high quality |
| **Piper TTS** | ald, claude_mx (MX-México) | Voces masculinas medium/high |
| **Kokoro ONNX** | ef_dora | Voz femenina española |
| **Kokoro ONNX** | em_alex, em_santa | Voces masculinas españolas |

### Variantes base por síntesis (13 variantes)

| Tipo | Valores |
|------|---------|
| Velocidad | 0.65×, 0.82×, 1.20×, 1.45× |
| Volumen | 0.50×, 1.50× |
| Pitch | −6, −4, −2, +2, +4, +6 semitonos |
| Original | sin cambios |

### Fases de aumentación (11 fases)

| Fase | Descripción | Escenario simulado |
|------|-------------|-------------------|
| 2a — Limpio | 13 variantes base | Entorno silencioso |
| 2b — Ruido | Babble, multitud, lluvia, viento, tráfico, rosa, blanco a SNR 20/10/5 dB | Distintos ambientes |
| 2c — Reverb | Eco de sala 0.10–0.35 s | Aula o sala cerrada |
| 2d — Mic filter | Bandpass 300–3400 Hz | Micrófono barato / teléfono |
| 2e — Reverb + ruido | Combinado | Sala ruidosa |
| 2f — Mic + ruido | Combinado | Mic barato en ambiente |
| 2g — Clipping | Saturación 55–80% del pico | Micrófono sobrecargado |
| 2h — EQ aleatorio | Boost/cut en 2–3 bandas | Diferentes salas / micrófonos |
| 2i — Doble ruido | 2 fuentes simultáneas | Entornos complejos |
| 2j — Eco de pasillo | 1–3 reflexiones a 50–200 ms | Pasillos / paredes |
| 2k — Lugar público | Voz al 25–55% + multitud fuerte SNR 1–5 dB | Presentación pública |

**Cada fase 2b–2k aplica las 13 variantes base de forma independiente**, generando combinaciones reales como "voz rápida + reverb", "voz grave + ruido doble", "voz aguda + eco de pasillo".

### Estadísticas del dataset

| Métrica | Valor |
|---------|-------|
| Total de muestras | **32,112** |
| Muestras por clase | **5,352** |
| Clases | 6 |
| Sample rate | 16,000 Hz |
| Duración máxima | 2.0 segundos |
| Formato | WAV mono float32 |

---

## Mel-Spectrogram (implementación NumPy manual)

El preprocesamiento se implementó completamente desde cero sin usar librerías de audio de alto nivel:

```
Audio (float32, mono, 16kHz)
  ↓ Pre-énfasis (coef = 0.97)
  ↓ Ventana Hann + RFFT (N_FFT=512, HOP=160)
  ↓ Espectro de potencia
  ↓ Banco de filtros triangulares mel (64 filtros, 0–8000 Hz)
  ↓ Compresión logarítmica (log + ε)
  ↓ Resize a 64×64 (scipy.ndimage.zoom, orden 1)
  ↓ Normalización Z-score por muestra
→ Tensor (1, 64, 64) float32
```

---

## Resultados de entrenamiento

| Métrica | Valor |
|---------|-------|
| Val accuracy (split 85/15) | **100.0%** |
| Épocas | 40 |
| Optimizador | Adam (lr=1e-3) |
| Scheduler | StepLR (step=10, γ=0.5) |
| Dispositivo | MPS (Apple Silicon) |

---

## Protocolo de comunicación

El sistema usa **UDP (1 byte por comando)** entre la laptop y el ESP32:

```
Laptop (PyTorch inference)  →  UDP  →  ESP32 (control de motores)
         192.168.x.x               puerto 9999
```

El byte enviado mapea directamente a un comando de movimiento del robot (`0x00`–`0x05`), minimizando la latencia de comunicación.

---

## Estructura de archivos

```
feature/voice-control/
│
├── generate_voice_dataset.py  # Síntesis TTS + 11 fases de aumentación
├── voice_dataset.py           # Dataset PyTorch + mel-spectrogram NumPy
├── model_voice.py             # Arquitectura VoiceCNN
├── train_voice.py             # Entrenamiento rápido (split 85/15)
├── train_voice_kfold.py       # 5-Fold CV + modelo final + reporte markdown
├── main_voice.py              # Pipeline en tiempo real (VAD y PTT)
├── diagnose_voice.py          # Matriz de confusión y métricas por clase
│
├── utils.py                   # IP ESP32, protocolo UDP, constantes
├── esp32_firmware/            # Firmware del microcontrolador
│
├── models/
│   └── voice_model.pth        # Modelo entrenado
│
└── VOICE_CHEATSHEET.md        # Referencia rápida de comandos
```

---

## Instalación y uso

### Requisitos

```bash
# Instalar dependencias (gestor uv)
uv sync
```

**Dependencias principales:** PyTorch ≥ 2.0, sounddevice, soundfile, scipy, pynput, kokoro-onnx

### Generar dataset

```bash
# Dataset completo (~32,000 muestras, ~20-30 min)
uv run python generate_voice_dataset.py

# Solo una clase (para reemplazar una clase específica)
uv run python generate_voice_dataset.py --only DETENER
```

> **Nota:** Requiere modelos TTS en `voices/`. Se descargan automáticamente al ejecutar (Piper desde HuggingFace). Los modelos Kokoro ONNX deben descargarse manualmente desde [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases).

### Entrenar el modelo

```bash
# Entrenamiento estándar (recomendado, más rápido)
uv run python train_voice.py --epochs 40

# K-Fold cross-validation (genera reporte completo de métricas)
uv run python train_voice_kfold.py --epochs 40
```

### Diagnóstico del modelo

```bash
uv run python diagnose_voice.py
```

### Ejecutar el sistema de voz

```bash
# Ver micrófonos disponibles
uv run python main_voice.py --list-devices

# Modo VAD — detección automática de voz (ambiente tranquilo)
uv run python main_voice.py --microphone 1

# Modo PTT — Push-to-Talk con ESPACIO (ambiente ruidoso / presentación)
uv run python main_voice.py --ptt --microphone 1

# Prueba sin enviar comandos al ESP32
uv run python main_voice.py --ptt --microphone 1 --verbose --dry-run
```

### Flags disponibles

| Flag | Default | Descripción |
|------|---------|-------------|
| `--ptt` | off | Push-to-Talk: mantén ESPACIO para hablar |
| `--dry-run` | off | Muestra predicciones sin enviar al ESP32 |
| `--verbose` | off | Barras de probabilidad por clase en pantalla |
| `--microphone N` | auto | Índice del micrófono |
| `--threshold X` | 0.015 | Sensibilidad VAD base (RMS) |
| `--confidence X` | 0.80 | Confianza mínima para aceptar predicción |
| `--no-vad` | off | Desactiva detección de energía, predice continuamente |
| `--list-devices` | — | Lista micrófonos y sale |

---

## Restricciones del proyecto cumplidas

| Restricción | Cumplimiento |
|-------------|-------------|
| Sin modelos preentrenados | VoiceCNN inicializada aleatoriamente y entrenada desde cero |
| Sin datasets públicos | Dataset 100% generado por síntesis TTS propia |
| PyTorch para el modelo | Sí — VoiceCNN implementada con `nn.Module` |
| NumPy para preprocesamiento | Mel-spectrogram implementado íntegramente con NumPy/FFT |
| OpenCV solo para lectura | No aplica en este módulo (audio, no imagen) |
| 100% offline en evaluación | Sin conexión a internet durante inferencia |

> Piper TTS y Kokoro ONNX se usan únicamente como **herramientas de generación de datos** (equivalente a grabar voces humanas). El modelo de IA en producción es exclusivamente la VoiceCNN.

---

## Modos de detección de voz

### VAD — Voice Activity Detection (default)

Escucha continuamente el micrófono y calcula la energía RMS por chunk de 50 ms. Cuando la energía supera el umbral adaptativo (`noise_floor × 4.0`), captura hasta detectar 450 ms de silencio o alcanzar 3 segundos. Recomendado en ambientes tranquilos.

### PTT — Push-to-Talk (`--ptt`)

El usuario controla exactamente cuándo grabar presionando ESPACIO. Incluye trim automático de silencio previo al habla para que el mel-spectrogram coincida con el patrón aprendido durante entrenamiento. Recomendado en presentaciones o ambientes ruidosos.

---

*Proyecto Final — Universidad Rafael Landívar | Inteligencia Artificial 2026*

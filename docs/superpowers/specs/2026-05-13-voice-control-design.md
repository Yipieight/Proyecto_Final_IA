# Diseño: Rama de Control por Voz — `feature/voice-control`

**Fecha:** 2026-05-13  
**Rama:** `feature/voice-control` (creada desde `main`)  
**Propósito:** Plan B de presentación. Si el sistema de cámara falla, se demuestra control manual del robot por comandos de voz en español.

---

## 1. Contexto

El proyecto principal (rama `main`) usa un iPhone como cámara cenital, un CNN entrenado en PyTorch que clasifica 6 patrones de pista, y envía comandos UDP al ESP32. Esta rama es independiente: reutiliza el mismo ESP32, el mismo firmware y el mismo protocolo UDP, pero reemplaza la cámara+CNN por un micrófono+VoiceCNN.

**Restricciones del proyecto que esta rama debe respetar:**
- Sin modelos preentrenados para el módulo de IA de reconocimiento
- Operación 100% offline durante la evaluación
- Librerías permitidas: PyTorch, NumPy, OpenCV (no usado aquí)
- Dataset 100% propio (generado con TTS local)

---

## 2. Arquitectura

```
Micrófono (sounddevice)
  └─► Captura continua de audio (16kHz, mono)
       └─► VAD simple por energía → ventana de 1-2s cuando hay voz
            └─► Mel-spectrogram 64×64 (NumPy: FFT → filtro mel → log)
                 └─► VoiceCNN.forward() → 6 logits → argmax → clase
                      └─► UDPSender → 1 byte → ESP32 (puerto 9999)
                           └─► setMotores() → L298N → motores
```

---

## 3. Dataset de voz

### Generación con Piper TTS
- **Herramienta:** Piper TTS con modelo de voz español (`es_MX` o `es_ES`)
- **Clases (6):**

| Clase | Palabras clave | Byte UDP |
|-------|---------------|----------|
| STOP | "para", "stop", "detente" | 0x00 |
| ADELANTE | "adelante", "avanza" | 0x01 |
| IZQUIERDA | "izquierda", "curva izquierda" | 0x02 |
| DERECHA | "derecha", "curva derecha" | 0x03 |
| GIRO_IZQ | "giro izquierda", "noventa izquierda" | 0x04 |
| GIRO_DER | "giro derecha", "noventa derecha" | 0x05 |

- **Volumen:** 200–300 WAVs por clase
- **Augmentación:** variación de velocidad (0.85×–1.15×), tono (±2 semitones), ruido blanco suave
- **Múltiples voces:** al menos 2-3 voces distintas de Piper para reducir domain gap
- **Storage:** `data/voice/<CLASE>/sample_NNNN.wav`

### Formato de audio
- Sample rate: 16 000 Hz
- Mono, 16-bit PCM
- Duración: 0.5–2.0s por muestra

---

## 4. Preprocesamiento (NumPy manual)

Mel-spectrogram calculado manualmente para cumplir la regla de "preprocesamiento matricial":

1. **Pre-énfasis:** `y[t] = x[t] - 0.97 * x[t-1]`
2. **Ventaneo:** Hann window de 25ms, hop de 10ms
3. **FFT:** `np.fft.rfft()` por ventana
4. **Filtro mel:** banco de 64 filtros triangulares (Hz → mel escala)
5. **Log:** `np.log(mel + 1e-8)`
6. **Resize:** interpolación a 64×64 para igualar input del CNN

---

## 5. Arquitectura VoiceCNN

Inspirada en `NavCNN` (`model_nav.py`) pero adaptada a 1 canal (escala de grises del spectrogram):

```
Input: (1, 64, 64)  ← mel-spectrogram normalizado

Conv1: 1→32,  kernel 3×3, BN, ReLU, MaxPool 2×2  → (32, 32, 32)
Conv2: 32→64, kernel 3×3, BN, ReLU, MaxPool 2×2  → (64, 16, 16)
Conv3: 64→128,kernel 3×3, BN, ReLU, MaxPool 2×2  → (128, 8, 8)

Flatten → 8192
FC1: 8192 → 256, ReLU, Dropout 0.5
FC2: 256  → 64,  ReLU
FC3: 64   → 6   (logits)
```

- Loss: CrossEntropyLoss
- Optimizer: Adam lr=1e-3
- Epochs: 20–30 (early stopping manual)
- Guardado: `models/voice_model.pth`

---

## 6. Pipeline de inferencia en tiempo real

**Archivo:** `main_voice.py`

```
Loop:
  1. Capturar chunk de 100ms con sounddevice (ring buffer)
  2. Calcular energía RMS del chunk
  3. Si energía > umbral (VAD activo):
       - Acumular audio hasta silencio (energía < umbral por >300ms)
       - Extraer mel-spectrogram de la ventana acumulada
       - Inferir con VoiceCNN → clase
       - Enviar byte UDP al ESP32
  4. Mostrar estado en consola (clase detectada, comando, WiFi)
  5. Ctrl+C para detener y enviar STOP al ESP32
```

**Parámetros ajustables:**
```python
SAMPLE_RATE   = 16000
VAD_THRESHOLD = 0.02    # RMS mínimo para activar VAD
SILENCE_MS    = 300     # ms de silencio para cortar utterance
MAX_AUDIO_S   = 2.0     # máx duración de un comando
```

---

## 7. Archivos de la rama

### Nuevos (solo en `feature/voice-control`)
| Archivo | Función |
|---------|---------|
| `generate_voice_dataset.py` | Instala Piper, genera WAVs por clase con augmentación |
| `voice_dataset.py` | PyTorch Dataset: carga WAVs → mel-spectrogram → tensor |
| `model_voice.py` | VoiceCNN (3 conv + 3 FC) |
| `train_voice.py` | Entrena VoiceCNN, guarda `models/voice_model.pth` |
| `main_voice.py` | Pipeline completo mic → CNN → UDP |

### Reutilizados (solo lectura, sin modificar)
| Archivo | Qué usa |
|---------|---------|
| `utils.py` | `ESP32_IP`, `ESP32_PORT`, `CMD_STOP`, `CMD_*` bytes |

### Sin tocar
Todos los demás: `main_robot.py`, `state_machine.py`, `model_nav.py`, `dataset.py`, `train_kfold.py`, `preprocessing.py`, etc.

---

## 8. Dependencias nuevas (solo en esta rama)

```
piper-tts       ← generación del dataset (offline)
sounddevice     ← captura de micrófono en tiempo real
soundfile       ← lectura/escritura de WAVs
scipy           ← filtros de audio para augmentación
```

Se agregan al `pyproject.toml` solo en esta rama.

---

## 9. Flujo de trabajo

```
1. git checkout -b feature/voice-control
2. uv run python generate_voice_dataset.py   ← genera data/voice/
3. uv run python train_voice.py              ← entrena y guarda models/voice_model.pth
4. uv run python main_voice.py               ← demo en tiempo real
```

---

## 10. Riesgos y mitigación

| Riesgo | Mitigación |
|--------|-----------|
| Domain gap TTS→voz real | Múltiples voces Piper + augmentación de ruido |
| Latencia alta en inferencia | VoiceCNN pequeño; Metal/CPU de Mac es suficiente |
| VAD dispara con ruido del salón | Ajustar `VAD_THRESHOLD` en el aula real |
| Piper no tiene buena voz española | Probar `es_MX-IngridOlga-medium` y `es_ES-sharvard-medium` |

---

## 11. Criterio de éxito

El sistema funciona si, en condiciones del aula, una persona dice un comando y el robot ejecuta la acción correcta en menos de 2 segundos, sin conexión a internet, con el modelo cargado localmente.

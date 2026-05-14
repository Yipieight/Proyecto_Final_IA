# Control por Voz — Referencia Rápida

## Comandos que reconoce el modelo

| Di esto       | El robot hace    | Byte UDP |
|---------------|-----------------|----------|
| **"stop"**         | Se detiene       | `0x00`   |
| **"adelante"**     | Avanza           | `0x01`   |
| **"izquierda"**    | Gira izquierda   | `0x02`   |
| **"derecha"**      | Gira derecha     | `0x03`   |
| **"giro izquierda"** | Giro completo IZQ | `0x04` |
| **"giro derecha"** | Giro completo DER | `0x05`  |

---

## Antes de empezar

```bash
# Ver micrófonos disponibles (anota el número del que vas a usar)
uv run python main_voice.py --list-devices
```

---

## Modos de uso

### VAD — Detección automática (recomendado en ambiente tranquilo)
El micrófono escucha siempre. Cuando detecta voz, captura y predice solo.

```bash
# Prueba sin enviar al ESP32
uv run python main_voice.py --dry-run

# Con micrófono específico
uv run python main_voice.py --microphone 1 --dry-run

# En vivo (envía al ESP32)
uv run python main_voice.py --microphone 1

# Ver probabilidades de cada clase en pantalla
uv run python main_voice.py --microphone 1 --verbose --dry-run
```

### PTT — Push-to-Talk (recomendado en ambiente ruidoso)
Mantén **ESPACIO** presionado → habla → suelta → predice.

```bash
# Prueba sin enviar al ESP32
uv run python main_voice.py --ptt --dry-run

# Con micrófono específico
uv run python main_voice.py --ptt --microphone 1 --dry-run

# En vivo (envía al ESP32)
uv run python main_voice.py --ptt --microphone 1

# Ver probabilidades de cada clase
uv run python main_voice.py --ptt --microphone 1 --verbose --dry-run
```

---

## Todos los flags

| Flag | Tipo | Default | Para qué sirve |
|------|------|---------|----------------|
| `--ptt` | switch | off | Push-to-talk (ESPACIO para hablar) |
| `--dry-run` | switch | off | Muestra predicciones sin enviar al ESP32 |
| `--verbose` | switch | off | Barras de probabilidad por clase en pantalla |
| `--microphone N` | entero | auto | Índice del micrófono a usar |
| `--threshold X` | decimal | `0.015` | Sensibilidad VAD (bájalo si no detecta, súbelo si hay falsos positivos) |
| `--confidence X` | decimal | `0.80` | Confianza mínima para aceptar predicción (0.0–1.0) |
| `--list-devices` | switch | — | Lista micrófonos y sale |

---

## Ajuste fino VAD

Si el VAD **no detecta** tu voz:
```bash
uv run python main_voice.py --threshold 0.008 --microphone 1 --dry-run
```

Si el VAD **activa con ruido ambiente** (falsos positivos):
```bash
uv run python main_voice.py --threshold 0.030 --microphone 1 --dry-run
```

Si el modelo descarta predicciones por baja confianza:
```bash
uv run python main_voice.py --confidence 0.65 --microphone 1 --dry-run
```

---

## Dataset y entrenamiento

```bash
# Regenerar dataset (borra y recrea ~25 920 WAVs)
uv run python generate_voice_dataset.py

# Entrenar modelo desde cero
uv run python train_voice.py --epochs 40

# Ver métricas de confusión del modelo actual
uv run python diagnose_voice.py
```

---

## Combinaciones útiles para la presentación

```bash
# Demostración segura (no mueve el robot)
uv run python main_voice.py --ptt --microphone 1 --verbose --dry-run

# Demostración en vivo con ESP32
uv run python main_voice.py --ptt --microphone 1

# Si hay mucho ruido en el salón
uv run python main_voice.py --ptt --microphone 1 --confidence 0.85
```

---

## Estado del modelo actual

- **Muestras:** 25,920 (4,320 por clase)
- **Val accuracy:** 100%
- **Voces entrenadas:** 5 Piper + 3 Kokoro ONNX (español)
- **Aumentaciones:** 10 fases — ruido, reverb, mic, clipping, EQ, doble ruido, eco

# Guía de Supervivencia para la Defensa

> [!warning] LEE ESTO ANTES QUE CUALQUIER OTRO ARCHIVO
> Este archivo te dice exactamente qué estudiar, en qué orden, y cómo explicarlo. Cada sección tiene una caja **"Dónde está en el código"** con el archivo y línea exacta.

---

## Los 5 temas que sí o sí te van a preguntar

1. ¿Qué es y cómo funciona el Mel-Spectrogram?
2. ¿Cómo funciona el entrenamiento de la CNN?
3. ¿Qué hace cada capa de la CNN?
4. ¿Por qué GRU y no LSTM?
5. ¿Cómo funciona el pipeline completo (voz → robot)?

---

## El Sistema Completo

```
TU VOZ
  ↓
Micrófono → VAD detecta si hay voz → Mel-Spectrogram → CNN o GRU → UDP → ESP32 → MOTORES
```

> [!example] Dónde está en el código
> - **Micrófono + VAD** → `main_voice.py` líneas 192–263, función `run_vad()`
> - **Mel-Spectrogram CNN** → `voice_dataset.py` línea 63, función `compute_mel_spectrogram()`
> - **Mel-Spectrogram GRU** → `model_gru.py` línea 66, función `compute_mel_sequence()`
> - **CNN** → `model_voice.py` línea 23, clase `VoiceCNN`
> - **GRU** → `model_gru.py` línea 121, clase `VoiceGRU`
> - **UDP** → `main_voice.py` línea 65, `socket.SOCK_DGRAM`

---

# TEMA 1 — El Mel-Spectrogram

## ¿Qué es?

El mel-spectrogram convierte el audio en una **imagen 2D**: el eje horizontal es el tiempo, el eje vertical son las frecuencias en escala perceptual humana, y el brillo es la intensidad. La CNN lo analiza como si fuera una foto.

**¿Por qué no el audio crudo?** El audio crudo son 32.000 números (2 s × 16.000 muestras/s). El mel-spectrogram lo comprime a 64×64 = 4.096 números que contienen lo más importante.

> [!example] Dónde está en el código
> **`voice_dataset.py` líneas 17–22** — parámetros base
> ```python
> TARGET_SR  = 16000   # 16.000 muestras por segundo
> N_FFT      = 512     # ventana de 32 ms
> HOP_LENGTH = 160     # salto de 10 ms entre ventanas
> N_MELS     = 64      # 64 filtros mel
> SPEC_SIZE  = 64      # resultado final 64×64 píxeles
> ```

## Los 8 pasos

### Paso 1 — Pre-énfasis

Amplifica frecuencias altas (consonantes "s", "t") que son débiles.

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 90**
> ```python
> audio = np.concatenate([[audio[0]], audio[1:] - 0.97 * audio[:-1]])
> ```

### Paso 2 — Dividir en ventanas

Divide el audio en pedazos de 512 muestras (32 ms) con salto de 160 muestras (10 ms).

> [!example] Dónde está en el código
> **`voice_dataset.py` líneas 96–99**
> ```python
> n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
> indices  = np.arange(N_FFT)[None,:] + HOP_LENGTH * np.arange(n_frames)[:,None]
> frames   = audio[indices]
> ```

### Paso 3 — Ventana Hann

Multiplica cada ventana por una curva de campana para evitar "fantasmas" de frecuencias falsas.

> [!example] Dónde está en el código
> **`voice_dataset.py` líneas 102–103**
> ```python
> window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))
> frames = frames * window
> ```

### Paso 4 — FFT

Convierte cada ventana de tiempo a espectro de frecuencias (512 muestras → 257 frecuencias).

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 106**
> ```python
> power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
> ```

### Paso 5 — Banco de filtros Mel

Agrupa las 257 frecuencias en 64 grupos en escala mel (como escucha el oído humano).

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 109** (banco construido en líneas 34–57)
> ```python
> mel = _FILTERBANK @ power.T    # resultado: (64, n_frames)
> ```
> La conversión Hz ↔ Mel: `hz_to_mel(hz) = 2595 × log10(1 + hz/700)`

### Paso 6 — Logaritmo

El oído percibe el volumen en escala logarítmica. El `1e-8` evita `log(0)`.

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 112**
> ```python
> mel = np.log(mel + 1e-8)
> ```

### Paso 7 — Resize a 64×64

Estira o comprime para que todos los audios tengan el mismo tamaño (requerido por la CNN).

> [!example] Dónde está en el código
> **`voice_dataset.py` líneas 115–117**
> ```python
> mel = zoom(mel, (SPEC_SIZE / h, SPEC_SIZE / w), order=1)
> ```

### Paso 8 — Normalización Z-score

Elimina diferencias de volumen entre hablantes (fuerte o suave → mismo resultado).

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 120**
> ```python
> mel = (mel - mel.mean()) / (mel.std() + 1e-8)
> ```

> [!tip] Resultado final
> `(1, 64, 64)` — una imagen en escala de grises de tu voz, lista para entrar a la CNN.
> Código: **`voice_dataset.py` línea 165** → `tensor = torch.from_numpy(mel[np.newaxis])`

---

# TEMA 2 — El Entrenamiento de la CNN

## ¿Qué significa entrenar?

Entrenar la CNN es como enseñarle a un niño a reconocer animales mostrándole miles de fotos con etiquetas. El niño se equivoca al principio, corrige sus errores, y aprende. La CNN hace exactamente lo mismo.

> [!example] Dónde está en el código
> **`train_voice.py` líneas 55–101** — bucle de entrenamiento completo

## El ciclo que se repite por cada batch de 32 imágenes

> [!example] Dónde está en el código
> **`train_voice.py` líneas 61–65** — el corazón del entrenamiento
> ```python
> optimizer.zero_grad()          # línea 61 — borra gradientes del batch anterior
> logits = model(x)              # línea 62 — la CNN predice
> loss   = criterion(logits, y)  # línea 63 — Cross-Entropy mide el error
> loss.backward()                # línea 64 — Backprop calcula la culpa de cada peso
> optimizer.step()               # línea 65 — Adam actualiza los pesos
> ```

**Una vuelta por todo el dataset = 1 época. Se hicieron 30 épocas.**

## Cross-Entropy Loss — ¿qué mide?

Mide qué tan diferente es lo que predijo la CNN vs. la realidad.

**Fórmula:** `Loss = -log(probabilidad predicha para la clase correcta)`

| Predicción para ADELANTE | Loss | Interpretación |
|---|---|---|
| 0.99 | 0.01 | casi sin error ✓ |
| 0.60 | 0.51 | error moderado |
| 0.01 | 4.60 | error muy grande ✗ |

> [!example] Dónde está en el código
> **`train_voice.py` línea 47**
> ```python
> criterion = nn.CrossEntropyLoss()
> ```

## Adam Optimizer

Algoritmo que actualiza los pesos. Adapta el tamaño del paso automáticamente para cada parámetro.

> [!example] Dónde está en el código
> **`train_voice.py` líneas 45–46**
> ```python
> optimizer = torch.optim.Adam(model.parameters(), lr=lr)
> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
> ```
> - `lr = 0.001` — tamaño del paso inicial
> - `StepLR` — cada 10 épocas reduce el lr a la mitad (pasos más pequeños y precisos al final)

## Backpropagation

Calcula cuánto contribuyó cada peso al error, usando la regla de la cadena. PyTorch lo hace automáticamente con `loss.backward()`.

> [!example] Dónde está en el código
> **`train_voice.py` línea 64** → `loss.backward()`

## Guardar el mejor modelo

Solo se guarda cuando la precisión de validación mejora.

> [!example] Dónde está en el código
> **`train_voice.py` líneas 91–97**
> ```python
> if val_acc > best_val_acc:
>     torch.save({"model_state": model.state_dict(),
>                 "best_val_acc": best_val_acc}, MODEL_VOICE_PATH)
> ```

---

# TEMA 3 — Arquitectura de la VoiceCNN

> [!example] Dónde está en el código
> **`model_voice.py` líneas 23–63** — clase `VoiceCNN` completa

## Los 3 bloques convolucionales

Cada bloque tiene el mismo patrón: **Conv → BatchNorm → ReLU → MaxPool**

> [!example] Dónde está en el código
> **`model_voice.py` líneas 28–45**
> ```python
> self.block1 = nn.Sequential(
>     nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 filtros
>     nn.BatchNorm2d(32),
>     nn.ReLU(inplace=True),
>     nn.MaxPool2d(2, 2),    # (32, 32, 32)
> )
> self.block2 = nn.Sequential(...)  # (64, 16, 16)
> self.block3 = nn.Sequential(...)  # (128, 8, 8)
> ```

| Bloque | Código | Entrada | Salida |
|---|---|---|---|
| Bloque 1 | `model_voice.py:28` | (1, 64, 64) | (32, 32, 32) |
| Bloque 2 | `model_voice.py:34` | (32, 32, 32) | (64, 16, 16) |
| Bloque 3 | `model_voice.py:40` | (64, 16, 16) | (128, 8, 8) |
| Flatten | `model_voice.py:47` | (128, 8, 8) | 8.192 números |

## El clasificador final (capas FC)

> [!example] Dónde está en el código
> **`model_voice.py` líneas 49–57**
> ```python
> self.classifier = nn.Sequential(
>     nn.Flatten(),
>     nn.Linear(8192, 256),   # línea 51
>     nn.ReLU(inplace=True),
>     nn.Dropout(0.5),         # línea 53
>     nn.Linear(256, 64),
>     nn.ReLU(inplace=True),
>     nn.Linear(64, 6),        # 6 clases de comandos
> )
> ```

## Forward pass — cómo fluye la imagen

> [!example] Dónde está en el código
> **`model_voice.py` líneas 59–63**
> ```python
> def forward(self, x):
>     x = self.block1(x)        # (1,64,64) → (32,32,32)
>     x = self.block2(x)        # → (64,16,16)
>     x = self.block3(x)        # → (128,8,8)
>     return self.classifier(x)  # → 6 logits
> ```

## Qué hace cada componente

| Componente | Línea | Qué hace |
|---|---|---|
| `Conv2d` | `model_voice.py:29` | Detecta patrones locales (bordes, texturas) |
| `BatchNorm2d` | `model_voice.py:30` | Normaliza activaciones → entrenamiento estable |
| `ReLU` | `model_voice.py:31` | Activa solo valores positivos → no-linealidad |
| `MaxPool2d` | `model_voice.py:32` | Reduce tamaño a la mitad → eficiencia |
| `Dropout(0.5)` | `model_voice.py:53` | Apaga 50% neuronas → evita overfitting |

---

# TEMA 4 — La VoiceGRU

> [!example] Dónde está en el código
> **`model_gru.py` líneas 121–154** — clase `VoiceGRU`

## ¿Por qué necesitamos una GRU?

La CNN clasifica **una sola imagen** (una palabra). Para una frase de 2 palabras se necesita un modelo que entienda **secuencias en el tiempo**.

```
"adelante izquierda" hablado de corrido:
[adelante...............][izquierda..]
[f1][f2][f3]...[f150][f151]...[f300]

La GRU lee los 300 frames uno por uno y recuerda el contexto anterior.
```

> [!example] Dónde está en el código
> **`model_gru.py` línea 148** → `_, h_n = self.gru(x)`
> **`model_gru.py` línea 32** → `T_MAX = 300` (3 segundos de audio)

## GRU vs LSTM

| Característica | GRU | LSTM |
|---|---|---|
| Gates | 2 (reset, update) | 3 (input, forget, output) |
| Parámetros | Menos | Más |
| Velocidad | Más rápido | Más lento |
| Para secuencias cortas | Igual de bueno | No justifica el costo |

> [!tip] Respuesta para la defensa
> *"Elegimos GRU porque nuestras secuencias son cortas — máximo 3 segundos, 300 frames. La GRU tiene menos parámetros y entrena más rápido con rendimiento equivalente para esta tarea."*

## Arquitectura

> [!example] Dónde está en el código
> **`model_gru.py` líneas 132–145**
> ```python
> self.gru = nn.GRU(
>     input_size=64,    # 64 features mel por frame
>     hidden_size=256,  # estado oculto de 256 dimensiones
>     num_layers=2,     # 2 capas apiladas
>     batch_first=True,
>     dropout=0.3,
> )
> self.head = nn.Sequential(
>     nn.Linear(256, 128),
>     nn.ReLU(inplace=True),
>     nn.Dropout(0.3),
>     nn.Linear(128, 4),   # 4 clases compuestas
> )
> ```

## compute_mel_sequence vs compute_mel_spectrogram

| | CNN | GRU |
|---|---|---|
| Función | `compute_mel_spectrogram()` | `compute_mel_sequence()` |
| Archivo | `voice_dataset.py:63` | `model_gru.py:66` |
| Salida | `(64, 64)` imagen cuadrada | `(300, 64)` secuencia temporal |
| Resize | Sí, con `zoom()` | No — pad/truncate a T_MAX |
| Analogía | Foto | Película |

> [!example] Dónde está en el código — pad/truncate
> **`model_gru.py` líneas 107–111**
> ```python
> if mel.shape[0] < t_max:
>     mel = np.pad(mel, ((0, t_max - mel.shape[0]), (0, 0)))
> else:
>     mel = mel[:t_max]
> ```

## Cómo se construye el dataset del GRU

Se concatenan WAVs de palabras individuales con silencio aleatorio entre ellas.

> [!example] Dónde está en el código
> **`train_gru.py` líneas 112–116**
> ```python
> gap_samples = int(sr1 * rng.integers(0, 301) / 1000)  # 0-300 ms de silencio
> silence     = np.zeros(gap_samples, dtype=np.float32)
> compound    = np.concatenate([a1, silence, a2])         # palabra1 + silencio + palabra2
> mel         = compute_mel_sequence(compound, sr1)
> ```

## Las 4 clases compuestas

> [!example] Dónde está en el código
> **`model_gru.py` líneas 36–41** — `COMPOUND_CLASSES`
> **`model_gru.py` líneas 56–61** — `COMPOUND_CMD_BYTES` (bytes UDP al ESP32)

| Dices | El robot hace | Bytes UDP |
|---|---|---|
| "adelante izquierda" | Avanza 1 s → Gira izq 1 s → Para | `0x01` → `0x02` |
| "derecha adelante" | Gira der 1 s → Avanza 1 s → Para | `0x03` → `0x01` |
| "giro izquierda adelante" | Pivota izq 1 s → Avanza 1 s → Para | `0x04` → `0x01` |
| "giro derecha adelante" | Pivota der 1 s → Avanza 1 s → Para | `0x05` → `0x01` |

La secuencia de envío al ESP32:

> [!example] Dónde está en el código
> **`main_compound.py` líneas 149–156**
> ```python
> ok1 = sender.send(byte1)    # CMD1
> time.sleep(args.delay)       # espera N segundos
> ok2 = sender.send(byte2)    # CMD2
> time.sleep(args.delay)
> sender.send(CMD_STOP)        # STOP
> ```

> [!question] ¿Por qué solo 4 clases y no 7?
> *"Probamos 7 pero el GRU confundía los pares espejo (ADELANTE_IZQUIERDA vs IZQUIERDA_ADELANTE) porque acústicamente son las mismas palabras en diferente orden. Con 4 clases sin ambigüedad, la precisión subió de 96% a 98.83%."*

---

# TEMA 5 — El VAD

## ¿Qué es el VAD?

Voice Activity Detection — detecta automáticamente cuándo estás hablando sin necesitar que presiones un botón.

> [!example] Dónde está en el código
> **`main_voice.py` líneas 192–263** — función `run_vad()`
> **Parámetros: `main_voice.py` líneas 41–47**
> ```python
> CHUNK_DURATION_S   = 0.05   # 50 ms por chunk
> SILENCE_DURATION_S = 0.45   # 450 ms de silencio para cerrar
> NOISE_ALPHA        = 0.98   # suavizado del piso de ruido
> VAD_MARGIN         = 4.0    # threshold = noise_floor × 4
> ```

## Algoritmo paso a paso

> [!example] Dónde está en el código
> **`main_voice.py` líneas 237–260**
> ```python
> rms       = float(np.sqrt(np.mean(chunk ** 2)))          # línea 237
> threshold = max(vad_threshold, noise_floor * VAD_MARGIN) # línea 238
>
> if not recording:
>     # Actualiza piso de ruido con suavizado exponencial
>     noise_floor = NOISE_ALPHA * noise_floor + (1 - NOISE_ALPHA) * rms  # línea 241
>     if rms >= threshold:
>         recording = True   # INICIO DE GRABACIÓN ▶
> else:
>     buffer.append(chunk)
>     if rms < threshold * 0.6:
>         silent_count += 1   # cuenta silencios
>     if silent_count >= silence_chunks:
>         # FIN → predecir
>         cls_name, conf, probs = infer(model, np.concatenate(buffer), ...)
> ```

**¿Por qué `NOISE_ALPHA = 0.98`?** El piso de ruido se actualiza lentamente (98% del valor anterior + 2% nuevo). Evita que un ruido repentino suba el umbral y deje de detectar tu voz.

Para comandos compuestos el silencio de cierre es mayor:

> [!example] Dónde está en el código
> **`main_compound.py` línea 57**
> ```python
> SILENCE_DURATION_S = 0.70   # 700 ms — no cortará entre las 2 palabras
> ```

---

# TEMA 6 — El Dataset

> [!example] Dónde está en el código
> **`generate_voice_dataset.py`** — genera todos los WAVs con TTS y aumentación
> **`voice_dataset.py` líneas 127–166** — clase `VoiceDataset` que los carga

## Las 6 clases de voz aislada

> [!example] Dónde está en el código
> **`voice_dataset.py` línea 26**
> ```python
> VOICE_CLASSES = ["DETENER", "ADELANTE", "IZQUIERDA", "DERECHA", "GIRO_IZQ", "GIRO_DER"]
> ```

## Por qué TTS y no grabaciones reales

El proyecto pedía grabar voces humanas. Usamos síntesis TTS con 8 voces de 3 acentos distintos + 11 fases de aumentación para cubrir más variabilidad que grabaciones manuales.

| Elemento | Cantidad |
|---|---|
| Voces TTS | 8 (Piper ES/AR/MX + Kokoro) |
| Variantes por voz | 13 (velocidad, volumen, pitch) |
| Fases de aumentación | 11 (entornos distintos) |
| Total de muestras | 32.112 WAVs |
| Por clase | 5.352 muestras |

> [!tip] Defensa del corpus TTS
> *"Usamos TTS cubriendo 3 acentos (España, Argentina, México), variedad de género, y 11 tipos de aumentación ambiental — más variabilidad que grabaciones manuales en un solo entorno."*

---

# Top 15 Preguntas — Respuesta + Dónde Señalar

> [!faq] ¿Qué es el mel-spectrogram?
> *"Una imagen 2D del audio: eje X = tiempo, eje Y = frecuencias en escala perceptual humana, brillo = intensidad. Implementado desde cero en NumPy."*
> **Señalar → `voice_dataset.py:63`** función `compute_mel_spectrogram()`

> [!faq] ¿Qué hace el pre-énfasis?
> *"Amplifica frecuencias altas. Fórmula: `y[n] = x[n] - 0.97×x[n-1]`. Las consonantes 's' y 't' son débiles; el pre-énfasis las hace visibles para el modelo."*
> **Señalar → `voice_dataset.py:90`**

> [!faq] ¿Por qué no usaron MFCC?
> *"El mel-spectrogram preserva más información espectral. Los MFCC aplican una DCT adicional que comprime más. Con CNN 2D, el mel-spectrogram funciona mejor porque la CNN aprende qué partes son relevantes."*

> [!faq] ¿Qué es backpropagation?
> *"Calcula el gradiente del error respecto a cada peso usando la regla de la cadena. Le dice a cada peso cuánto contribuyó al error para corregirlo. PyTorch lo hace con `loss.backward()`."*
> **Señalar → `train_voice.py:64`**

> [!faq] ¿Por qué Adam y no SGD?
> *"Adam adapta la tasa de aprendizaje individualmente para cada parámetro según el historial de gradientes. Converge más rápido y es más robusto para redes profundas."*
> **Señalar → `train_voice.py:45`** `torch.optim.Adam`

> [!faq] ¿Qué hace el Dropout?
> *"Apaga aleatoriamente el 50% de las neuronas en cada entrenamiento. Obliga a la red a aprender representaciones redundantes y evita el overfitting."*
> **Señalar → `model_voice.py:53`** `nn.Dropout(0.5)`

> [!faq] ¿Qué es BatchNorm?
> *"Normaliza las activaciones de cada capa a media 0 y varianza 1. Estabiliza y acelera el entrenamiento, también funciona como regularizador."*
> **Señalar → `model_voice.py:30`** `nn.BatchNorm2d(32)`

> [!faq] ¿Por qué UDP y no TCP?
> *"UDP no tiene handshake ni confirmación, lo que reduce la latencia a menos de 1 ms. Para control de robots en tiempo real es preferible un comando perdido ocasionalmente que un retraso de 20–50 ms."*
> **Señalar → `main_voice.py:65`** `socket.SOCK_DGRAM`

> [!faq] ¿Cuál es la latencia total del sistema?
> *"Menos de 100 ms: ~50 ms captura VAD, ~15 ms mel-spectrogram, ~20 ms inferencia en MPS, menos de 1 ms UDP. Muy por debajo del límite de 500 ms."*

> [!faq] ¿Por qué solo 4 clases compuestas?
> *"Probamos 7 pero los pares espejo se confundían. Con 4 clases sin ambigüedad, la precisión subió de 96% a 98.83%."*
> **Señalar → `model_gru.py:36`** `COMPOUND_CLASSES`

> [!faq] ¿Cómo funciona el VAD?
> *"Calcula la energía RMS de chunks de 50 ms y los compara con un umbral = 4× el piso de ruido ambiente. El piso se actualiza con suavizado exponencial α=0.98."*
> **Señalar → `main_voice.py:237`**

> [!faq] ¿Por qué usaron TTS?
> *"Generamos 32.112 muestras con 8 voces de 3 acentos y 11 tipos de aumentación ambiental — más variabilidad que grabaciones manuales en un solo entorno."*

> [!faq] ¿Qué es la Cross-Entropy Loss?
> *"Mide la diferencia entre la distribución predicha y la real. Matemáticamente es `-log(prob. predicha para la clase correcta)`. A mayor certeza y acierto, menor el loss."*
> **Señalar → `train_voice.py:47`** `nn.CrossEntropyLoss()`

> [!faq] ¿Qué es el StepLR?
> *"Reduce el learning rate multiplicándolo por γ=0.5 cada 10 épocas. Pasos grandes al inicio para explorar rápido; pasos pequeños al final para afinar los pesos."*
> **Señalar → `train_voice.py:46`** `StepLR(optimizer, step_size=10, gamma=0.5)`

> [!faq] ¿Cuántos parámetros tiene la CNN?
> *"2.207.142 parámetros entrenables. La mayoría están en la primera capa FC (8192→256), donde se combina toda la información espacial de las convoluciones."*
> **Señalar → `model_voice.py:51`** `nn.Linear(8192, 256)`

---

# Mapa Rápido de Archivos

Si el ingeniero señala un archivo y pregunta qué hace:

| Archivo | Qué hace | Línea clave |
|---|---|---|
| `voice_dataset.py` | Mel-spectrogram desde cero en NumPy | `:63` `compute_mel_spectrogram()` |
| `model_voice.py` | Arquitectura de la CNN | `:23` clase `VoiceCNN` |
| `train_voice.py` | Bucle de entrenamiento CNN | `:55` bucle por época |
| `model_gru.py` | GRU + preprocesamiento temporal | `:66` secuencia / `:121` `VoiceGRU` |
| `train_gru.py` | Dataset compuesto + entrenamiento GRU | `:55` `CompoundAudioDataset` |
| `main_voice.py` | Control por voz en tiempo real VAD/PTT | `:192` `run_vad()` |
| `main_compound.py` | Control frases compuestas VAD/PTT | `:163` `run_vad()` |
| `generate_voice_dataset.py` | Genera corpus TTS con aumentación | dataset completo |
| `utils.py` | IP del ESP32 y constantes UDP | `ESP32_IP`, `ESP32_PORT` |

---

# Plan de Estudio (Hoy)

- **Hora 1** — Temas 1 y 2 (Mel-Spectrogram + Entrenamiento). Abre `voice_dataset.py:63` y `train_voice.py:55` mientras lees.
- **Hora 2** — Temas 3 y 4 (CNN + GRU). Abre `model_voice.py:23` y `model_gru.py:121` como referencia.
- **Hora 3** — Temas 5 y 6 (VAD + Dataset). Abre `main_voice.py:237` para ver el VAD en vivo.
- **Hora 4** — Repasa las 15 preguntas en voz alta. Para cada una di también el archivo donde vive ese código.

> [!tip] El consejo más importante
> No memorices definiciones. Entiende el **flujo completo** de tu voz hasta el movimiento del robot. Si entiendes eso, puedes responder cualquier pregunta porque todo se conecta.
> 
> **Cuando no sepas responder algo, abre el código y señala la línea** — eso demuestra dominio real.

---

*La Presentación y Defensa vale 40 de 100 puntos. Tu objetivo es demostrar que ENTIENDES lo que construiste, no que memorizaste fórmulas.*

# REGLAS CRÍTICAS DEL PROYECTO
## Robot Autónomo — Navegación por CNN Espacio-Temporal
### Universidad Rafael Landívar | IA 2026-1

> Documento de referencia rápida para el equipo.
> **El Reto Bonus de señales fue cancelado.** Este proyecto cubre únicamente la Navegación Base.

---

## REGLAS IRROMPIBLES (violación = 0 en esa sección)

| # | Regla | Consecuencia si se viola |
|---|---|---|
| R1 | **Cero modelos preentrenados** — NavCNN inicializada aleatoriamente, entrenada 100% con datos propios | Anulación del criterio "CNN Temporal" (25 pts) |
| R2 | **Cero datasets públicos** — solo imágenes capturadas en la pista física del equipo | Anulación del criterio "Dataset" (15 pts) |
| R3 | **Cero torchvision.models, ResNet, YOLO, MobileNet** con pesos de ImageNet | Igual que R1 |
| R4 | Al menos **un filtro crítico implementado manualmente en NumPy** (Gaussian o Sobel) | Pérdida parcial del criterio "Preprocesamiento" |
| R5 | OpenCV **solo** para `cv2.VideoCapture`, `cv2.resize`, `cv2.imshow` | Pérdida del criterio de preprocesamiento matricial |
| R6 | **Modo avión** durante la demo — sin APIs externas, inferencia local | Anulación de la demo (35 pts) |
| R7 | **Mínimo 3,000 imágenes** en el dataset de navegación | Descuento proporcional en "Dataset" |
| R8 | Dataset **100% balanceado** — ninguna clase por debajo del 15% | Penalización en "Dataset" |
| R9 | El robot debe **detenerse 2 segundos en CRUCE_T** y elegir dirección aleatoriamente (`random.choice`) | Falla en la demo |
| R10 | **Todos los integrantes** deben poder defender cualquier componente del sistema | Pérdida del 35% de "Presentación y Defensa" |

---

## LIBRERÍAS PERMITIDAS

```
PyTorch / TensorFlow-Keras    → arquitectura CNN, entrenamiento
NumPy                         → preprocesamiento matricial manual
OpenCV                        → lectura de video, resize básico
Matplotlib / Seaborn          → visualización de métricas
scikit-learn                  → métricas (solo evaluate.py)
socket + threading            → comunicación UDP con ESP32
requests                      → NO usar para ESP32 (usar UDP)
```

**Prohibido explícitamente:**
- `torchvision.models.*`
- `tf.keras.applications.*`
- Hugging Face `transformers` con pesos preentrenados
- Datasets: GTSRB, ImageNet, CIFAR, COCO, etc.
- AutoML, NAS, HPO automatizado

---

## ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────┐
│                    MacBook (CPU)                         │
│  Hotspot WiFi ─────────────────────────────────────┐    │
│  Thread 1: cv2.VideoCapture (stream MJPEG)         │    │
│  Thread 2: preprocess → NavCNN → StateMachine      │    │
│            → UDP datagram (1 byte) ────────────────┼──► │ ESP32
│  HUD: clase nav, FPS, comando, WiFi OK             │    │
└─────────────────────────────────────────────────────────┘
         ▲ stream MJPEG                    ▼ PWM
    Celular (IP Webcam)           L298N → Motores DC
```

**Protocolo UDP:**
| Byte | Comando | Cuándo |
|---|---|---|
| `0x00` | STOP | CRUCE_T (primeros 2 s), inicio |
| `0x01` | ADELANTE | RECTA |
| `0x02` | IZQUIERDA | CURVA_IZQ, GIRO_90_IZQ |
| `0x03` | DERECHA | CURVA_DER, GIRO_90_DER |
| `0x04` | T_CROSS_GIRO | Después de los 2 s en CRUCE_T |

---

## CLASES DE NAVEGACIÓN (6 en total)

| Clase | Tile físico | Visual en el frame |
|---|---|---|
| `RECTA` | Tile recto 50×50 cm | Línea entra borde inferior, sube recta |
| `CURVA_IZQ` | Tile CURVA_RADIO_MEDIO (girado) | Línea curva hacia la izquierda |
| `CURVA_DER` | Tile CURVA_RADIO_MEDIO | Línea curva hacia la derecha |
| `GIRO_90_IZQ` | Tile GIRO_90_IZquierda | Quiebre duro a 90° izquierda |
| `GIRO_90_DER` | Tile GIRO_90_Derecha | Quiebre duro a 90° derecha |
| `CRUCE_T` | Tile CRUCE_T | Línea transversal visible (T) |

**Pista física — 7 tiles (50×50 cm c/u, ~1.5×1.0 m ensamblada):**
- 3× RECTA
- 1× CURVA_RADIO_MEDIO (usar dos veces: una para IZQ, una para DER rotando 180°)
- 1× GIRO_90_IZQ
- 1× GIRO_90_DER
- 1× CRUCE_T

**Material:** papel bond mate 180g impreso en plotter. Sin cubierta acrílica.

---

## CRITERIOS DE EVALUACIÓN (15 puntos netos)

| Criterio | Puntos | Umbral mínimo |
|---|---|---|
| Documentación técnica | 10% (1.5 pts) | Diagrama de arquitectura, matriz de confusión, análisis cuantitativo |
| Dataset + pista | 15% (2.25 pts) | ≥ 3,000 imgs, balance ≥ 15% por clase |
| CNN Temporal + preprocesamiento | 25% (3.75 pts) | Val accuracy ≥ 85%, recall ≥ 75% por clase |
| Control + hardware + latencia | 15% (2.25 pts) | FPS ≥ 8, movimiento fluido, latencia UDP < 50 ms |
| Presentación y defensa | 35% (5.25 pts) | Demo en pista + preguntas técnicas a los 5 integrantes |

---

## CRITERIOS DEMO-READY (verificar con evaluate.py antes de la demo)

```
[OK] Val accuracy global       ≥ 85%
[OK] Recall por clase (todas)  ≥ 75%   ← ninguna clase por debajo
[OK] FPS del pipeline          ≥ 8 FPS
[OK] Robot completa pista sin caerse en 2 min continuos
[OK] CRUCE_T: parada exacta de 2 s + dirección aleatoria
[OK] Batería cargada al 100%
[OK] Laptop en modo avión
[OK] Hotspot del Mac activo, ESP32 conectado (ver Serial Monitor)
```

---

## CAPÍTULO TEÓRICO OBLIGATORIO (entregable escrito, no código)

> Sección del documento PDF: **"Escalamiento a Tráfico Dinámico"**

Investigar y plasmar (sin implementar):
1. Por qué la clasificación (CNN actual) es insuficiente para evitar colisiones con otros robots.
2. Qué arquitectura de **Object Detection** (YOLO, SSD, FCOS) se propondría.
3. Impacto en la recolección del dataset (bounding boxes, anotaciones).
4. Cómo abordar el cuello de botella computacional en la MacBook/CPU para mantener ≥ 8 FPS con detección de objetos.

---

*Última actualización: 2026-05-06 — Reto Bonus cancelado por el profesor.*

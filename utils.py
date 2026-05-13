"""
Constantes globales, rutas y utilidades compartidas del proyecto.
Reto Bonus cancelado — solo navegación.
"""

import os
import numpy as np
from collections import deque

# ── Dimensiones de imagen ────────────────────────────────────────────────────
IMG_WIDTH  = 64
IMG_HEIGHT = 64

# ── Stack temporal: N frames consecutivos apilados como canales ──────────────
FRAME_STACK = 3

# ── Clases de navegación (6 clases, ninguna de señales) ──────────────────────
NAV_CLASSES     = ['RECTA', 'CURVA_IZQ', 'CURVA_DER', 'GIRO_90_IZQ', 'GIRO_90_DER', 'CRUCE_T']
NAV_CLASS_IDX   = {c: i for i, c in enumerate(NAV_CLASSES)}
NAV_IDX_CLASS   = {i: c for i, c in enumerate(NAV_CLASSES)}
NUM_NAV_CLASSES = len(NAV_CLASSES)

# ── ROI: fracción del alto del frame que se ignora (parte superior) ───────────
ROI_TOP_FRAC = 0.35

# ── ESP32 — comunicación UDP ──────────────────────────────────────────────────
# Actualizar ESP32_IP con la IP que aparece en el Serial Monitor al arrancar.
# El Mac actúa como hotspot; el ESP32 se conecta a él.
ESP32_IP    = "192.168.1.7"    # IP típica cuando el Mac es el hotspot
ESP32_PORT  = 9999             # puerto UDP — coincidir con el firmware del ESP32

# ── Protocolo UDP (1 byte por comando) ────────────────────────────────────────
CMD_STOP       = 0x00   # parar motores
CMD_FORWARD    = 0x01   # avanzar recto                    (RECTA)
CMD_LEFT       = 0x02   # curva suave izquierda (diferencial: izq lento, der rápido)
CMD_RIGHT      = 0x03   # curva suave derecha   (diferencial: izq rápido, der lento)
CMD_GIRO_LEFT  = 0x04   # pivote 90° izquierda  (solo lado derecho activo)
CMD_GIRO_RIGHT = 0x05   # pivote 90° derecha    (solo lado izquierdo activo)

# Mapa byte → nombre legible (para el HUD)
CMD_NAME = {
    CMD_STOP:       "STOP",
    CMD_FORWARD:    "ADELANTE",
    CMD_LEFT:       "CURVA_IZQ",
    CMD_RIGHT:      "CURVA_DER",
    CMD_GIRO_LEFT:  "GIRO_IZQ",
    CMD_GIRO_RIGHT: "GIRO_DER",
}

# ── Hiperparámetros de entrenamiento ──────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
NUM_EPOCHS_NAV = 60
TRAIN_SPLIT    = 0.80
VAL_SPLIT      = 0.10
# TEST_SPLIT   = 0.10  (lo que queda)

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_NAV_DIR   = os.path.join("data", "navegacion")
MODEL_NAV_PATH = os.path.join("models", "nav_model.pth")
METRICS_DIR    = "metrics"

for _d in [DATA_NAV_DIR, "models", METRICS_DIR]:
    os.makedirs(_d, exist_ok=True)
for _cls in NAV_CLASSES:
    os.makedirs(os.path.join(DATA_NAV_DIR, _cls), exist_ok=True)


# ── Buffer circular de frames ─────────────────────────────────────────────────

class FrameBuffer:
    """Buffer circular de los últimos FRAME_STACK frames preprocesados."""

    def __init__(self, size: int = FRAME_STACK):
        self.size = size
        blank = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        self._buf = __import__('collections').deque(
            [blank.copy() for _ in range(size)], maxlen=size)

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(frame.astype(np.float32))

    def get_stack(self) -> np.ndarray:
        """Devuelve array de forma (FRAME_STACK, H, W)."""
        return np.array(self._buf, dtype=np.float32)

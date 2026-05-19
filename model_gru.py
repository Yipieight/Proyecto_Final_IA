# model_gru.py
"""
GRU para reconocimiento de comandos compuestos de 2 palabras habladas de corrido.

Entrada:  (batch, T_MAX, N_MELS)  ← secuencia temporal de mel-spectrogram completo
Salida:   (batch, N_COMPOUND)     ← logits de comandos compuestos

Arquitectura:
  GRU(input=64, hidden=256, layers=2, batch_first=True)
  → último hidden state de la capa final (batch, 256)
  FC(256 → 128) + ReLU + Dropout(0.3)
  FC(128 → N_COMPOUND)

Preprocesamiento: compute_mel_sequence
  Audio (float32, mono, 16kHz, hasta 3s)
  ↓ Pre-énfasis (α=0.97)
  ↓ Hann window + RFFT (N_FFT=512, HOP=160)
  ↓ Filtro mel triangular (64 filtros, mismo que VoiceCNN)
  ↓ Log compression
  ↓ Pad / truncate → T_MAX=300 frames
  ↓ Z-score
  → (300, 64) float32
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample as scipy_resample

from voice_dataset import TARGET_SR, N_FFT, HOP_LENGTH, N_MELS, _FILTERBANK

T_MAX = 300   # 3 s a 10 ms por frame (HOP=160 a 16kHz)

# ── Definición de clases compuestas ───────────────────────────────────────────

COMPOUND_CLASSES = [
    "ADELANTE_IZQUIERDA",
    "ADELANTE_DERECHA",
    "ADELANTE_DETENER",
    "GIRO_IZQ_ADELANTE",
    "GIRO_DER_ADELANTE",
    "IZQUIERDA_ADELANTE",
    "DERECHA_ADELANTE",
]

COMPOUND_CLASS_IDX = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
COMPOUND_IDX_CLASS = {i: c for i, c in enumerate(COMPOUND_CLASSES)}
NUM_COMPOUND       = len(COMPOUND_CLASSES)

# Par de clases de voz individuales que forman cada comando compuesto
COMPOUND_WORD_PAIRS = {
    "ADELANTE_IZQUIERDA": ("ADELANTE",  "IZQUIERDA"),
    "ADELANTE_DERECHA":   ("ADELANTE",  "DERECHA"),
    "ADELANTE_DETENER":   ("ADELANTE",  "DETENER"),
    "GIRO_IZQ_ADELANTE":  ("GIRO_IZQ",  "ADELANTE"),
    "GIRO_DER_ADELANTE":  ("GIRO_DER",  "ADELANTE"),
    "IZQUIERDA_ADELANTE": ("IZQUIERDA", "ADELANTE"),
    "DERECHA_ADELANTE":   ("DERECHA",   "ADELANTE"),
}

# Par de bytes UDP a enviar al ESP32 para cada comando compuesto
COMPOUND_CMD_BYTES = {
    "ADELANTE_IZQUIERDA": (0x01, 0x02),
    "ADELANTE_DERECHA":   (0x01, 0x03),
    "ADELANTE_DETENER":   (0x01, 0x00),
    "GIRO_IZQ_ADELANTE":  (0x04, 0x01),
    "GIRO_DER_ADELANTE":  (0x05, 0x01),
    "IZQUIERDA_ADELANTE": (0x02, 0x01),
    "DERECHA_ADELANTE":   (0x03, 0x01),
}


# ── Preprocesamiento temporal (sin resize) ────────────────────────────────────

def compute_mel_sequence(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    t_max: int = T_MAX,
) -> np.ndarray:
    """
    Convierte audio completo de un comando compuesto a secuencia temporal de mel.

    A diferencia de compute_mel_spectrogram (que hace resize a 64×64),
    aquí se preserva la dimensión temporal: se obtienen T_MAX frames
    mediante pad/truncate, manteniendo la estructura secuencial para el GRU.

    Retorna: (t_max, N_MELS) float32
    """
    if sr != TARGET_SR:
        audio = scipy_resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)

    # Pre-énfasis
    audio = np.concatenate([[audio[0]], audio[1:] - 0.97 * audio[:-1]])

    # Longitud mínima para al menos un frame
    min_len = N_FFT + HOP_LENGTH
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    # Padding simétrico
    audio = np.pad(audio, N_FFT // 2, mode="reflect")

    # Construcción matricial de frames
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    indices  = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    frames   = audio[indices]

    # Ventana Hann + espectro de potencia
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
    power  = np.abs(np.fft.rfft(frames * window, n=N_FFT)) ** 2  # (n_frames, N_FFT//2+1)

    # Banco de filtros mel: (N_MELS, N_FFT//2+1) @ (N_FFT//2+1, n_frames)
    mel = _FILTERBANK @ power.T          # (N_MELS, n_frames)
    mel = np.log(mel + 1e-8).T           # (n_frames, N_MELS)

    # Pad o truncar a t_max
    if mel.shape[0] < t_max:
        mel = np.pad(mel, ((0, t_max - mel.shape[0]), (0, 0)))
    else:
        mel = mel[:t_max]

    # Normalización Z-score por muestra
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    return mel.astype(np.float32)   # (t_max, N_MELS)


# ── Modelo ────────────────────────────────────────────────────────────────────

class VoiceGRU(nn.Module):

    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden: int = 256,
        num_classes: int = NUM_COMPOUND,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T_MAX, N_MELS)
        _, h_n = self.gru(x)      # h_n: (num_layers, batch, hidden)
        return self.head(h_n[-1]) # tomar última capa: (batch, hidden)


def build_gru_model(device: str = "cpu") -> VoiceGRU:
    return VoiceGRU().to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_gru_model()
    dummy = torch.zeros(4, T_MAX, N_MELS)
    out   = model(dummy)
    print(f"VoiceGRU — Parámetros: {count_parameters(model):,}")
    print(f"Input: {tuple(dummy.shape)}  →  Output: {tuple(out.shape)}")
    print(f"Clases ({NUM_COMPOUND}): {COMPOUND_CLASSES}")

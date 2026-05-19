# model_gru.py
"""
GRU para reconocimiento de comandos compuestos de exactamente 2 palabras.

Entrada:  (batch, 2, 64)  ← secuencia de 2 embeddings de 64 dims extraídos de VoiceCNN
Salida:   (batch, N_COMPOUND) ← logits de comandos compuestos

Arquitectura:
  GRU(input=64, hidden=128, layers=1, batch_first=True)
  → último hidden state (batch, 128)
  FC(128 → 64) + ReLU + Dropout(0.3)
  FC(64 → N_COMPOUND)
"""

import torch
import torch.nn as nn

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

# Mapeo: comando compuesto → secuencia de 2 clases de voz individuales
# (usadas para construir pares de entrenamiento desde embeddings existentes)
COMPOUND_WORD_PAIRS = {
    "ADELANTE_IZQUIERDA": ("ADELANTE",  "IZQUIERDA"),
    "ADELANTE_DERECHA":   ("ADELANTE",  "DERECHA"),
    "ADELANTE_DETENER":   ("ADELANTE",  "DETENER"),
    "GIRO_IZQ_ADELANTE":  ("GIRO_IZQ",  "ADELANTE"),
    "GIRO_DER_ADELANTE":  ("GIRO_DER",  "ADELANTE"),
    "IZQUIERDA_ADELANTE": ("IZQUIERDA", "ADELANTE"),
    "DERECHA_ADELANTE":   ("DERECHA",   "ADELANTE"),
}

EMBED_DIM = 64   # dimensión del embedding extraído de VoiceCNN


class VoiceGRU(nn.Module):

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = 128,
        num_classes: int = NUM_COMPOUND,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, 64)
        _, h_n = self.gru(x)           # h_n: (1, batch, hidden)
        h = h_n.squeeze(0)             # (batch, hidden)
        return self.head(h)            # (batch, N_COMPOUND)


def build_gru_model(device: str = "cpu") -> VoiceGRU:
    return VoiceGRU().to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_gru_model()
    dummy = torch.zeros(4, 2, EMBED_DIM)
    out   = model(dummy)
    print(f"VoiceGRU — Parámetros: {count_parameters(model):,}")
    print(f"Input: {tuple(dummy.shape)}  →  Output: {tuple(out.shape)}")
    print(f"Clases ({NUM_COMPOUND}): {COMPOUND_CLASSES}")

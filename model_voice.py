# model_voice.py
"""
CNN para clasificación de comandos de voz sobre mel-spectrograms.

Entrada: (batch, 1, 64, 64)  ← mel-spectrogram canal único
Salida:  (batch, 6)          ← logits de 6 comandos

Arquitectura (3 bloques + 3 FC):
  Conv(1→32)   → BN → ReLU → MaxPool2×2   → (32, 32, 32)
  Conv(32→64)  → BN → ReLU → MaxPool2×2   → (64, 16, 16)
  Conv(64→128) → BN → ReLU → MaxPool2×2   → (128, 8, 8)
  Flatten → 8192
  FC(8192→256) → ReLU → Dropout(0.5)
  FC(256→64)   → ReLU
  FC(64→6)     ← logits
"""

import torch
import torch.nn as nn
from voice_dataset import SPEC_SIZE, NUM_VOICE_CLASSES


class VoiceCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_VOICE_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        flat = 128 * (SPEC_SIZE // 8) * (SPEC_SIZE // 8)   # 128 * 8 * 8 = 8192

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def build_voice_model(device: str = "cpu") -> VoiceCNN:
    return VoiceCNN().to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_voice_model()
    dummy = torch.zeros(1, 1, 64, 64)
    out   = model(dummy)
    print(f"VoiceCNN — Parámetros: {count_parameters(model):,}")
    print(f"Input: {tuple(dummy.shape)}  →  Output: {tuple(out.shape)}")

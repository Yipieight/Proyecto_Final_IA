"""
CNN espacio-temporal para clasificación de comandos de navegación.

Estrategia temporal: Frame Stacking
  - Los últimos FRAME_STACK frames en escala de grises se apilan como canales.
  - Entrada: (batch, FRAME_STACK, H, W)  →  equivalente a una imagen de 3 canales.
  - Sin Conv3D ni LSTM: menor latencia en hardware de borde.

Por qué frame stacking sobre Conv3D/LSTM:
  • Una sola pasada forward de CNN 2D — latencia mínima en Edge AI.
  • Sin estado oculto que gestionar entre frames.
  • El contexto temporal queda codificado implícitamente en los canales de entrada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import FRAME_STACK, IMG_HEIGHT, IMG_WIDTH, NUM_NAV_CLASSES


class NavCNN(nn.Module):
    """
    CNN con 4 bloques convolucionales + 3 capas fully-connected.

    Arquitectura:
      Conv(FRAME_STACK→16) → BN → ReLU → MaxPool2×2
      Conv(16→32)          → BN → ReLU → MaxPool2×2
      Conv(32→64)          → BN → ReLU → MaxPool2×2
      Conv(64→128)         → BN → ReLU → MaxPool2×2
      Flatten → FC(256) → Dropout → FC(128) → Dropout → FC(NUM_NAV_CLASSES)

    Con IMG=64×64 y 4 MaxPool: mapa espacial final = 4×4 → 128*4*4 = 2048 neuronas.
    """

    def __init__(self,
                 in_channels: int = FRAME_STACK,
                 num_classes: int = NUM_NAV_CLASSES,
                 dropout: float = 0.4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Después de 4 MaxPool(2×2): H/16 × W/16
        flat = 128 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, FRAME_STACK, H, W)
        x = self.block1(x)   # → (B, 16,  H/2,  W/2)
        x = self.block2(x)   # → (B, 32,  H/4,  W/4)
        x = self.block3(x)   # → (B, 64,  H/8,  W/8)
        x = self.block4(x)   # → (B, 128, H/16, W/16)
        return self.classifier(x)   # logits crudos


def build_nav_model(device: str = 'cpu') -> NavCNN:
    """Construye e inicializa el modelo de navegación (sin pesos preentrenados)."""
    return NavCNN().to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_nav_model()
    dummy = torch.zeros(1, FRAME_STACK, IMG_HEIGHT, IMG_WIDTH)
    out   = model(dummy)
    print(f"NavCNN — Parámetros entrenables: {count_parameters(model):,}")
    print(f"Entrada: {tuple(dummy.shape)}  →  Salida: {tuple(out.shape)}")

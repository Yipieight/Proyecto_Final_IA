"""
Dataset PyTorch y augmentación de datos — solo navegación.

Uso:
    python dataset.py       # reporte de balance del dataset de navegación
"""

import os
import argparse
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF

from utils import (
    IMG_WIDTH, IMG_HEIGHT, FRAME_STACK,
    NAV_CLASSES, NAV_CLASS_IDX,
    DATA_NAV_DIR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
)
from preprocessing import preprocess_frame


# ── Augmentación ──────────────────────────────────────────────────────────────

def augment(img: np.ndarray) -> np.ndarray:
    """
    Augmentación aleatoria sobre un frame preprocesado (H, W) float32 en [0,1].
    Devuelve array de la misma forma y tipo.

    Transformaciones aplicadas:
      - Brillo aleatorio ×[0.6, 1.4]
      - Contraste aleatorio ×[0.7, 1.3]
      - Rotación ±12°
      - Ruido gaussiano σ=0.03
      - Zoom aleatorio (recorte + resize al 85–100% del frame)

    NO se aplica flip horizontal — las clases izquierda/derecha son asimétricas.
    """
    t = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)

    if random.random() < 0.5:
        t = TF.adjust_brightness(t, random.uniform(0.6, 1.4))
    if random.random() < 0.35:
        t = TF.adjust_contrast(t, random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        t = TF.rotate(t, random.uniform(-12, 12))
    if random.random() < 0.4:
        t = torch.clamp(t + torch.randn_like(t) * 0.03, 0.0, 1.0)
    if random.random() < 0.3:
        h, w   = t.shape[-2], t.shape[-1]
        frac   = random.uniform(0.85, 1.0)
        ch, cw = int(h * frac), int(w * frac)
        i      = random.randint(0, h - ch)
        j      = random.randint(0, w - cw)
        t = TF.resize(TF.crop(t, i, j, ch, cw), [IMG_HEIGHT, IMG_WIDTH])

    return t.squeeze(0).numpy()


# ── Dataset de navegación ─────────────────────────────────────────────────────

class NavigationDataset(Dataset):
    """
    Carga frames de las 6 clases de navegación y construye stacks temporales.

    Cada muestra tiene forma (FRAME_STACK, H, W): los FRAME_STACK frames
    consecutivos de la misma carpeta de clase, apilados como canales.
    Provee contexto espacio-temporal sin necesidad de Conv3D ni LSTM.

    Para las clases CURVA_IZQ y CURVA_DER se usa el mismo tile físico
    (CURVA_RADIO_MEDIO), abordado desde lados opuestos o rotado 180°.
    """

    def __init__(self, root_dir: str = DATA_NAV_DIR,
                 augment_data: bool = False,
                 stack_size: int = FRAME_STACK):
        self.augment_data = augment_data
        self.stack_size   = stack_size
        self.samples: list[tuple[str, int]] = []

        for cls in NAV_CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = sorted(f for f in os.listdir(cls_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            for name in imgs:
                self.samples.append((os.path.join(cls_dir, name), NAV_CLASS_IDX[cls]))

        if not self.samples:
            print(f"[WARN] NavigationDataset: no se encontraron imágenes en {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path)
        if bgr is None:
            return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        return preprocess_frame(bgr)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        folder = os.path.dirname(path)
        all_imgs = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        try:
            i = all_imgs.index(path)
        except ValueError:
            i = 0

        # Stack: [i-(stack-1), ..., i-1, i] — frames consecutivos de la misma clase
        stack = []
        for offset in range(-(self.stack_size - 1), 1):
            j     = max(0, min(len(all_imgs) - 1, i + offset))
            frame = self._load(all_imgs[j])
            if self.augment_data:
                frame = augment(frame)
            stack.append(frame)

        x = torch.tensor(np.array(stack, dtype=np.float32))   # (FRAME_STACK, H, W)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ── Split train/val/test ──────────────────────────────────────────────────────

def get_loaders(augment_train: bool = True, root_dir: str = DATA_NAV_DIR):
    """
    Divide el dataset de navegación en train/val/test y devuelve tres DataLoaders.

    Usa dos instancias separadas (una con augmentación, una sin) para garantizar
    que val y test NUNCA reciben datos aumentados.
    """
    plain = NavigationDataset(root_dir, augment_data=False)
    n     = len(plain)
    n_tr  = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_te  = n - n_tr - n_val

    perm    = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    tr_idx  = perm[:n_tr]
    val_idx = perm[n_tr:n_tr + n_val]
    te_idx  = perm[n_tr + n_val:]

    if augment_train:
        aug  = NavigationDataset(root_dir, augment_data=True)
        tr_ds = Subset(aug, tr_idx)
    else:
        tr_ds = Subset(plain, tr_idx)

    val_ds = Subset(plain, val_idx)
    te_ds  = Subset(plain, te_idx)

    kwargs = dict(num_workers=0, pin_memory=False)  # macOS Python 3.14: num_workers>0 cuelga
    return (
        DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  **kwargs),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **kwargs),
        DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, **kwargs),
    )


# ── Reporte de balance ────────────────────────────────────────────────────────

def print_balance_report(root_dir: str = DATA_NAV_DIR) -> None:
    print(f"\n{'='*55}")
    print(f"  Navegación — Reporte de Balance del Dataset")
    print(f"{'='*55}")

    counts: dict[str, int] = {}
    for cls in NAV_CLASSES:
        d = os.path.join(root_dir, cls)
        counts[cls] = sum(
            1 for f in os.listdir(d)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ) if os.path.isdir(d) else 0

    total = sum(counts.values())
    if total == 0:
        print("  No se encontraron imágenes. Ejecuta capture_dataset.py primero.")
        return

    for cls, n in counts.items():
        pct  = 100.0 * n / total
        bar  = '█' * int(pct / 2)
        warn = " ← BAJO (<15%)" if pct < 15 else ""
        print(f"  {cls:<22} {n:>5} imgs  {pct:5.1f}%  {bar}{warn}")

    print(f"  {'─'*53}")
    print(f"  {'TOTAL':<22} {total:>5} imgs  (mínimo requerido: 3,000)")
    print(f"  Con augmentación ×4:    ~{total * 4} imgs\n")

    issues = False
    for cls, n in counts.items():
        pct = 100.0 * n / total
        if n < 200:
            print(f"  [!!] {cls}: solo {n} imgs reales — augmentación INSUFICIENTE.")
            issues = True
        elif pct < 15:
            print(f"  [ !] {cls}: {pct:.1f}% — por debajo del mínimo (15%).")
            issues = True
    if not issues:
        print("  [OK] Balance aceptable en todas las clases.")

    if total < 3000:
        print(f"\n  [!!] Total {total} imgs < 3,000 mínimo requerido por el enunciado.")


if __name__ == "__main__":
    print_balance_report()

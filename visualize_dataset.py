"""
Visualiza muestras de cada clase para verificar que las imágenes
están correctamente etiquetadas y son distinguibles entre sí.

Genera dos PNG:
  metrics/dataset_samples_raw.png      — imágenes originales por clase
  metrics/dataset_samples_processed.png — preprocesadas (lo que ve la red)

Uso:
    uv run python visualize_dataset.py
"""

import os
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import NAV_CLASSES, DATA_NAV_DIR, METRICS_DIR, IMG_WIDTH, IMG_HEIGHT
from preprocessing import preprocess_frame

random.seed(42)

N_PER_CLASS = 6
fig_raw, axes_raw = plt.subplots(len(NAV_CLASSES), N_PER_CLASS, figsize=(N_PER_CLASS*2, len(NAV_CLASSES)*2))
fig_pro, axes_pro = plt.subplots(len(NAV_CLASSES), N_PER_CLASS, figsize=(N_PER_CLASS*2, len(NAV_CLASSES)*2))

for ci, cls in enumerate(NAV_CLASSES):
    d = os.path.join(DATA_NAV_DIR, cls)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))

    # Tomar 1 muestra cada N para cubrir todo el rango (no las primeras 6 que son frames consecutivos)
    step = max(1, len(files) // N_PER_CLASS)
    picks = [files[i * step] for i in range(N_PER_CLASS)]

    for si, fname in enumerate(picks):
        path = os.path.join(d, fname)
        bgr  = cv2.imread(path)
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        proc = preprocess_frame(bgr)

        axes_raw[ci, si].imshow(rgb)
        axes_raw[ci, si].axis('off')
        if si == 0:
            axes_raw[ci, si].set_ylabel(cls, fontsize=10, rotation=0,
                                         labelpad=50, ha='right')

        axes_pro[ci, si].imshow(proc, cmap='gray', vmin=0, vmax=1)
        axes_pro[ci, si].axis('off')
        if si == 0:
            axes_pro[ci, si].set_ylabel(cls, fontsize=10, rotation=0,
                                         labelpad=50, ha='right')

fig_raw.suptitle('Muestras crudas por clase (lo que viene del extractor)', fontsize=12)
fig_pro.suptitle(f'Después de preprocess_frame ({IMG_WIDTH}x{IMG_HEIGHT}, lo que ve la red)', fontsize=12)

os.makedirs(METRICS_DIR, exist_ok=True)
out_raw = os.path.join(METRICS_DIR, 'dataset_samples_raw.png')
out_pro = os.path.join(METRICS_DIR, 'dataset_samples_processed.png')
fig_raw.tight_layout()
fig_pro.tight_layout()
fig_raw.savefig(out_raw, dpi=120, bbox_inches='tight')
fig_pro.savefig(out_pro, dpi=120, bbox_inches='tight')
plt.close('all')

print(f"\n✓ Generados:")
print(f"  {out_raw}")
print(f"  {out_pro}")
print(f"\nAbre los PNG y verifica que:")
print(f"  1. Cada clase muestra el patrón correcto (no están mezcladas)")
print(f"  2. Las clases son visualmente distinguibles entre sí en la versión procesada")

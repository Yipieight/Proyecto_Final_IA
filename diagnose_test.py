"""
Diagnóstico rápido del modelo guardado sobre el test set.
No reentrena — usa el modelo en models/nav_model.pth.

Uso:
    uv run python diagnose_test.py
"""

import numpy as np
import torch

from utils import NAV_CLASSES, NAV_CLASS_IDX, NAV_IDX_CLASS, MODEL_NAV_PATH
from model_nav import build_nav_model
from train_kfold import (
    load_samples, split_held_out_test, build_image_cache,
    FoldDataset, evaluate_test,
)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[diag] Dispositivo: {device}")

    # Cargar modelo
    ckpt = torch.load(MODEL_NAV_PATH, map_location=device)
    model = build_nav_model(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[diag] Modelo cargado. val_acc guardado: {ckpt.get('best_val_acc', '?'):.4f}")

    # Reproducir el mismo split
    samples = load_samples()
    test, cv = split_held_out_test(samples, target=int(len(samples) * 0.20))
    print(f"[diag] Test set: {len(test)} imgs")

    cache, folder_lists, pos_map = build_image_cache(samples)

    # Evaluación detallada
    print("\n" + "="*60)
    print("  EVALUACIÓN EN TEST FIJO")
    print("="*60)
    evaluate_test(test, ckpt['model_state'],
                  cache, folder_lists, pos_map, device)

    # Diagnóstico extra: predicciones en train data
    print("\n" + "="*60)
    print("  DIAGNÓSTICO EN CV DATA (no usado en test)")
    print("="*60)
    # Tomamos 100 muestras al azar de CV
    import random
    random.seed(0)
    cv_sample = random.sample(cv, min(200, len(cv)))
    evaluate_test(cv_sample, ckpt['model_state'],
                  cache, folder_lists, pos_map, device)


if __name__ == '__main__':
    main()

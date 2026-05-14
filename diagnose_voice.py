# diagnose_voice.py
"""
Diagnóstico del modelo de voz: matriz de confusión y métricas por clase.
No reentrena — usa models/voice_model.pth tal como está.

Uso:
    uv run python diagnose_voice.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from voice_dataset import VoiceDataset, VOICE_CLASSES, NUM_VOICE_CLASSES
from model_voice import build_voice_model

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")
METRICS_DIR      = "metrics"


def evaluate(loader, model, device) -> tuple[np.ndarray, float]:
    """Devuelve (matriz de confusión N×N, accuracy global)."""
    cm = np.zeros((NUM_VOICE_CLASSES, NUM_VOICE_CLASSES), dtype=int)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            for true, pred in zip(y.numpy(), preds):
                cm[true, pred] += 1
    acc = cm.diagonal().sum() / cm.sum()
    return cm, acc


def print_report(cm: np.ndarray, split_name: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {split_name}")
    print(f"{'='*62}")

    total = cm.sum()
    acc   = cm.diagonal().sum() / total
    print(f"\nAccuracy global: {acc:.1%}  ({cm.diagonal().sum()}/{total})\n")

    # Métricas por clase
    header = f"{'Clase':<14} {'Correctas':>10} {'Total':>7} {'Recall':>8} {'Precision':>10}"
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(VOICE_CLASSES):
        tp        = cm[i, i]
        total_cls = cm[i].sum()
        total_pred = cm[:, i].sum()
        recall    = tp / total_cls  if total_cls  > 0 else 0.0
        precision = tp / total_pred if total_pred > 0 else 0.0
        bar = "█" * int(recall * 20)
        print(f"  {cls:<12} {tp:>6}/{total_cls:<6} {recall:>7.1%}  {precision:>8.1%}  {bar}")

    # Matriz de confusión
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    pad = max(len(c) for c in VOICE_CLASSES) + 1
    header_row = " " * (pad + 2) + "  ".join(f"{c[:4]:>4}" for c in VOICE_CLASSES)
    print(header_row)
    print(" " * (pad + 2) + "-" * (6 * NUM_VOICE_CLASSES))
    for i, cls in enumerate(VOICE_CLASSES):
        row = "  ".join(
            f"\033[32m{cm[i,j]:>4}\033[0m" if i == j else f"{cm[i,j]:>4}"
            for j in range(NUM_VOICE_CLASSES)
        )
        print(f"  {cls:<{pad}} {row}")

    # Confusiones más frecuentes
    print("\nConfusiones más frecuentes (excluyendo diagonal):")
    off_diag = []
    for i in range(NUM_VOICE_CLASSES):
        for j in range(NUM_VOICE_CLASSES):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], VOICE_CLASSES[i], VOICE_CLASSES[j]))
    off_diag.sort(reverse=True)
    if off_diag:
        for count, real, pred in off_diag[:5]:
            print(f"  {real:<12} → {pred:<12}  {count}x")
    else:
        print("  Ninguna — clasificación perfecta en este split.")


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[diagnose_voice] Dispositivo: {device}")

    ckpt  = torch.load(MODEL_VOICE_PATH, map_location=device)
    model = build_voice_model(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[diagnose_voice] Modelo cargado  (val_acc guardado: {ckpt.get('best_val_acc', 0):.1%})")

    # Mismo split que train_voice.py (seed 42, 15% val)
    full_ds = VoiceDataset(augment=False)
    n_val   = max(1, int(len(full_ds) * 0.15))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[diagnose_voice] Split — train: {n_train}  val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    cm_train, _ = evaluate(train_loader, model, device)
    cm_val,   _ = evaluate(val_loader,   model, device)

    print_report(cm_train, "TRAIN SET")
    print_report(cm_val,   "VALIDATION SET  ← más importante")

    # Guardar CSVs
    header = ",".join(VOICE_CLASSES)
    np.savetxt(os.path.join(METRICS_DIR, "voice_cm_train.csv"),
               cm_train, fmt="%d", delimiter=",", header=header, comments="")
    np.savetxt(os.path.join(METRICS_DIR, "voice_cm_val.csv"),
               cm_val, fmt="%d", delimiter=",", header=header, comments="")
    print(f"\n[diagnose_voice] CSVs guardados en {METRICS_DIR}/voice_cm_*.csv")


if __name__ == "__main__":
    main()

"""
Evaluación standalone del modelo de navegación entrenado.
Genera reporte completo con verificación de criterios demo-ready.

Uso:
    uv run python evaluate.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_fscore_support)

from utils import (
    NAV_CLASSES,
    MODEL_NAV_PATH,
    DATA_NAV_DIR, METRICS_DIR,
)
from dataset import NavigationDataset, get_loaders
from model_nav import build_nav_model


def evaluate() -> None:
    device = torch.device('cpu')

    if not os.path.exists(MODEL_NAV_PATH):
        print(f"[ERROR] Modelo no encontrado: {MODEL_NAV_PATH}")
        print("  Ejecuta primero: uv run python train.py")
        return

    ckpt = torch.load(MODEL_NAV_PATH, map_location=device)
    model = build_nav_model(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    _, _, te_loader = get_loaders(augment_train=False)
    criterion = nn.CrossEntropyLoss()

    preds, labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for x, y in te_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            p = logits.argmax(1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
            total += x.size(0)

    acc      = sum(p == l for p, l in zip(preds, labels)) / total
    avg_loss = total_loss / total

    print(f"\n{'='*62}")
    print(f"  Modelo: NAV  |  Checkpoint: {MODEL_NAV_PATH}")
    print(f"  Test Accuracy: {acc:.4f}  |  Avg Loss: {avg_loss:.4f}")
    print(f"{'='*62}")
    print(f"\n{classification_report(labels, preds, target_names=NAV_CLASSES)}")

    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=range(len(NAV_CLASSES)), zero_division=0)

    # ── Verificación demo-ready ───────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  VERIFICACIÓN DEMO-READY")
    print(f"{'─'*62}")

    g_ok = acc >= 0.85
    c_ok = all(rec[i] >= 0.75 for i in range(len(NAV_CLASSES)))

    print(f"  [{'OK' if g_ok else 'FAIL'}] Accuracy global ≥ 85%:  {acc:.1%}")
    print(f"  [{'OK' if c_ok else 'FAIL'}] Todas las clases recall ≥ 75%:")
    for i, cls in enumerate(NAV_CLASSES):
        ok  = rec[i] >= 0.75
        tag = " ← BAJO" if not ok else ""
        print(f"       [{'OK' if ok else 'FAIL'}] {cls:<22}: {rec[i]:.1%}{tag}")

    all_ok = g_ok and c_ok
    print(f"\n  {'[DEMO-READY ✓]' if all_ok else '[NO DEMO-READY — ver items FAIL arriba]'}")

    # ── Curvas del checkpoint (si existen) ────────────────────────────────────
    if all(k in ckpt for k in ['tr_losses', 'vl_losses', 'tr_accs', 'vl_accs']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ep = range(1, len(ckpt['tr_losses']) + 1)
        ax1.plot(ep, ckpt['tr_losses'], label='Train')
        ax1.plot(ep, ckpt['vl_losses'], label='Val')
        ax1.set(xlabel='Epoch', ylabel='Loss', title='Curvas de Loss')
        ax1.legend(); ax1.grid(True)
        ax2.plot(ep, ckpt['tr_accs'],   label='Train')
        ax2.plot(ep, ckpt['vl_accs'],   label='Val')
        ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Curvas de Accuracy')
        ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        curve_path = os.path.join(METRICS_DIR, "nav_eval_curves.png")
        plt.savefig(curve_path, dpi=150); plt.close()
        print(f"\n  Curvas guardadas: {curve_path}")

    # ── Matriz de confusión 6×6 ───────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=NAV_CLASSES, yticklabels=NAV_CLASSES, ax=ax)
    ax.set(xlabel='Predicho', ylabel='Real',
           title='Matriz de Confusión — Navegación (6×6)')
    plt.tight_layout()
    cm_path = os.path.join(METRICS_DIR, "nav_eval_confusion.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"  Matriz de confusión guardada: {cm_path}")

    # ── Guardar reporte JSON ──────────────────────────────────────────────────
    report = {
        'accuracy':       acc,
        'loss':           avg_loss,
        'best_val_acc':   float(ckpt.get('best_val_acc', 0.0)),
        'demo_ready':     all_ok,
        'classes': {cls: {
            'precision': float(prec[i]),
            'recall':    float(rec[i]),
            'f1':        float(f1[i]),
            'support':   int(sup[i]),
        } for i, cls in enumerate(NAV_CLASSES)},
    }
    rpt_path = os.path.join(METRICS_DIR, "nav_eval_report.json")
    with open(rpt_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Reporte JSON guardado: {rpt_path}")


if __name__ == "__main__":
    evaluate()

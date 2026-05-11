"""
Entrenamiento de la CNN de navegación espacio-temporal.

Uso:
    uv run python train.py               # entrenar desde cero
    uv run python train.py --resume      # retomar desde el último checkpoint
    uv run python train.py --epochs 80   # sobrescribir número de epochs
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from utils import (
    LEARNING_RATE, NUM_EPOCHS_NAV,
    NAV_CLASSES, MODEL_NAV_PATH,
    DATA_NAV_DIR, METRICS_DIR,
)
from dataset import NavigationDataset, get_loaders
from model_nav import build_nav_model


# ── Dispositivo ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Epoch de entrenamiento ────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


# ── Evaluación ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        total   += x.size(0)
        preds.extend(p.cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, preds, labels


# ── Guardar artefactos ─────────────────────────────────────────────────────────

def save_curves(tr_loss, vl_loss, tr_acc, vl_acc, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tr_loss) + 1)
    ax1.plot(ep, tr_loss, label='Train'); ax1.plot(ep, vl_loss, label='Val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Curvas de Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(ep, tr_acc, label='Train'); ax2.plot(ep, vl_acc, label='Val')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Curvas de Accuracy')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Curvas guardadas: {path}")


def save_confusion(preds, labels, path: str) -> None:
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=NAV_CLASSES, yticklabels=NAV_CLASSES, ax=ax)
    ax.set(xlabel='Predicho', ylabel='Real', title='Matriz de Confusión — Navegación (6×6)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada: {path}")


# ── Entrenamiento principal ────────────────────────────────────────────────────

def train(resume: bool = False, epochs_override: int = None) -> None:
    device     = get_device()
    num_epochs = epochs_override or NUM_EPOCHS_NAV
    print(f"\n[train] Dispositivo: {device}  |  Epochs: {num_epochs}")

    tr_loader, vl_loader, te_loader = get_loaders(augment_train=True)
    print(f"  Train: {len(tr_loader.dataset)} | Val: {len(vl_loader.dataset)} | Test: {len(te_loader.dataset)}")

    model     = build_nav_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_ep  = 1
    best_acc  = 0.0
    tr_losses, vl_losses, tr_accs, vl_accs = [], [], [], []

    if resume and os.path.exists(MODEL_NAV_PATH):
        ckpt      = torch.load(MODEL_NAV_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_ep  = ckpt.get('epoch', 0) + 1
        best_acc  = ckpt.get('best_val_acc', 0.0)
        tr_losses = ckpt.get('tr_losses', [])
        vl_losses = ckpt.get('vl_losses', [])
        tr_accs   = ckpt.get('tr_accs', [])
        vl_accs   = ckpt.get('vl_accs', [])
        print(f"  Retomado desde epoch {start_ep} | Mejor val acc: {best_acc:.4f}")

    print(f"\n  Iniciando entrenamiento...\n")

    for ep in range(start_ep, num_epochs + 1):
        tr_l, tr_a       = train_epoch(model, tr_loader, optimizer, criterion, device)
        vl_l, vl_a, _, _ = eval_epoch(model,  vl_loader, criterion, device)
        scheduler.step(vl_a)

        tr_losses.append(tr_l); tr_accs.append(tr_a)
        vl_losses.append(vl_l); vl_accs.append(vl_a)

        tag = " ← MEJOR" if vl_a > best_acc else ""
        print(f"  Epoch {ep:3d}/{num_epochs}  "
              f"Loss {tr_l:.4f}/{vl_l:.4f}  "
              f"Acc {tr_a:.4f}/{vl_a:.4f}{tag}")

        if vl_a > best_acc:
            best_acc = vl_a
            torch.save({
                'epoch':        ep,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'best_val_acc': best_acc,
                'tr_losses':    tr_losses, 'vl_losses': vl_losses,
                'tr_accs':      tr_accs,   'vl_accs':   vl_accs,
            }, MODEL_NAV_PATH)

    # ── Evaluación final en test set ───────────────────────────────────────────
    print(f"\n  Cargando mejor checkpoint...")
    ckpt = torch.load(MODEL_NAV_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    te_l, te_a, preds, labels = eval_epoch(model, te_loader, criterion, device)
    print(f"\n  Test Loss: {te_l:.4f}  |  Test Accuracy: {te_a:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=NAV_CLASSES)}")

    save_curves(tr_losses, vl_losses, tr_accs, vl_accs,
                os.path.join(METRICS_DIR, "nav_curves.png"))
    save_confusion(preds, labels,
                   os.path.join(METRICS_DIR, "nav_confusion.png"))

    metrics = {
        'test_accuracy':     te_a,
        'test_loss':         te_l,
        'best_val_accuracy': best_acc,
        'report': classification_report(labels, preds,
                                        target_names=NAV_CLASSES,
                                        output_dict=True),
    }
    with open(os.path.join(METRICS_DIR, "nav_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Entrenamiento completo. Mejor val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(args.resume, args.epochs)

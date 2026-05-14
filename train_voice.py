# train_voice.py
"""
Entrenamiento del VoiceCNN para reconocimiento de comandos de voz.

Uso:
    uv run python train_voice.py --epochs 30

Requiere que data/voice/ esté generado:
    uv run python generate_voice_dataset.py
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from voice_dataset import VoiceDataset, VOICE_CLASSES
from model_voice import build_voice_model, count_parameters

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")


def train(epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
          val_split: float = 0.15) -> None:

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[train_voice] Device: {device}")

    full_ds = VoiceDataset(augment=False)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = build_voice_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"Parámetros: {count_parameters(model):,}")
    print(f"Train: {n_train}  Val: {n_val}  Clases: {VOICE_CLASSES}\n")

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1).cpu()
            y_cpu = y.cpu()
            train_correct += (preds == y_cpu).sum().item()
            train_total   += len(y_cpu)
            train_loss    += loss.item() * len(y_cpu)

        train_acc  = train_correct / train_total
        train_loss = train_loss    / train_total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(device), y.to(device)
                preds = model(x).argmax(1).cpu()
                y_cpu = y.cpu()
                val_correct += (preds == y_cpu).sum().item()
                val_total   += len(y_cpu)

        val_acc = val_correct / val_total
        scheduler.step()

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                        "classes": VOICE_CLASSES}, MODEL_VOICE_PATH)
            flag = "  ← guardado"

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_acc:.3f}  val={val_acc:.3f}  "
              f"loss={train_loss:.4f}{flag}")

    print(f"\nMejor val_acc: {best_val_acc:.3f}")
    print(f"Modelo guardado en: {MODEL_VOICE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)

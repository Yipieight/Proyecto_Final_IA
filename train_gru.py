# train_gru.py
"""
Entrena VoiceGRU sobre pares sintéticos de embeddings extraídos de VoiceCNN.

Flujo:
  1. Carga embeddings/<CLASE>.npy para cada clase (extraídos por extract_embeddings.py)
  2. Construye pares sintéticos (embed_palabra1, embed_palabra2) → label compuesto
  3. Split 85/15 por clase (estratificado)
  4. Entrena VoiceGRU con Adam + StepLR, 30 épocas
  5. Guarda models/gru_model.pth + métricas en metrics/gru_report.json

Uso:
    uv run python train_gru.py
    uv run python train_gru.py --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model_gru import (
    VoiceGRU,
    COMPOUND_CLASSES,
    COMPOUND_CLASS_IDX,
    COMPOUND_WORD_PAIRS,
    NUM_COMPOUND,
    EMBED_DIM,
)

EMBEDDINGS_DIR = Path("embeddings")
MODEL_OUT      = Path("models") / "gru_model.pth"
METRICS_OUT    = Path("metrics") / "gru_report.json"
PAIRS_PER_CLASS = 1000   # pares sintéticos por clase de comando compuesto

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Dataset sintético ─────────────────────────────────────────────────────────

class CompoundDataset(Dataset):
    """
    Construye pares (embed_w1, embed_w2) → label_compuesto desde los .npy.

    Para cada clase compuesta se muestrean aleatoriamente (con reemplazo)
    PAIRS_PER_CLASS pares independientes de sus dos clases fuente.
    """

    def __init__(self, embeddings: dict[str, np.ndarray], pairs_per_class: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        X, Y = [], []

        for compound_name, (w1, w2) in COMPOUND_WORD_PAIRS.items():
            label = COMPOUND_CLASS_IDX[compound_name]
            e1 = embeddings[w1]   # (N1, 64)
            e2 = embeddings[w2]   # (N2, 64)

            idx1 = rng.integers(0, len(e1), size=pairs_per_class)
            idx2 = rng.integers(0, len(e2), size=pairs_per_class)

            for i, j in zip(idx1, idx2):
                pair = np.stack([e1[i], e2[j]], axis=0)   # (2, 64)
                X.append(pair)
                Y.append(label)

        self.X = torch.from_numpy(np.array(X, dtype=np.float32))  # (N, 2, 64)
        self.Y = torch.tensor(Y, dtype=torch.long)                 # (N,)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def load_embeddings() -> dict[str, np.ndarray]:
    embeddings = {}
    for cls_name in set(w for pair in COMPOUND_WORD_PAIRS.values() for w in pair):
        path = EMBEDDINGS_DIR / f"{cls_name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta primero: uv run python extract_embeddings.py"
            )
        embeddings[cls_name] = np.load(path)
        print(f"  {cls_name:<14}: {embeddings[cls_name].shape[0]} embeddings")
    return embeddings


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Y)
        correct    += (logits.argmax(1) == Y).sum().item()
        total      += len(Y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss   = criterion(logits, Y)
        total_loss += loss.item() * len(Y)
        correct    += (logits.argmax(1) == Y).sum().item()
        total      += len(Y)
    return total_loss / total, correct / total


@torch.no_grad()
def confusion_matrix_np(model, loader, device, n_classes):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        preds = model(X).argmax(1).cpu().numpy()
        for t, p in zip(Y.cpu().numpy(), preds):
            cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--pairs",        type=int,   default=PAIRS_PER_CLASS,
                        help="Pares sintéticos por clase")
    parser.add_argument("--val-split",    type=float, default=0.15)
    args = parser.parse_args()

    print(f"Dispositivo: {DEVICE}")
    print("\nCargando embeddings...")
    embeddings = load_embeddings()

    print(f"\nConstruyendo dataset ({args.pairs} pares/clase × {NUM_COMPOUND} clases)...")
    dataset = CompoundDataset(embeddings, args.pairs)
    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model     = VoiceGRU().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nEntrenando VoiceGRU — {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parámetros")
    print(f"{'Época':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'val_loss':>9}  {'val_acc':>8}  {'lr':>8}")
    print("-" * 60)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc,  4))
        history["val_loss"].append(round(vl_loss,  4))
        history["val_acc"].append(round(vl_acc,    4))

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.2%}  {vl_loss:>9.4f}  {vl_acc:>8.2%}  {lr_now:>8.6f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_OUT)

    elapsed = time.time() - t0
    print(f"\nEntrenamiento completado en {elapsed:.1f}s")
    print(f"Mejor val_acc: {best_val_acc:.2%}  →  guardado en {MODEL_OUT}")

    # Cargar mejor modelo para métricas finales
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE, weights_only=True))
    cm = confusion_matrix_np(model, val_loader, DEVICE, NUM_COMPOUND)

    # Métricas por clase
    per_class = {}
    for i, cls_name in enumerate(COMPOUND_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "n": int(cm[i].sum())}

    report = {
        "val_accuracy": round(best_val_acc, 4),
        "epochs": args.epochs,
        "pairs_per_class": args.pairs,
        "per_class": per_class,
        "history": history,
        "confusion_matrix": cm.tolist(),
    }

    METRICS_OUT.parent.mkdir(exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Reporte guardado en {METRICS_OUT}")

    print("\nMétricas por clase (val):")
    print(f"  {'Clase':<24}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>5}")
    print("  " + "-" * 52)
    for cls_name, m in per_class.items():
        print(f"  {cls_name:<24}  {m['precision']:>6.2%}  {m['recall']:>6.2%}  {m['f1']:>6.2%}  {m['n']:>5}")


if __name__ == "__main__":
    main()

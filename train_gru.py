# train_gru.py
"""
Entrena VoiceGRU sobre audios compuestos construidos programáticamente.

Flujo:
  1. Para cada clase compuesta (ej. ADELANTE_IZQUIERDA), carga WAVs de ambas
     palabras del dataset existente (data/voice/).
  2. Concatena: audio_palabra1 + silencio_aleatorio(50-200ms) + audio_palabra2
     → audio compuesto de la frase completa hablada de corrido.
  3. Calcula compute_mel_sequence → (T_MAX=300, 64) por muestra.
  4. Entrena VoiceGRU con Adam + StepLR, 30 épocas.
  5. Guarda models/gru_model.pth

No se graban nuevos audios — se reutilizan los 32,112 WAVs del dataset de voz.

Uso:
    uv run python train_gru.py
    uv run python train_gru.py --epochs 40 --pairs 800
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from voice_dataset import DATA_VOICE_DIR
from model_gru import (
    VoiceGRU,
    compute_mel_sequence,
    COMPOUND_CLASSES,
    COMPOUND_CLASS_IDX,
    COMPOUND_WORD_PAIRS,
    NUM_COMPOUND,
    T_MAX,
)

MODEL_OUT   = Path("models") / "gru_model.pth"
METRICS_OUT = Path("metrics") / "gru_report.json"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CompoundAudioDataset(Dataset):
    """
    Genera audios compuestos concatenando pares de WAVs del dataset existente.

    Para cada clase compuesta:
      audio_w1 + silencio(50–200ms) + audio_w2  →  compute_mel_sequence  →  (T_MAX, 64)

    Los mel-spectrograms se pre-computan en RAM al construir el dataset.
    """

    def __init__(
        self,
        root: Path = DATA_VOICE_DIR,
        pairs_per_class: int = 500,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Cargar rutas de WAVs por clase fuente
        wav_paths: dict[str, list[Path]] = {}
        needed = set(w for pair in COMPOUND_WORD_PAIRS.values() for w in pair)
        for cls_name in needed:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(
                    f"Carpeta no encontrada: {cls_dir}\n"
                    "  Ejecuta primero: uv run python generate_voice_dataset.py"
                )
            paths = sorted(cls_dir.glob("*.wav"))
            if not paths:
                raise FileNotFoundError(f"Sin WAVs en {cls_dir}")
            wav_paths[cls_name] = paths

        # Construir pares (path_w1, path_w2, label)
        pairs: list[tuple[Path, Path, int]] = []
        for compound_name, (w1, w2) in COMPOUND_WORD_PAIRS.items():
            label = COMPOUND_CLASS_IDX[compound_name]
            p1    = wav_paths[w1]
            p2    = wav_paths[w2]
            idx1  = rng.integers(0, len(p1), size=pairs_per_class)
            idx2  = rng.integers(0, len(p2), size=pairs_per_class)
            for i, j in zip(idx1, idx2):
                pairs.append((p1[i], p2[j], label))

        # Pre-computar mel sequences
        total = len(pairs)
        print(f"[dataset] Pre-computando {total} audios compuestos...")
        t0 = time.time()

        X_list, Y_list = [], []
        for k, (path1, path2, label) in enumerate(pairs):
            a1, sr1 = sf.read(str(path1), dtype="float32")
            a2, sr2 = sf.read(str(path2), dtype="float32")
            if a1.ndim > 1: a1 = a1.mean(axis=1)
            if a2.ndim > 1: a2 = a2.mean(axis=1)

            # Silencio aleatorio entre palabras (50–200 ms)
            gap_samples = int(sr1 * rng.integers(50, 201) / 1000)
            silence     = np.zeros(gap_samples, dtype=np.float32)
            compound    = np.concatenate([a1, silence, a2])

            mel = compute_mel_sequence(compound, sr1)
            X_list.append(mel)
            Y_list.append(label)

            if (k + 1) % 500 == 0:
                print(f"  {k+1}/{total}  ({time.time()-t0:.0f}s)")

        self.X = torch.from_numpy(np.stack(X_list))       # (N, T_MAX, N_MELS)
        self.Y = torch.tensor(Y_list, dtype=torch.long)   # (N,)
        mem_mb = self.X.numel() * 4 / 1024 / 1024
        print(f"[dataset] Listo en {time.time()-t0:.1f}s  (~{mem_mb:.0f} MB en RAM)")

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# ── Entrenamiento ─────────────────────────────────────────────────────────────

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
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--pairs",     type=int,   default=500,
                        help="Pares de audio compuesto por clase (default: 500)")
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    print(f"Dispositivo: {DEVICE}")
    print(f"Secuencia temporal: T_MAX={T_MAX} frames (3 s)\n")

    dataset = CompoundAudioDataset(pairs_per_class=args.pairs)
    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}  |  Val: {n_val}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model     = VoiceGRU().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Entrenando VoiceGRU — {n_params:,} parámetros")
    print(f"{'Época':>5}  {'tr_loss':>8}  {'tr_acc':>7}  {'val_loss':>9}  {'val_acc':>8}")
    print("-" * 52)

    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    t0            = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader,   criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc,   4))
        history["val_loss"].append(round(vl_loss,   4))
        history["val_acc"].append(round(vl_acc,     4))

        print(f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.2%}  {vl_loss:>9.4f}  {vl_acc:>8.2%}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"\nEntrenamiento completado en {time.time()-t0:.1f}s")
    print(f"Mejor val_acc: {best_val_acc:.2%}  →  {MODEL_OUT}")

    # Métricas finales
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE, weights_only=True))
    cm = confusion_matrix_np(model, val_loader, DEVICE, NUM_COMPOUND)

    per_class = {}
    for i, cls_name in enumerate(COMPOUND_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "n":         int(cm[i].sum()),
        }

    report = {
        "val_accuracy":   round(best_val_acc, 4),
        "epochs":         args.epochs,
        "pairs_per_class": args.pairs,
        "t_max":          T_MAX,
        "per_class":      per_class,
        "history":        history,
        "confusion_matrix": cm.tolist(),
    }
    METRICS_OUT.parent.mkdir(exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Reporte guardado en {METRICS_OUT}")

    print(f"\n{'Clase':<24}  {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>5}")
    print("  " + "-" * 48)
    for cls_name, m in per_class.items():
        print(f"  {cls_name:<22}  {m['precision']:>6.2%}  {m['recall']:>6.2%}  {m['f1']:>6.2%}  {m['n']:>5}")


if __name__ == "__main__":
    main()

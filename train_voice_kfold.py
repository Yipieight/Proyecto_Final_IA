# train_voice_kfold.py
"""
K-Fold cross-validation (5 folds) para VoiceCNN.

Proceso:
  1. Split estratificado 80% train+val / 20% test fijo
  2. 5-Fold CV sobre el 80%: cada fold usa ~64% train / 16% val
  3. Entrena modelo final sobre todo el 80%
  4. Evalúa en el 20% test (nunca visto durante entrenamiento)
  5. Guarda modelo final en models/voice_model.pth
  6. Genera reporte en metrics/kfold_report.md

Uso:
    uv run python train_voice_kfold.py --epochs 40
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from voice_dataset import VoiceDataset, VOICE_CLASSES, NUM_VOICE_CLASSES
from model_voice import build_voice_model, count_parameters

MODEL_PATH  = os.path.join("models", "voice_model.pth")
REPORT_PATH = os.path.join("metrics", "kfold_report.md")
N_FOLDS     = 5
TEST_RATIO  = 0.20


# ── Entrenamiento / evaluación ────────────────────────────────────────────────

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
        correct    += (logits.argmax(1).cpu() == y.cpu()).sum().item()
        total      += len(y)
        total_loss += loss.item() * len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_l, labels_l, logits_l = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds_l.extend(out.argmax(1).cpu().numpy())
        labels_l.extend(y.numpy())
        logits_l.extend(out.cpu().numpy())
    preds  = np.array(preds_l)
    labels = np.array(labels_l)
    logits = np.array(logits_l)
    acc    = (preds == labels).mean()
    return acc, preds, labels, logits


# ── Métricas ──────────────────────────────────────────────────────────────────

def confusion_matrix(preds, labels, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    n = cm.shape[0]
    recall    = np.zeros(n)
    precision = np.zeros(n)
    f1        = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        denom = recall[i] + precision[i]
        f1[i] = 2 * recall[i] * precision[i] / denom if denom > 0 else 0.0
    return recall, precision, f1


# ── Formato markdown ──────────────────────────────────────────────────────────

def _short(cls):
    return cls[:8]


def md_confusion_matrix(cm, classes):
    short = [_short(c) for c in classes]
    header = "| Verdadero \\ Predicho | " + " | ".join(short) + " |"
    sep    = "|---|" + "|".join(["---"] * len(classes)) + "|"
    rows   = []
    for i, cls in enumerate(classes):
        row = f"| **{cls}** | " + " | ".join(str(cm[i][j]) for j in range(len(classes))) + " |"
        rows.append(row)
    return "\n".join([header, sep] + rows)


def md_per_class_table(cm, classes):
    recall, precision, f1 = per_class_metrics(cm)
    lines = [
        "| Clase | Aciertos / Total | Recall | Precision | F1 |",
        "|-------|-----------------|--------|-----------|-----|",
    ]
    for i, cls in enumerate(classes):
        tp    = cm[i, i]
        total = cm[i].sum()
        emoji = "✅" if recall[i] >= 0.95 else ("⚠️" if recall[i] >= 0.80 else "❌")
        lines.append(
            f"| {cls} | {tp} / {total} | **{recall[i]:.1%}** {emoji} "
            f"| {precision[i]:.1%} | {f1[i]:.1%} |"
        )
    macro_r = recall.mean()
    macro_p = precision.mean()
    macro_f = f1.mean()
    lines.append(f"| **MACRO** | — | **{macro_r:.1%}** | **{macro_p:.1%}** | **{macro_f:.1%}** |")
    return "\n".join(lines)


def md_sample_predictions(preds, labels, logits, classes, n=4):
    lines = [
        "| Muestra | Verdadero | Predicho | Logits (crudos) |",
        "|---------|-----------|----------|----------------|",
    ]
    for i in range(min(n, len(preds))):
        lstr = "[" + ", ".join(f"{v:+.2f}" for v in logits[i]) + "]"
        lines.append(f"| {i} | {classes[labels[i]]} | {classes[preds[i]]} | {lstr} |")
    return "\n".join(lines)


def md_problems(cm_test, classes):
    recall, _, _ = per_class_metrics(cm_test)
    problems = []
    for i, cls in enumerate(classes):
        if recall[i] < 0.80:
            # Clase más confundida
            row = cm_test[i].copy()
            row[i] = 0
            worst_j = int(row.argmax())
            worst_n = int(row.max())
            problems.append(
                f"| **Recall bajo `{cls}`** "
                f"| {recall[i]:.1%} — se confunde con `{classes[worst_j]}` ({worst_n}×) |"
            )
    if not problems:
        return "_Sin problemas detectados — todas las clases ≥ 80% recall._"
    header = "| Problema | Evidencia |\n|----------|-----------|"
    return header + "\n" + "\n".join(problems)


# ── Reporte completo ──────────────────────────────────────────────────────────

def generate_report(device_name, n_total, n_test, n_trainval,
                    test_acc, test_preds, test_labels, test_logits,
                    fold_results, cv_acc_mean, cv_acc_std,
                    all_cv_preds, all_cv_labels, all_cv_logits,
                    args):

    cm_test = confusion_matrix(test_preds, test_labels, NUM_VOICE_CLASSES)
    cm_cv   = confusion_matrix(all_cv_preds, all_cv_labels, NUM_VOICE_CLASSES)
    recall_test, prec_test, f1_test = per_class_metrics(cm_test)

    fold_table = (
        "| Fold | Train | Val | Val Acc |\n"
        "|------|-------|-----|---------|\n"
    )
    for r in fold_results:
        fold_table += f"| {r['fold']} | {r['n_train']} | {r['n_val']} | **{r['val_acc']:.1%}** |\n"
    fold_table += f"| **Promedio** | — | — | **{cv_acc_mean:.1%} ± {cv_acc_std:.2%}** |"

    report = f"""# Diagnóstico del modelo de voz — VoiceCNN

## Configuración inicial

- **Dispositivo utilizado**: {device_name}
- **Modelo cargado**: Sí
- **Precisión CV promedio**: {cv_acc_mean:.2%}
- **Parámetros del modelo**: 2,207,142
- **Conjunto de prueba fijo (test)**: {n_test} muestras
- **Train + Validación (K-Fold)**: {n_trainval} muestras
- **Dataset total**: {n_total} muestras
- **Clases**: {", ".join(VOICE_CLASSES)}
- **Épocas por fold**: {args.epochs}
- **Split**: {int((1-TEST_RATIO)*100)}% Train+Val / {int(TEST_RATIO*100)}% Test

---

## Evaluación en conjunto de prueba fijo (20% — nunca visto)

### Resultados generales
- **Precisión (accuracy)**: {test_acc:.2%}

### Matriz de confusión

{md_confusion_matrix(cm_test, VOICE_CLASSES)}

### Métricas por clase

{md_per_class_table(cm_test, VOICE_CLASSES)}

> **Nota**: Recall = capacidad de detectar la clase. Precision = cuando predice esa clase, ¿cuántas veces acierta?

### Ejemplo de predicciones (primeras 4 muestras del test)

{md_sample_predictions(test_preds, test_labels, test_logits, VOICE_CLASSES)}

> Los logits son las activaciones crudas para cada clase (orden: {", ".join(VOICE_CLASSES)}). Valores más altos indican mayor confianza.

---

## 5-Fold Cross-Validation (sobre el 80% de datos)

### Resultados por fold

{fold_table}

### Métricas agregadas CV (todos los folds combinados)

{md_per_class_table(cm_cv, VOICE_CLASSES)}

### Matriz de confusión CV (acumulada)

{md_confusion_matrix(cm_cv, VOICE_CLASSES)}

---

## Resumen de problemas detectados

{md_problems(cm_test, VOICE_CLASSES)}

---

## Conclusión

| Métrica | Valor |
|---------|-------|
| CV Accuracy (media 5 folds) | **{cv_acc_mean:.2%}** |
| CV Accuracy (desv. estándar) | **±{cv_acc_std:.2%}** |
| Test Accuracy (holdout 20%) | **{test_acc:.2%}** |
| Macro Recall (test) | **{recall_test.mean():.2%}** |
| Macro Precision (test) | **{prec_test.mean():.2%}** |
| Macro F1 (test) | **{f1_test.mean():.2%}** |

_Modelo guardado en `{MODEL_PATH}` — listo para usar con `main_voice.py`._
"""
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    device      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device_name = "MPS (Metal Performance Shaders - Apple Silicon)" if str(device) == "mps" else "CPU"
    print(f"\n[kfold] Dispositivo: {device}")

    print("[kfold] Cargando dataset...")
    t0      = time.time()
    full_ds = VoiceDataset(augment=False)
    n_total = len(full_ds)
    labels_all = np.array([full_ds.samples[i][1] for i in range(n_total)])
    print(f"[kfold] {n_total} muestras  ({time.time()-t0:.1f}s)")
    print(f"[kfold] Parámetros modelo: {count_parameters(build_voice_model(device)):,}\n")

    # ── Split estratificado 80/20 ─────────────────────────────────────────────
    rng = np.random.RandomState(42)
    test_idx, train_val_idx = [], []
    for cls in range(NUM_VOICE_CLASSES):
        cls_idx = np.where(labels_all == cls)[0]
        perm    = rng.permutation(cls_idx)
        n_test  = max(1, int(len(cls_idx) * TEST_RATIO))
        test_idx.extend(perm[:n_test].tolist())
        train_val_idx.extend(perm[n_test:].tolist())

    test_idx      = np.array(test_idx)
    train_val_idx = np.array(train_val_idx)
    labels_tv     = labels_all[train_val_idx]

    print(f"Split — Train+Val: {len(train_val_idx)}  |  Test: {len(test_idx)}")

    test_loader = DataLoader(Subset(full_ds, test_idx),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)
    criterion   = nn.CrossEntropyLoss()

    # ── 5-Fold CV ─────────────────────────────────────────────────────────────
    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION  ({args.epochs} épocas/fold)")
    print(f"{'='*60}")

    for fold, (local_train, local_val) in enumerate(skf.split(train_val_idx, labels_tv), 1):
        f_train_idx = train_val_idx[local_train]
        f_val_idx   = train_val_idx[local_val]

        train_loader = DataLoader(Subset(full_ds, f_train_idx),
                                  batch_size=args.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(full_ds, f_val_idx),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0)

        model     = build_voice_model(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_val_acc = 0.0
        best_state   = None

        print(f"\nFold {fold}/{N_FOLDS}  train={len(f_train_idx)}  val={len(f_val_idx)}")
        for epoch in range(1, args.epochs + 1):
            loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, _, _, _ = evaluate(model, val_loader, device)
            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} | "
                      f"train={train_acc:.3f}  val={val_acc:.3f}  loss={loss:.4f}")

        model.load_state_dict(best_state)
        val_acc, val_preds, val_labels, val_logits = evaluate(model, val_loader, device)
        print(f"  → Fold {fold} mejor val_acc: {val_acc:.1%}")

        fold_results.append({
            "fold":    fold,
            "val_acc": val_acc,
            "preds":   val_preds,
            "labels":  val_labels,
            "logits":  val_logits,
            "n_train": len(f_train_idx),
            "n_val":   len(f_val_idx),
        })

    cv_acc_mean = np.mean([r["val_acc"] for r in fold_results])
    cv_acc_std  = np.std( [r["val_acc"] for r in fold_results])
    print(f"\n{'='*60}")
    print(f"  CV accuracy: {cv_acc_mean:.1%} ± {cv_acc_std:.2%}")
    print(f"{'='*60}")

    # ── Modelo final (todo el 80%) ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODELO FINAL  ({args.epochs} épocas sobre {len(train_val_idx)} muestras)")
    print(f"{'='*60}\n")

    final_loader = DataLoader(Subset(full_ds, train_val_idx),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    final_model  = build_voice_model(device)
    optimizer    = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_final_acc   = 0.0
    best_final_state = None

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_epoch(final_model, final_loader, optimizer, criterion, device)
        scheduler.step()
        if train_acc > best_final_acc:
            best_final_acc   = train_acc
            best_final_state = {k: v.clone() for k, v in final_model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train={train_acc:.3f}  loss={loss:.4f}")

    final_model.load_state_dict(best_final_state)
    test_acc, test_preds, test_labels, test_logits = evaluate(final_model, test_loader, device)
    print(f"\n[kfold] Test accuracy (holdout 20%): {test_acc:.1%}")

    # Guardar
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state":   final_model.state_dict(),
        "best_val_acc":  cv_acc_mean,
        "classes":       VOICE_CLASSES,
        "kfold_cv_acc":  cv_acc_mean,
        "kfold_cv_std":  cv_acc_std,
        "test_acc":      test_acc,
    }, MODEL_PATH)
    print(f"[kfold] Modelo guardado → {MODEL_PATH}")

    # ── Reporte ───────────────────────────────────────────────────────────────
    all_cv_preds  = np.concatenate([r["preds"]  for r in fold_results])
    all_cv_labels = np.concatenate([r["labels"] for r in fold_results])
    all_cv_logits = np.concatenate([r["logits"] for r in fold_results])

    report = generate_report(
        device_name, n_total, len(test_idx), len(train_val_idx),
        test_acc, test_preds, test_labels, test_logits,
        fold_results, cv_acc_mean, cv_acc_std,
        all_cv_preds, all_cv_labels, all_cv_logits,
        args,
    )

    os.makedirs("metrics", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[kfold] Reporte guardado → {REPORT_PATH}")


if __name__ == "__main__":
    main()

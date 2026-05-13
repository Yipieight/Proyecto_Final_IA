"""
K-Fold cross-validation con separación por sesión de video.

Estrategia:
  - ~500 imgs como test fijo (held-out, sesiones completas por clase)
  - Resto: 5-Fold CV por sesión de video (no por frame individual)
  - Nunca frames del mismo clip en train y val simultáneamente

Uso:
    uv run python train_kfold.py
    uv run python train_kfold.py --folds 5 --epochs 40
"""

import os
import argparse
import json
import time
import random
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from utils import (
    IMG_WIDTH, IMG_HEIGHT, FRAME_STACK,
    NAV_CLASSES, NAV_CLASS_IDX, NAV_IDX_CLASS,
    DATA_NAV_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS_NAV,
    MODEL_NAV_PATH, METRICS_DIR,
)
from preprocessing import preprocess_frame
from model_nav import build_nav_model


# ── Augmentación optimizada: una sola llamada por stack ───────────────────────

def augment_stack(stack: np.ndarray) -> np.ndarray:
    """
    Aplica UNA augmentación aleatoria suave a todo el stack (3 frames).
    Temporalmente consistente — más correcto que aplicar 3 transformaciones
    distintas a frames que representan el mismo momento.

    Augmentación SUAVE: el modelo es pequeño (~660K params) y con augmentación
    agresiva no logra converger establemente. Rangos reducidos:
      brillo 0.85-1.15 (antes 0.6-1.4)
      contraste 0.9-1.1 (antes 0.7-1.3)
      rotación ±5° (antes ±12°)
      ruido σ=0.01 (antes 0.03)
      crop 92-100% (antes 85-100%)
    """
    t = torch.from_numpy(stack)

    if random.random() < 0.5:
        t = TF.adjust_brightness(t, random.uniform(0.85, 1.15))
    if random.random() < 0.35:
        t = TF.adjust_contrast(t, random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        t = TF.rotate(t.unsqueeze(0), random.uniform(-5, 5)).squeeze(0)
    if random.random() < 0.3:
        t = torch.clamp(t + torch.randn_like(t) * 0.01, 0.0, 1.0)
    if random.random() < 0.25:
        h, w   = t.shape[-2], t.shape[-1]
        frac   = random.uniform(0.92, 1.0)
        ch, cw = int(h * frac), int(w * frac)
        i      = random.randint(0, h - ch)
        j      = random.randint(0, w - cw)
        t      = TF.resize(TF.crop(t, i, j, ch, cw), [IMG_HEIGHT, IMG_WIDTH])

    return t.numpy()


# ── Extrae ID de sesión del nombre de archivo ──────────────────────────────────

def frame_num_of(path: str) -> int:
    """
    Extrae el número de frame del nombre del archivo.
    RECTA_20260512_104509_660890_000012.jpg → 12

    Este número se RESETEA a 0 al inicio de cada video, lo que nos permite
    detectar los límites físicos entre videos (más confiable que el timestamp
    de extracción, que puede ser continuo si los videos se extraen en cadena).
    """
    name = os.path.basename(path).rsplit('.', 1)[0]
    return int(name.split('_')[-1])


# ── Carga todas las muestras ───────────────────────────────────────────────────

def load_samples() -> list:
    """
    Devuelve lista de (path, label, video_id).
    El video_id se asigna detectando resets del frame number en archivos
    ordenados por nombre — cada reset marca el inicio de un nuevo video físico.
    """
    samples = []
    for cls in NAV_CLASSES:
        cls_dir = os.path.join(DATA_NAV_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        label = NAV_CLASS_IDX[cls]

        files = sorted(f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        video_idx, prev_frame = 0, -1
        for fname in files:
            path  = os.path.join(cls_dir, fname)
            f_num = frame_num_of(path)
            if f_num < prev_frame:           # reset → nuevo video físico
                video_idx += 1
            video_id = f"{cls}_v{video_idx}"
            samples.append((path, label, video_id))
            prev_frame = f_num

    return samples


# ── Separar test fijo por sesiones completas ───────────────────────────────────

def split_held_out_test(samples: list, target: int = 500):
    """
    Separa ~target imágenes como test fijo tomando VIDEOS COMPLETOS por clase.
    Nunca mezcla frames del mismo video físico entre train y test.
    Devuelve (test_samples, cv_samples).
    """
    by_cls_video = defaultdict(lambda: defaultdict(list))
    for s in samples:
        path, label, vid = s
        by_cls_video[label][vid].append(s)

    test, cv = [], []
    for label in sorted(by_cls_video):
        videos     = sorted(by_cls_video[label].keys())
        cls_total  = sum(len(v) for v in by_cls_video[label].values())
        target_cls = max(1, int(cls_total * (target / len(samples))))

        # Toma videos completos desde el último hasta cubrir target_cls
        test_vids, accumulated = set(), 0
        for vid in reversed(videos):
            test_vids.add(vid)
            accumulated += len(by_cls_video[label][vid])
            if accumulated >= target_cls:
                break

        for vid, imgs in by_cls_video[label].items():
            (test if vid in test_vids else cv).extend(imgs)

    return test, cv


# ── K-Fold por sesión ──────────────────────────────────────────────────────────

def make_folds(cv_samples: list, n_folds: int = 5) -> list:
    """
    Divide cv_samples en n_folds folds asignando VIDEOS COMPLETOS de forma
    round-robin por clase. Cada video físico cae entero en un solo fold.
    Devuelve lista de (train_samples, val_samples).
    """
    by_cls_video = defaultdict(set)
    for path, label, vid in cv_samples:
        by_cls_video[label].add(vid)

    video_fold: dict = {}
    for label in sorted(by_cls_video):
        for i, vid in enumerate(sorted(by_cls_video[label])):
            video_fold[(label, vid)] = i % n_folds

    folds = []
    for fold_idx in range(n_folds):
        train_s, val_s = [], []
        for path, label, vid in cv_samples:
            if video_fold[(label, vid)] == fold_idx:
                val_s.append((path, label, vid))
            else:
                train_s.append((path, label, vid))
        folds.append((train_s, val_s))

    return folds


# ── Cache global de imágenes preprocesadas ─────────────────────────────────────

def build_image_cache(samples: list) -> tuple:
    """
    Precarga TODAS las imágenes del dataset una sola vez.
    Se reusa entre todos los folds en lugar de reconstruirse 10 veces.

    Devuelve (cache, folder_lists, pos_map):
      cache       : dict path → ndarray (H, W) preprocesado
      folder_lists: dict folder → lista ordenada de paths
      pos_map     : dict path → (folder, position_in_folder_list)
    """
    by_folder: dict = defaultdict(list)
    for path, _, _ in samples:
        by_folder[os.path.dirname(path)].append(path)
    folder_lists = {k: sorted(v) for k, v in by_folder.items()}

    pos_map: dict[str, tuple] = {}
    for folder, paths in folder_lists.items():
        for i, p in enumerate(paths):
            pos_map[p] = (folder, i)

    total = sum(len(v) for v in folder_lists.values())
    print(f"\n[Cache] Precargando {total} imágenes preprocesadas en RAM...")
    t0 = time.time()

    cache: dict[str, np.ndarray] = {}
    loaded = 0
    for folder, paths in folder_lists.items():
        for p in paths:
            bgr = cv2.imread(p)
            cache[p] = (preprocess_frame(bgr) if bgr is not None
                        else np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32))
            loaded += 1
            if loaded % 500 == 0:
                print(f"  {loaded}/{total} imgs ({100*loaded/total:.0f}%)", flush=True)

    dt = time.time() - t0
    print(f"[Cache] {total} imgs cargadas en {dt:.1f}s "
          f"(~{total*64*64*4/1024/1024:.0f}MB de RAM)\n")
    return cache, folder_lists, pos_map


# ── Dataset ────────────────────────────────────────────────────────────────────

class FoldDataset(Dataset):
    """
    Dataset que opera sobre un cache compartido entre folds.
    NO carga imágenes en __init__ — el cache se construye una vez en main().
    """

    def __init__(self, samples: list,
                 cache: dict, folder_lists: dict, pos_map: dict,
                 do_augment: bool = False):
        self.do_augment   = do_augment
        self._cache       = cache
        self._folder_lists = folder_lists
        self._pos_map     = pos_map
        self.samples      = [(path, label) for path, label, _ in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        folder, i   = self._pos_map[path]
        folder_list = self._folder_lists[folder]

        # Construir stack directamente como ndarray contiguo — sin copias intermedias
        stack = np.empty((FRAME_STACK, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        for k, offset in enumerate(range(-(FRAME_STACK - 1), 1)):
            j = max(0, min(len(folder_list) - 1, i + offset))
            stack[k] = self._cache[folder_list[j]]

        # Una sola augmentación al stack completo (temporalmente consistente)
        if self.do_augment:
            stack = augment_stack(stack)

        return torch.from_numpy(stack), torch.tensor(label, dtype=torch.long)


# ── Entrenamiento de un fold ───────────────────────────────────────────────────

def train_fold(train_samples: list, val_samples: list,
               cache: dict, folder_lists: dict, pos_map: dict,
               epochs: int, fold_idx: int, device) -> tuple:
    train_ds = FoldDataset(train_samples, cache, folder_lists, pos_map, do_augment=True)
    val_ds   = FoldDataset(val_samples,   cache, folder_lists, pos_map, do_augment=False)

    # pin_memory acelera transfer CPU→GPU; sin workers para evitar hang en macOS Python 3.14
    kw       = dict(num_workers=0, pin_memory=(device.type != 'cpu'))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)

    # LR=1e-4 (10× más bajo que utils.LEARNING_RATE) — previene divergencia
    # con augmentación temporal y dataset pequeño por fold
    model     = build_nav_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    best_state   = None
    history      = []

    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # ── TRAIN ─────────────────────────────────────────────────────────────
        # CPU para comparaciones (MPS tiene bugs con == int64). Loss/forward en device.
        model.train()
        tr_loss_sum = 0.0
        tr_correct  = 0
        tr_total    = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out  = model(x)
            loss = criterion(out, y_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss_sum += loss.item() * y_dev.size(0)
            # Comparación en CPU para evitar bug de MPS con int comparisons
            preds_cpu = out.detach().argmax(1).cpu()
            y_cpu     = y_dev.cpu()
            tr_correct += int((preds_cpu == y_cpu).sum().item())
            tr_total   += y_dev.size(0)

        # ── VAL ───────────────────────────────────────────────────────────────
        model.eval()
        vl_loss_sum = 0.0
        vl_correct  = 0
        vl_total    = 0
        val_pred_dist  = [0] * 6
        val_label_dist = [0] * 6
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                out  = model(x)
                loss = criterion(out, y_dev)
                preds_cpu = out.argmax(1).cpu()
                y_cpu     = y_dev.cpu()
                vl_loss_sum += loss.item() * y_dev.size(0)
                vl_correct  += int((preds_cpu == y_cpu).sum().item())
                vl_total    += y_dev.size(0)
                for p in preds_cpu.tolist():
                    val_pred_dist[p] += 1
                for label_val in y_cpu.tolist():
                    val_label_dist[label_val] += 1

        tr_l   = tr_loss_sum / max(tr_total, 1)
        tr_acc = tr_correct  / max(tr_total, 1)
        vl_l   = vl_loss_sum / max(vl_total, 1)
        vl_acc = vl_correct  / max(vl_total, 1)

        scheduler.step(vl_l)
        history.append({'epoch': epoch,
                        'tr_acc': tr_acc, 'vl_acc': vl_acc,
                        'tr_loss': tr_l,  'vl_loss': vl_l})

        dt = time.time() - t_ep
        pred_str  = " ".join(f"{NAV_CLASSES[i][:3]}={val_pred_dist[i]}" for i in range(6))
        label_str = " ".join(f"{NAV_CLASSES[i][:3]}={val_label_dist[i]}" for i in range(6))
        print(f"  Fold {fold_idx+1} | Época {epoch:3d}/{epochs} | "
              f"train={tr_acc:.3f}  val={vl_acc:.3f} | "
              f"Δ={tr_acc - vl_acc:+.3f} "
              f"({'sobreajuste' if tr_acc - vl_acc > 0.10 else 'OK'}) "
              f"| {dt:.1f}s")
        if epoch == 1 or vl_acc >= best_val_acc:
            print(f"           preds:  {pred_str}")
            print(f"           labels: {label_str}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_acc, best_state, history


# ── Evaluación en test fijo ────────────────────────────────────────────────────

def evaluate_test(test_samples: list, state_dict: dict,
                  cache: dict, folder_lists: dict, pos_map: dict, device) -> float:
    test_ds = FoldDataset(test_samples, cache, folder_lists, pos_map, do_augment=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=(device.type != 'cpu'))

    model = build_nav_model(device)
    model.load_state_dict(state_dict)
    model.eval()

    n_classes = len(NAV_CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct = total = 0
    first_batch_logits = None

    with torch.no_grad():
        for x, y in test_dl:
            x     = x.to(device)
            y_dev = y.to(device)
            logits = model(x)
            preds_cpu = logits.argmax(1).cpu()
            y_cpu     = y_dev.cpu()
            if first_batch_logits is None:
                first_batch_logits = logits[:4].cpu().numpy()
                first_batch_labels = y_cpu[:4].numpy()
            for pred, true in zip(preds_cpu.tolist(), y_cpu.tolist()):
                confusion[true, pred] += 1
            correct += int((preds_cpu == y_cpu).sum().item())
            total   += len(y_cpu)

    acc = correct / total if total else 0.0
    print(f"\n  Test fijo ({total} imgs):  accuracy = {acc:.4f}\n")

    # Matriz de confusión
    print(f"  Matriz de confusión (filas=verdadero, columnas=predicho):")
    header = "          " + "".join(f"{cls[:6]:>7}" for cls in NAV_CLASSES)
    print(header)
    for i, cls in enumerate(NAV_CLASSES):
        row = f"  {cls[:8]:<8}" + "".join(f"{confusion[i, j]:>7}" for j in range(n_classes))
        print(row)

    # Accuracy por clase
    print(f"\n  Recall por clase:")
    for i, cls in enumerate(NAV_CLASSES):
        n  = confusion[i].sum()
        ok = confusion[i, i]
        bar = f"{100*ok/n:.1f}%" if n else "—"
        print(f"    {cls:<20}  {ok:>3}/{n:<3}  {bar}")

    # Diagnóstico: logits crudos de las primeras 4 muestras
    print(f"\n  Diagnóstico — logits crudos (primeras 4 muestras test):")
    for i, (logits, true) in enumerate(zip(first_batch_logits, first_batch_labels)):
        true_cls = NAV_IDX_CLASS[int(true)]
        logits_str = "  ".join(f"{v:+6.2f}" for v in logits)
        pred = int(logits.argmax())
        nan_warn = " ← NaN/Inf" if not np.isfinite(logits).all() else ""
        print(f"    [{i}] verdad={true_cls:<12} pred={NAV_IDX_CLASS[pred]:<12} "
              f"logits=[{logits_str}]{nan_warn}")

    return acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds',     type=int,   default=5,              help='Número de folds')
    parser.add_argument('--epochs',    type=int,   default=NUM_EPOCHS_NAV, help='Épocas por fold')
    parser.add_argument('--test-frac', type=float, default=0.20,           help='Fracción para test fijo (default: 0.20 = 20%%)')
    args = parser.parse_args()

    device = torch.device('mps'  if torch.backends.mps.is_available()  else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[KFold] Dispositivo: {device}")
    print(f"[KFold] Folds={args.folds}  Épocas/fold={args.epochs}  Test={int(args.test_frac*100)}%")

    samples = load_samples()
    total   = len(samples)
    test_target = int(total * args.test_frac)
    print(f"\n[KFold] Dataset completo: {total} imágenes")
    print(f"[KFold] Split: {total - test_target} train+CV ({100-int(args.test_frac*100)}%)  |  {test_target} test ({int(args.test_frac*100)}%)")

    print("\n  Clase                Imgs   Videos físicos")
    print("  " + "─" * 50)
    min_videos = float('inf')
    for cls in NAV_CLASSES:
        lbl    = NAV_CLASS_IDX[cls]
        imgs   = [s for s in samples if s[1] == lbl]
        videos = sorted(set(s[2] for s in imgs))
        warn   = " ← muy pocos" if len(videos) < 3 else ""
        min_videos = min(min_videos, len(videos))
        print(f"  {cls:<20} {len(imgs):>4}    {len(videos):>2} videos{warn}")

    if args.folds > min_videos:
        print(f"\n  [WARN] --folds={args.folds} > videos mínimos ({min_videos}).")
        print(f"         Algunos folds no tendrán datos de algunas clases en val.")
        print(f"         Recomendado: --folds {min_videos} para K-Fold honesto.")

    test_samples, cv_samples = split_held_out_test(samples, target=test_target)
    print(f"\n[KFold] Test fijo: {len(test_samples)} imgs ({len(test_samples)/total:.1%})  |  CV: {len(cv_samples)} imgs ({len(cv_samples)/total:.1%})")

    folds = make_folds(cv_samples, n_folds=args.folds)

    print("\n  Distribución de folds:")
    for i, (tr, vl) in enumerate(folds):
        print(f"    Fold {i+1}: train={len(tr)}  val={len(vl)}")

    # Cache compartido: se construye UNA vez para todo el dataset (CV + test)
    cache, folder_lists, pos_map = build_image_cache(samples)

    fold_accs  = []
    best_acc   = 0.0
    best_state = None
    all_history = []

    for fold_idx, (train_s, val_s) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx+1}/{args.folds}  — train:{len(train_s)}  val:{len(val_s)}")
        print(f"{'='*60}")

        if len(val_s) == 0:
            print("  [WARN] Val vacío — fold sin datos suficientes, se omite.")
            continue

        val_acc, state, history = train_fold(
            train_s, val_s, cache, folder_lists, pos_map,
            args.epochs, fold_idx, device)
        fold_accs.append(val_acc)
        all_history.append(history)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = state

        print(f"\n  → Mejor val_acc fold {fold_idx+1}: {val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"  RESUMEN K-FOLD ({args.folds} folds)")
    print(f"{'='*60}")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: val_acc = {acc:.4f}")
    print(f"  Media : {np.mean(fold_accs):.4f}")
    print(f"  Desvío: {np.std(fold_accs):.4f}")

    # Señal de sobreentrenamiento: si la std es alta, el modelo no generaliza bien
    if np.std(fold_accs) > 0.05:
        print("\n  [!!] Desvío alto entre folds — el modelo depende mucho de qué datos")
        print("       entran en train. Necesitas más grabaciones diversas.")
    else:
        print("\n  [OK] Desvío bajo — el modelo generaliza de forma consistente.")

    # Guardar mejor modelo y evaluar en test fijo
    if best_state is not None:
        torch.save({'model_state': best_state, 'best_val_acc': best_acc,
                    'kfold_mean': float(np.mean(fold_accs)),
                    'kfold_std':  float(np.std(fold_accs))},
                   MODEL_NAV_PATH)
        print(f"\n[KFold] Modelo guardado: {MODEL_NAV_PATH}")

        test_acc = evaluate_test(test_samples, best_state,
                                 cache, folder_lists, pos_map, device)

        gap = np.mean(fold_accs) - test_acc
        print(f"\n  Val media: {np.mean(fold_accs):.4f}  |  Test: {test_acc:.4f}  |  Gap: {gap:+.4f}")
        if gap > 0.07:
            print("  [!!] Gap alto — sobreentrenamiento confirmado.")
            print("       Solución: más grabaciones diversas + más augmentación.")
        else:
            print("  [OK] Gap aceptable — sin señal clara de sobreentrenamiento.")

        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(os.path.join(METRICS_DIR, 'kfold_results.json'), 'w') as f:
            json.dump({'fold_accs': fold_accs,
                       'mean': float(np.mean(fold_accs)),
                       'std':  float(np.std(fold_accs)),
                       'test_acc': test_acc}, f, indent=2)
        print(f"  Resultados guardados en metrics/kfold_results.json")


if __name__ == '__main__':
    main()

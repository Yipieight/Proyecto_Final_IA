"""
Prueba rápida: entrena un mini-modelo con las clases disponibles
y lo prueba en un video mostrando las predicciones en tiempo real.

Uso:
    uv run python quick_test.py --train                        # solo entrenar
    uv run python quick_test.py --test ~/Downloads/video.mp4  # solo probar
    uv run python quick_test.py --train --test ~/Downloads/video.mp4

El modelo se guarda en models/quick_test.pth (no sobreescribe nav_model.pth).
"""

import argparse
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from preprocessing import preprocess_frame
from utils import (DATA_NAV_DIR, FRAME_STACK, IMG_HEIGHT, IMG_WIDTH,
                   NAV_CLASSES, FrameBuffer)

QUICK_MODEL_PATH = os.path.join("models", "quick_test.pth")

# ── Detectar qué clases tienen imágenes ───────────────────────────────────────

def get_available_classes():
    available = []
    for cls in NAV_CLASSES:
        d = os.path.join(DATA_NAV_DIR, cls)
        if os.path.isdir(d):
            n = sum(1 for f in os.listdir(d)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if n >= 50:
                available.append((cls, n))
    return available


# ── Dataset simplificado (1 frame, no stack) para velocidad ──────────────────

class QuickDataset(Dataset):
    """Preprocesa TODAS las imágenes una sola vez al init y las guarda en RAM.
    Así el entrenamiento no vuelve a tocar disco ni corre preprocessing.

    NOTA: el orden es DETERMINÍSTICO (sorted). El shuffle lo hace DataLoader
    o AugSubset, nunca aquí — así los índices del random_split son siempre
    coherentes con este dataset."""

    def __init__(self, classes_with_idx):
        self.frames = []   # np.float32 (H, W)
        self.labels = []

        paths = []
        for cls, idx in classes_with_idx:
            d = os.path.join(DATA_NAV_DIR, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append((os.path.join(d, f), idx))
        # NO random.shuffle aquí — orden estable para que random_split sea válido

        total = len(paths)
        print(f"  Preprocesando {total} imágenes... ", end='', flush=True)
        t0 = time.time()
        for path, idx in paths:
            bgr = cv2.imread(path)
            if bgr is None:
                frame = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
            else:
                frame = preprocess_frame(bgr)
            self.frames.append(frame)
            self.labels.append(idx)
        print(f"listo en {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        frame = self.frames[i].copy()
        stack = np.stack([frame, frame, frame], axis=0).astype(np.float32)
        return torch.tensor(stack), torch.tensor(self.labels[i], dtype=torch.long)


class AugSubset(Dataset):
    """Subconjunto de QuickDataset con augmentación en línea.
    Copia los frames referenciados por 'indices' — mismo orden que el dataset
    base, por eso no hay mismatch de etiquetas."""

    def __init__(self, base_ds: QuickDataset, indices):
        self.frames = [base_ds.frames[i] for i in indices]
        self.labels = [base_ds.labels[i] for i in indices]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        frame = self.frames[i].copy()

        if random.random() < 0.5:
            frame = np.clip(frame * random.uniform(0.7, 1.3), 0, 1)
        if random.random() < 0.4:
            frame = np.clip(frame + np.random.randn(*frame.shape) * 0.03, 0, 1)

        stack = np.stack([frame, frame, frame], axis=0).astype(np.float32)
        return torch.tensor(stack), torch.tensor(self.labels[i], dtype=torch.long)


# ── Mini CNN (mismo que model_nav pero instanciado con N clases) ──────────────

class QuickCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(
            block(FRAME_STACK, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Entrenamiento rápido ──────────────────────────────────────────────────────

def train_quick():
    available = get_available_classes()
    if len(available) < 2:
        print("[ERROR] Necesitas al menos 2 clases con ≥50 imágenes cada una.")
        return

    print(f"\n{'='*55}")
    print("  ENTRENAMIENTO RÁPIDO")
    print(f"{'='*55}")
    print("  Clases encontradas:")
    classes_with_idx = []
    for i, (cls, n) in enumerate(available):
        print(f"    [{i}] {cls:<22} {n} imágenes")
        classes_with_idx.append((cls, i))

    num_classes = len(classes_with_idx)
    CLASS_NAMES = [c for c, _ in classes_with_idx]

    device = torch.device('mps') if hasattr(torch.backends, 'mps') \
             and torch.backends.mps.is_available() else torch.device('cpu')
    print(f"\n  Dispositivo: {device}")

    # UN solo dataset cacheado — orden determinístico, sin shuffle interno
    full_ds = QuickDataset(classes_with_idx)
    n       = len(full_ds)
    n_val   = max(50, int(n * 0.15))
    n_tr    = n - n_val

    tr_split, vl_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42))

    # AugSubset copia exactamente los mismos índices — sin riesgo de mismatch
    tr_ds = AugSubset(full_ds, tr_split.indices)

    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True,  num_workers=0)
    vl_loader = DataLoader(vl_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"  Train: {n_tr}  |  Val: {n_val}")

    model     = QuickCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4)

    EPOCHS   = 15
    best_acc = 0.0
    print(f"\n  Entrenando hasta {EPOCHS} epochs (para si llega a 99%)...\n")

    for ep in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total   += x.size(0)

        # Val
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in vl_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl_loss    += criterion(logits, y).item() * x.size(0)
                vl_correct += (logits.argmax(1) == y).sum().item()
                vl_total   += x.size(0)

        tr_a = tr_correct / tr_total
        vl_a = vl_correct / vl_total
        scheduler.step(vl_a)

        tag = " ← MEJOR" if vl_a > best_acc else ""
        print(f"  Ep {ep:2d}/{EPOCHS}  "
              f"Loss {tr_loss/tr_total:.3f}/{vl_loss/vl_total:.3f}  "
              f"Acc {tr_a:.3f}/{vl_a:.3f}{tag}")

        if vl_a > best_acc:
            best_acc = vl_a
            torch.save({
                'model_state': model.state_dict(),
                'class_names': CLASS_NAMES,
                'num_classes': num_classes,
                'best_val_acc': best_acc,
            }, QUICK_MODEL_PATH)

        if best_acc >= 0.99:
            print(f"\n  ✅ Early stop — accuracy ≥ 99% alcanzada en epoch {ep}")
            break

    print(f"\n  ✅ Mejor val accuracy: {best_acc:.1%}")
    print(f"  Modelo guardado: {QUICK_MODEL_PATH}")
    return CLASS_NAMES


# ── Test en video ─────────────────────────────────────────────────────────────

COLORS = [
    (0, 220, 0),    # verde
    (0, 180, 255),  # naranja
    (255, 80, 80),  # azul
    (0, 60, 220),   # rojo
    (220, 0, 220),  # magenta
    (0, 220, 220),  # amarillo
]


@torch.no_grad()
def test_camera(camera_source, use_nav: bool = False, scale: float = 0.75):
    """Igual que test_video pero con fuente en vivo (índice de cámara o URL)."""
    try:
        camera_source = int(camera_source)
    except (ValueError, TypeError):
        pass
    test_video(camera_source, use_nav=use_nav, is_camera=True, scale=scale)


@torch.no_grad()
def test_video(video_path, use_nav: bool = False, is_camera: bool = False,
               scale: float = 1.0):
    from utils import NAV_CLASSES, MODEL_NAV_PATH

    device = torch.device('mps') if hasattr(torch.backends, 'mps') \
             and torch.backends.mps.is_available() else torch.device('cpu')

    if use_nav:
        # ── Modelo completo (nav_model.pth) ──────────────────────────────────
        if not os.path.exists(MODEL_NAV_PATH):
            print(f"[ERROR] Modelo no encontrado: {MODEL_NAV_PATH}")
            print("  Ejecuta primero: uv run python train.py")
            return
        from model_nav import build_nav_model
        ckpt        = torch.load(MODEL_NAV_PATH, map_location=device)
        CLASS_NAMES = list(NAV_CLASSES)
        model       = build_nav_model(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"\n[test] Modelo: {MODEL_NAV_PATH}  (NavCNN completo)")
        print(f"[test] Val accuracy: {ckpt.get('best_val_acc', 0):.1%}  "
              f"(epoch {ckpt.get('epoch', '?')})")
    else:
        # ── Mini-modelo (quick_test.pth) ──────────────────────────────────────
        if not os.path.exists(QUICK_MODEL_PATH):
            print(f"[ERROR] Modelo no encontrado: {QUICK_MODEL_PATH}")
            print("  Ejecuta primero con --train")
            return
        ckpt        = torch.load(QUICK_MODEL_PATH, map_location=device)
        CLASS_NAMES = ckpt['class_names']
        num_classes = ckpt['num_classes']
        model       = QuickCNN(num_classes).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        print(f"\n[test] Modelo: {QUICK_MODEL_PATH}  (QuickCNN)")
        print(f"[test] Val accuracy: {ckpt['best_val_acc']:.1%}")

    print(f"[test] Clases: {CLASS_NAMES}")
    if is_camera:
        print(f"[test] Fuente: cámara en vivo → {video_path}")
    else:
        print(f"[test] Video: {video_path}")
    print("  Presiona Q para salir, ESPACIO para pausar\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if is_camera:
        # Limitar a 1280×720 en cámara en vivo → más FPS, ventana manejable
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    buf     = FrameBuffer(FRAME_STACK)
    paused  = False
    fps_ts  = []

    while True:
        if not paused:
            ret, bgr = cap.read()
            if not ret:
                if is_camera:
                    break          # cámara desconectada → salir
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # video → loop
                continue

            # Preprocesar
            proc  = preprocess_frame(bgr)
            buf.push(proc)
            stack = torch.tensor(buf.get_stack()[None], dtype=torch.float32).to(device)

            logits = model(stack)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred   = int(probs.argmax())

            fps_ts.append(time.time())
            if len(fps_ts) > 30:
                fps_ts.pop(0)
            fps = (len(fps_ts) - 1) / (fps_ts[-1] - fps_ts[0]) if len(fps_ts) > 1 else 0

        # ── Dibujar HUD ───────────────────────────────────────────────────────
        h, w = bgr.shape[:2]
        display = bgr.copy()

        # ROI line
        roi_y = int(h * 0.35)
        cv2.line(display, (0, roi_y), (w, roi_y), (0, 255, 255), 2)
        cv2.putText(display, "ROI", (8, roi_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Predicción principal
        pred_name  = CLASS_NAMES[pred]
        pred_color = COLORS[pred % len(COLORS)]
        cv2.rectangle(display, (0, 0), (w, 75), (0, 0, 0), -1)
        cv2.putText(display, pred_name, (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, pred_color, 3)
        cv2.putText(display, f"FPS: {fps:.1f}", (w - 140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        if paused:
            cv2.putText(display, "PAUSA", (w // 2 - 60, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

        # Barras de confianza por clase
        bar_x, bar_y = 15, 90
        bar_h, bar_max_w = 28, 300
        for i, cls in enumerate(CLASS_NAMES):
            p = float(probs[i])
            bw = int(p * bar_max_w)
            c  = COLORS[i % len(COLORS)]
            cv2.rectangle(display, (bar_x, bar_y + i*(bar_h+6)),
                          (bar_x + bar_max_w, bar_y + i*(bar_h+6) + bar_h),
                          (40, 40, 40), -1)
            cv2.rectangle(display, (bar_x, bar_y + i*(bar_h+6)),
                          (bar_x + bw, bar_y + i*(bar_h+6) + bar_h), c, -1)
            cv2.putText(display,
                        f"{cls}: {p:.0%}",
                        (bar_x + bar_max_w + 10, bar_y + i*(bar_h+6) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

        if scale != 1.0:
            dh, dw = display.shape[:2]
            display = cv2.resize(display,
                                 (int(dw * scale), int(dh * scale)),
                                 interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Quick Test — Q=salir  ESPACIO=pausar", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Entrenar el mini-modelo")
    parser.add_argument("--test",  type=str, default=None,
                        help="Ruta al video para probar (.mp4, .mov)")
    parser.add_argument("--nav",    action="store_true",
                        help="Usar nav_model.pth (modelo completo) en lugar del quick")
    parser.add_argument("--camera", type=str, default=None,
                        help="Cámara en vivo: índice (2=iPhone Continuity) o URL")
    parser.add_argument("--scale",  type=float, default=0.75,
                        help="Escala de la ventana para --camera (default: 0.75). "
                             "Usa 0.5 para pantallas pequeñas, 1.0 para pantalla completa")
    args = parser.parse_args()

    if not args.train and args.test is None and args.camera is None:
        print("Uso:")
        print("  uv run python quick_test.py --train")
        print("  uv run python quick_test.py --test ~/Downloads/video.mp4")
        print("  uv run python quick_test.py --test ~/Downloads/video.mp4 --nav")
        print("  uv run python quick_test.py --camera 2 --nav              # iPhone Continuity")
        print("  uv run python quick_test.py --camera 2 --nav --scale 0.5  # ventana más pequeña")
    else:
        if args.train:
            train_quick()
        if args.test:
            test_video(args.test, use_nav=args.nav)
        if args.camera:
            test_camera(args.camera, use_nav=args.nav, scale=args.scale)

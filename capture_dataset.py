"""
Herramienta de captura interactiva del dataset de navegación desde IP Webcam.

Uso:
    uv run python capture_dataset.py --ip 192.168.2.1 --port 8080

Teclas:
    1 = RECTA          2 = CURVA_IZQ      3 = CURVA_DER
    4 = GIRO_90_IZQ    5 = GIRO_90_DER    6 = CRUCE_T

    ESPACIO = activar / pausar captura automática (~5 FPS)
    d       = eliminar el último frame guardado
    q       = salir y mostrar resumen

NOTA sobre CURVA_IZQ / CURVA_DER:
    Se usa un solo tile físico (CURVA_RADIO_MEDIO).
    Para CURVA_IZQ: colocar el tile con la curva girando a la izquierda.
    Para CURVA_DER: rotar el tile 180° o abordar desde el extremo opuesto.
"""

import argparse
import os
import time
import cv2
from datetime import datetime

from utils import NAV_CLASSES, DATA_NAV_DIR

CAPTURE_INTERVAL_MS = 200   # 1 frame cada 200 ms → ~5 FPS de captura

KEY_MAP = {
    ord('1'): 'RECTA',
    ord('2'): 'CURVA_IZQ',
    ord('3'): 'CURVA_DER',
    ord('4'): 'GIRO_90_IZQ',
    ord('5'): 'GIRO_90_DER',
    ord('6'): 'CRUCE_T',
}


def _count(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def _draw_hud(frame, current_class, capturing, counts):
    display = frame.copy()
    h = display.shape[0]

    color  = (0, 200, 0) if capturing else (0, 60, 200)
    status = "► CAPTURANDO" if capturing else "■ PAUSADO"
    cv2.putText(display, status, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(display, f"Clase: {current_class or '---'}", (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    y = 100
    for cls in NAV_CLASSES:
        n    = counts.get(cls, 0)
        c    = (0, 220, 0) if n >= 500 else (0, 165, 255) if n >= 250 else (0, 0, 220)
        cv2.putText(display, f"{cls}: {n}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1)
        y += 24

    total = sum(counts.values())
    cv2.putText(display, f"Total: {total} / 3000", (10, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    hint = "1-6=clase | ESPACIO=capturar | d=borrar ultimo | q=salir"
    cv2.putText(display, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    return display


def run_capture(ip: str, port: int) -> None:
    url = f"http://{ip}:{port}/video"
    print(f"\n[capture] Conectando a {url} ...")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir {url}")
        print("  Verifica: IP Webcam activo, hotspot del Mac activo, IP correcta.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("[capture] Conectado.\n")
    print("Teclas: 1=RECTA  2=CURVA_IZQ  3=CURVA_DER")
    print("        4=GIRO_90_IZQ  5=GIRO_90_DER  6=CRUCE_T")
    print("ESPACIO=toggle captura | d=borrar último | q=salir\n")

    counts       = {cls: _count(os.path.join(DATA_NAV_DIR, cls)) for cls in NAV_CLASSES}
    current_cls  = None
    capturing    = False
    last_ms      = 0
    last_path    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame no recibido, reintentando...")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        now_ms = int(time.time() * 1000)

        if capturing and current_cls and (now_ms - last_ms) >= CAPTURE_INTERVAL_MS:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(DATA_NAV_DIR, current_cls, f"{ts}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            counts[current_cls] += 1
            last_ms   = now_ms
            last_path = path

        overlay = _draw_hud(frame, current_cls, capturing, counts)
        cv2.imshow("Dataset Capture — q para salir", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if current_cls:
                capturing = not capturing
                print(f"  {'► CAPTURANDO' if capturing else '■ PAUSADO'} — {current_cls}")
            else:
                print("  [!] Selecciona una clase primero (teclas 1-6)")
        elif key == ord('d') and last_path and os.path.exists(last_path):
            os.remove(last_path)
            counts[current_cls] = max(0, counts[current_cls] - 1)
            print(f"  Eliminado: {os.path.basename(last_path)}")
            last_path = None
        elif key in KEY_MAP:
            new_cls = KEY_MAP[key]
            if new_cls != current_cls:
                current_cls = new_cls
                capturing   = False
                print(f"  Clase: {current_cls} ({counts[current_cls]} imgs actuales)")

    cap.release()
    cv2.destroyAllWindows()

    print("\n[capture] Sesión terminada. Resumen:")
    total = 0
    for cls in NAV_CLASSES:
        n = counts[cls]
        total += n
        warn = " ← BAJO" if n < 400 else ""
        print(f"  {cls:<22}: {n:>5} imágenes{warn}")
    print(f"  {'TOTAL':<22}: {total:>5} / 3,000 mínimo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Captura dataset de navegación")
    parser.add_argument("--ip",   default="192.168.2.1",
                        help="IP del celular con IP Webcam (conectado al hotspot del Mac)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    run_capture(args.ip, args.port)

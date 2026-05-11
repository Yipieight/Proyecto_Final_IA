"""
Extrae frames de un video grabado en el iPhone y los guarda
en la carpeta de clase correspondiente del dataset.

Uso:
    uv run python extract_frames.py video.mov RECTA
    uv run python extract_frames.py video.mp4 CURVA_IZQ --fps 5
    uv run python extract_frames.py video.mov GIRO_90_IZQ --fps 4 --preview

Argumentos:
    video   : ruta al archivo de video (.mov, .mp4)
    clase   : una de las 6 clases de navegación
    --fps   : cuántos frames por segundo extraer (default: 5)
              Con video a 30 FPS y --fps 5 se guarda 1 de cada 6 frames.
    --preview: muestra cada frame extraído en pantalla (más lento)
    --skip  : ignorar los primeros N segundos del video (default: 0)
              Útil para descartar el inicio borroso o antes de posicionarse.
    --limit : extraer máximo N frames (default: sin límite)
"""

import argparse
import os
import cv2
import numpy as np
from datetime import datetime

from utils import NAV_CLASSES, DATA_NAV_DIR


def is_blurry(frame: np.ndarray, threshold: float = 50.0) -> bool:
    """Detecta frames borrosos usando varianza del Laplaciano.
    Un frame es borroso si la varianza del Laplaciano es < threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def extract(video_path: str, clase: str,
            target_fps: float = 5.0,
            skip_seconds: float = 0.0,
            limit: int = None,
            preview: bool = False,
            blur_threshold: float = 40.0) -> None:

    if clase not in NAV_CLASSES:
        print(f"[ERROR] Clase '{clase}' no válida.")
        print(f"  Opciones: {', '.join(NAV_CLASSES)}")
        return

    if not os.path.exists(video_path):
        print(f"[ERROR] Archivo no encontrado: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {video_path}")
        return

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / video_fps
    interval     = max(1, round(video_fps / target_fps))

    out_dir = os.path.join(DATA_NAV_DIR, clase)
    os.makedirs(out_dir, exist_ok=True)
    existing = sum(1 for f in os.listdir(out_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    print(f"\n{'='*55}")
    print(f"  EXTRACTOR DE FRAMES — {clase}")
    print(f"{'='*55}")
    print(f"  Video       : {os.path.basename(video_path)}")
    print(f"  Duración    : {duration_s:.1f} s  ({total_frames} frames a {video_fps:.0f} FPS)")
    print(f"  Extracción  : cada {interval} frames → ~{video_fps/interval:.1f} FPS efectivos")
    print(f"  Saltar      : primeros {skip_seconds:.1f} s")
    print(f"  Destino     : {out_dir}")
    print(f"  Ya guardados: {existing} imágenes")
    print(f"{'='*55}\n")

    # Avanzar al segundo de inicio
    if skip_seconds > 0:
        skip_frame = int(skip_seconds * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        print(f"  Saltando {skip_seconds:.1f} s ({skip_frame} frames)...\n")

    saved      = 0
    skipped_blur = 0
    frame_n    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if limit and saved >= limit:
            break

        if frame_n % interval == 0:
            # Descartar frames muy borrosos (sacudidas, transiciones)
            if is_blurry(frame, blur_threshold):
                skipped_blur += 1
            else:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(out_dir, f"{clase}_{ts}_{frame_n:06d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved += 1

                if preview:
                    cv2.imshow(f"Extrayendo — {clase} (q=salir)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n  Extracción interrumpida por el usuario.")
                        break

            if (saved + skipped_blur) % 50 == 0 and (saved + skipped_blur) > 0:
                pct = 100.0 * frame_n / max(1, total_frames - int(skip_seconds * video_fps))
                print(f"  {saved:>4} guardados | {skipped_blur:>3} borrosos descartados "
                      f"| {pct:.0f}% del video procesado")

        frame_n += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    total_now = existing + saved
    print(f"\n{'─'*55}")
    print(f"  Frames guardados en esta sesión : {saved}")
    print(f"  Frames borrosos descartados     : {skipped_blur}")
    print(f"  Total en '{clase}'              : {total_now}")

    if total_now < 500:
        print(f"\n  [!] Solo {total_now} imgs — graba más video de {clase}")
        print(f"      Objetivo mínimo: 500 imágenes por clase")
    elif total_now < 500:
        print(f"\n  [~] {total_now} imgs — aceptable pero apunta a ≥ 500")
    else:
        print(f"\n  [OK] ≥ 500 imgs para {clase}")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae frames de video iPhone al dataset de navegación")
    parser.add_argument("video",  help="Ruta al video (.mov, .mp4)")
    parser.add_argument("clase",  help=f"Clase: {NAV_CLASSES}")
    parser.add_argument("--fps",  type=float, default=5.0,
                        help="Frames por segundo a extraer (default: 5)")
    parser.add_argument("--skip", type=float, default=0.0,
                        help="Ignorar los primeros N segundos (default: 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Límite máximo de frames a guardar")
    parser.add_argument("--preview", action="store_true",
                        help="Mostrar cada frame durante la extracción")
    parser.add_argument("--blur", type=float, default=40.0,
                        help="Umbral de borrosidad — bajar si descarta demasiados (default: 40)")
    args = parser.parse_args()

    extract(
        video_path     = args.video,
        clase          = args.clase,
        target_fps     = args.fps,
        skip_seconds   = args.skip,
        limit          = args.limit,
        preview        = args.preview,
        blur_threshold = args.blur,
    )

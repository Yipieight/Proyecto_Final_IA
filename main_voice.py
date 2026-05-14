# main_voice.py
"""
Pipeline de control por voz en tiempo real.

Modos:
  VAD  (default) : detección automática por energía + umbral adaptativo
  PTT  (--ptt)   : mantén ESPACIO presionado para hablar, suelta para predecir

Flags útiles:
  --verbose      : muestra barras de probabilidad por clase en cada predicción
  --microphone N : índice del micrófono (usa --list-devices para ver opciones)
  --threshold X  : sensibilidad VAD base (default 0.015)
  --confidence X : confianza mínima para enviar (default 0.80)
  --dry-run      : muestra predicciones sin enviar al ESP32
  --list-devices : listar micrófonos disponibles

Uso:
    uv run python main_voice.py --ptt --microphone 0 --dry-run
    uv run python main_voice.py --microphone 0 --verbose --dry-run
"""

import argparse
import os
import queue
import socket
import threading
import time
import numpy as np
import sounddevice as sd
import torch

from utils import ESP32_IP, ESP32_PORT, CMD_STOP
from voice_dataset import (
    compute_mel_spectrogram, TARGET_SR,
    VOICE_CLASSES, VOICE_IDX_CLASS,
)
from model_voice import build_voice_model

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")

# ── Parámetros VAD ───────────────────────────────────────────────────────────
CHUNK_DURATION_S   = 0.05    # 50 ms por chunk
SILENCE_DURATION_S = 0.45    # silencio para cerrar utterance (un poco más para "giro X")
MAX_UTTERANCE_S    = 3.0     # máx duración — cubre "giro izquierda" hablado lento

NOISE_ALPHA    = 0.98         # suavizado exponencial del piso de ruido
VAD_MARGIN     = 4.0          # threshold = noise_floor × margen
MIN_CONFIDENCE = 0.80

VOICE_CMD_MAP = {
    "STOP":      0x00,
    "ADELANTE":  0x01,
    "IZQUIERDA": 0x02,
    "DERECHA":   0x03,
    "GIRO_IZQ":  0x04,
    "GIRO_DER":  0x05,
}


# ── UDP sender ────────────────────────────────────────────────────────────────

class UDPSender:
    def __init__(self, ip: str = ESP32_IP, port: int = ESP32_PORT):
        self._addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._ok   = False

    def send(self, cmd: int) -> None:
        try:
            self._sock.sendto(bytes([cmd]), self._addr)
            self._ok = True
        except Exception:
            self._ok = False

    @property
    def wifi_ok(self) -> bool:
        return self._ok

    def close(self) -> None:
        self._sock.close()


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_voice_model():
    if not os.path.exists(MODEL_VOICE_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_VOICE_PATH}\n"
            "  Ejecuta primero: uv run python train_voice.py"
        )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model  = build_voice_model(device)
    ckpt   = torch.load(MODEL_VOICE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[voice] Modelo cargado (val_acc: {ckpt.get('best_val_acc', 0):.1%})")
    return model, device


# ── Inferencia ────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, audio: np.ndarray, device) -> tuple[str, float, np.ndarray]:
    """Devuelve (clase_ganadora, confianza, array_probs_todas_clases)."""
    mel       = compute_mel_spectrogram(audio)
    tensor    = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)
    probs     = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    idx       = int(probs.argmax())
    return VOICE_IDX_CLASS[idx], float(probs[idx]), probs


def _dispatch(cls_name: str, confidence: float, probs: np.ndarray,
              sender: UDPSender, dry_run: bool, verbose: bool) -> None:

    if verbose:
        print()  # nueva línea tras el "Grabando..."
        for i, cls in enumerate(VOICE_CLASSES):
            p    = float(probs[i])
            bar  = "█" * int(p * 24)
            pad  = "░" * (24 - len(bar))
            mark = " ← enviado" if cls == cls_name and confidence >= MIN_CONFIDENCE else ""
            mark_low = " ← baja confianza" if cls == cls_name and confidence < MIN_CONFIDENCE else ""
            print(f"  {cls:<12} {bar}{pad}  {p:5.1%}{mark}{mark_low}")
        print()
    else:
        pass  # la línea de resultado se imprime abajo

    if confidence < MIN_CONFIDENCE:
        if not verbose:
            print(f" → [descartado — confianza {confidence:.0%}]")
        return

    cmd_byte = VOICE_CMD_MAP.get(cls_name, CMD_STOP)
    if dry_run:
        if not verbose:
            print(f" → DETECTADO: {cls_name}  ({confidence:.0%})")
    else:
        sender.send(cmd_byte)
        wifi = "OK " if sender.wifi_ok else "ERR"
        if not verbose:
            print(f" → {cls_name}  ({confidence:.0%})  UDP:0x{cmd_byte:02X}  WiFi:{wifi}")


# ── Modo VAD (umbral adaptativo) ──────────────────────────────────────────────

def run_vad(vad_threshold: float, device_idx, dry_run: bool, verbose: bool,
            model, torch_device, sender: UDPSender) -> None:
    chunk_samples  = int(TARGET_SR * CHUNK_DURATION_S)
    silence_chunks = int(SILENCE_DURATION_S / CHUNK_DURATION_S)
    max_chunks     = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    print(f"\n[VAD] Umbral base: {vad_threshold}  (se adapta al ruido ambiente)")
    print(f"[VAD] Confianza mínima: {MIN_CONFIDENCE:.0%}")
    print(f"[VAD] Clases: {VOICE_CLASSES}")
    print("[VAD] Escuchando... (Ctrl+C para salir)\n")

    recording    = False
    buffer       = []
    silent_count = 0
    noise_floor  = vad_threshold

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1,
                            dtype="float32", blocksize=chunk_samples,
                            device=device_idx, callback=callback):
            while True:
                chunk     = audio_q.get()
                rms       = float(np.sqrt(np.mean(chunk ** 2)))
                threshold = max(vad_threshold, noise_floor * VAD_MARGIN)

                if not recording:
                    noise_floor = NOISE_ALPHA * noise_floor + (1 - NOISE_ALPHA) * rms
                    if rms >= threshold:
                        recording    = True
                        buffer       = [chunk]
                        silent_count = 0
                        print("  [VAD] ▶ Voz detectada...", end="", flush=True)
                else:
                    buffer.append(chunk)
                    if rms < threshold * 0.6:
                        silent_count += 1
                    else:
                        silent_count = 0

                    if silent_count >= silence_chunks or len(buffer) >= max_chunks:
                        audio_data          = np.concatenate(buffer)
                        cls_name, conf, probs = infer(model, audio_data, torch_device)
                        _dispatch(cls_name, conf, probs, sender, dry_run, verbose)
                        recording    = False
                        buffer       = []
                        silent_count = 0

    except KeyboardInterrupt:
        pass


# ── Modo PTT (push-to-talk con ESPACIO) ───────────────────────────────────────

def run_ptt(device_idx, dry_run: bool, verbose: bool,
            model, torch_device, sender: UDPSender) -> None:
    from pynput import keyboard as kb

    chunk_samples = int(TARGET_SR * CHUNK_DURATION_S)
    max_chunks    = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)

    # El stream SIEMPRE captura — evita perder audio por race condition
    audio_q: queue.Queue = queue.Queue()

    press_event   = threading.Event()
    release_event = threading.Event()

    def on_press(key):
        if key == kb.Key.space:
            press_event.set()
            release_event.clear()

    def on_release(key):
        if key == kb.Key.space:
            release_event.set()
            press_event.clear()

    def audio_callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())   # siempre captura

    print(f"\n[PTT] Mantén ESPACIO para hablar, suelta para predecir.")
    print(f"[PTT] Confianza mínima: {MIN_CONFIDENCE:.0%}")
    print(f"[PTT] Clases: {VOICE_CLASSES}")
    print("[PTT] Listo. (Ctrl+C para salir)\n")

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1,
                            dtype="float32", blocksize=chunk_samples,
                            device=device_idx, callback=audio_callback):
            while True:
                # Esperar a que el usuario presione ESPACIO
                if not press_event.wait(timeout=0.1):
                    continue

                # Vaciar audio pre-pulsación (estático/ambiente)
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break

                print("  [PTT] ● Grabando...", end="", flush=True)
                buffer = []

                # Recoger audio hasta que suelte ESPACIO
                while not release_event.is_set():
                    try:
                        chunk = audio_q.get(timeout=0.05)
                        buffer.append(chunk)
                        if len(buffer) >= max_chunks:
                            break
                    except queue.Empty:
                        continue

                # Pequeña espera para capturar el final del audio
                time.sleep(0.08)
                while not audio_q.empty():
                    try:
                        buffer.append(audio_q.get_nowait())
                    except queue.Empty:
                        break

                if buffer:
                    audio_data            = np.concatenate(buffer)
                    cls_name, conf, probs = infer(model, audio_data, torch_device)
                    _dispatch(cls_name, conf, probs, sender, dry_run, verbose)
                else:
                    print(" → [sin audio]")

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global MIN_CONFIDENCE
    parser = argparse.ArgumentParser(description="Control por voz del robot")
    parser.add_argument("--microphone",    type=int,   default=None,
                        help="Índice del micrófono")
    parser.add_argument("--threshold",     type=float, default=0.015,
                        help="Umbral base RMS para VAD (default: 0.015)")
    parser.add_argument("--confidence",    type=float, default=MIN_CONFIDENCE,
                        help=f"Confianza mínima (default: {MIN_CONFIDENCE})")
    parser.add_argument("--ptt",           action="store_true",
                        help="Push-to-talk: mantén ESPACIO para hablar")
    parser.add_argument("--verbose",       action="store_true",
                        help="Mostrar barras de probabilidad por clase")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Mostrar predicciones sin enviar al ESP32")
    parser.add_argument("--list-devices",  action="store_true",
                        help="Listar micrófonos disponibles y salir")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    MIN_CONFIDENCE = args.confidence

    model, torch_device = load_voice_model()
    sender = UDPSender()

    if args.dry_run:
        print("[voice] MODO PRUEBA — no se envían comandos al ESP32")
    print(f"[voice] ESP32: {ESP32_IP}:{ESP32_PORT}")

    try:
        if args.ptt:
            run_ptt(args.microphone, args.dry_run, args.verbose,
                    model, torch_device, sender)
        else:
            run_vad(args.threshold, args.microphone, args.dry_run, args.verbose,
                    model, torch_device, sender)
    finally:
        if not args.dry_run:
            print("\n[voice] Enviando STOP al ESP32...")
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("[voice] Finalizado.")


if __name__ == "__main__":
    main()

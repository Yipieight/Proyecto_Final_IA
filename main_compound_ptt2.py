# main_compound_ptt2.py
"""
Control de comandos compuestos con dos pulsaciones PTT independientes.

Flujo (respaldo confiable):
  Pulso 1: ESPACIO → di palabra 1 → suelta  (VoiceCNN clasifica)
  Pulso 2: ESPACIO → di palabra 2 → suelta  (VoiceCNN clasifica)
  Lookup table → 2 bytes UDP al ESP32 en secuencia

No usa GRU. Cada palabra es clasificada por VoiceCNN (val_acc 100%).
La combinación se resuelve con una tabla determinista.

Uso:
    uv run python main_compound_ptt2.py --dry-run --verbose
    uv run python main_compound_ptt2.py --microphone 1
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
    compute_mel_spectrogram,
    TARGET_SR,
    VOICE_CLASSES,
    VOICE_IDX_CLASS,
)
from model_voice import build_voice_model

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")

CHUNK_DURATION_S = 0.05
MAX_RECORD_S     = 3.0
MIN_RECORD_S     = 0.30
MIN_CONFIDENCE   = 0.75

# Tabla lookup: (palabra1, palabra2) → (byte_cmd1, byte_cmd2)
LOOKUP: dict[tuple[str, str], tuple[int, int]] = {
    ("ADELANTE",  "IZQUIERDA"): (0x01, 0x02),
    ("ADELANTE",  "DERECHA"):   (0x01, 0x03),
    ("ADELANTE",  "DETENER"):   (0x01, 0x00),
    ("GIRO_IZQ",  "ADELANTE"):  (0x04, 0x01),
    ("GIRO_DER",  "ADELANTE"):  (0x05, 0x01),
    ("IZQUIERDA", "ADELANTE"):  (0x02, 0x01),
    ("DERECHA",   "ADELANTE"):  (0x03, 0x01),
}

DEVICE_STR = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Carga de modelo ───────────────────────────────────────────────────────────

def load_voice_model(device: str):
    if not os.path.exists(MODEL_VOICE_PATH):
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_VOICE_PATH}")
    model = build_voice_model(device)
    ckpt  = torch.load(MODEL_VOICE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[voice] VoiceCNN cargada (val_acc: {ckpt.get('best_val_acc', 0):.1%})")
    return model


# ── Inferencia VoiceCNN ───────────────────────────────────────────────────────

@torch.no_grad()
def infer_word(model, audio: np.ndarray, device: str) -> tuple[str, float, np.ndarray]:
    mel   = compute_mel_spectrogram(audio)
    x     = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx   = int(probs.argmax())
    return VOICE_IDX_CLASS[idx], float(probs[idx]), probs


# ── PTT — captura una palabra ─────────────────────────────────────────────────

def record_word(device_idx, word_num: int) -> np.ndarray:
    from pynput import keyboard as kb

    chunk_samples     = int(TARGET_SR * CHUNK_DURATION_S)
    max_chunks        = int(MAX_RECORD_S / CHUNK_DURATION_S)
    min_record_chunks = int(MIN_RECORD_S / CHUNK_DURATION_S)

    audio_q       = queue.Queue()
    pressed_event = threading.Event()
    release_event = threading.Event()

    def on_press(key):
        if key == kb.Key.space and not pressed_event.is_set():
            pressed_event.set()

    def on_release(key):
        if key == kb.Key.space:
            release_event.set()

    def audio_cb(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    print(f"\n  Palabra {word_num}: mantén ESPACIO → habla → suelta")
    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pressed_event.wait()
    print("  [●] Grabando...", end="", flush=True)

    buffer  = []
    n_pre   = audio_q.qsize()
    skipped = 0

    with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                        blocksize=chunk_samples, device=device_idx,
                        callback=audio_cb):
        while True:
            try:
                chunk = audio_q.get(timeout=0.08)
            except queue.Empty:
                if release_event.is_set() and len(buffer) >= min_record_chunks:
                    break
                continue

            if skipped < n_pre:
                skipped += 1
                continue

            buffer.append(chunk)
            if (release_event.is_set() and len(buffer) >= min_record_chunks) or len(buffer) >= max_chunks:
                break

        time.sleep(0.10)
        while not audio_q.empty():
            try:
                buffer.append(audio_q.get_nowait())
            except queue.Empty:
                break

    listener.stop()
    print(" ✓")
    return np.concatenate(buffer) if buffer else np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)


# ── UDP sender ────────────────────────────────────────────────────────────────

class UDPSender:
    def __init__(self, ip=ESP32_IP, port=ESP32_PORT):
        self._addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: int) -> bool:
        try:
            self._sock.sendto(bytes([cmd]), self._addr)
            return True
        except Exception:
            return False

    def close(self):
        self._sock.close()


# ── Loop principal ────────────────────────────────────────────────────────────

def run(args, model, sender: UDPSender) -> None:
    print(f"\n[ptt2] Comandos compuestos disponibles:")
    for (w1, w2), (b1, b2) in LOOKUP.items():
        print(f"  {w1} + {w2}  →  0x{b1:02X}, 0x{b2:02X}")
    print(f"\n[ptt2] Confianza mínima por palabra: {MIN_CONFIDENCE:.0%}")
    if args.dry_run:
        print("[ptt2] MODO PRUEBA — sin envío al ESP32")
    print("\n" + "═" * 52)
    print("  Pulsa ESPACIO para cada palabra (2 veces por comando)")
    print("  Ctrl+C para salir")
    print("═" * 52)

    while True:
        try:
            # ── Palabra 1 ─────────────────────────────────────────────────────
            audio1 = record_word(args.microphone, 1)
            w1, c1, p1 = infer_word(model, audio1, DEVICE_STR)
            print(f"  → Palabra 1: {w1:<14} ({c1:.0%})", end="")

            if args.verbose:
                print()
                for i, cls in enumerate(VOICE_CLASSES):
                    bar = "█" * int(float(p1[i]) * 20)
                    pad = "░" * (20 - len(bar))
                    print(f"     {cls:<12} {bar}{pad} {float(p1[i]):5.1%}")
            else:
                print()

            if c1 < MIN_CONFIDENCE:
                print(f"  ✗ Confianza baja ({c1:.0%}) — repite")
                continue

            # ── Palabra 2 ─────────────────────────────────────────────────────
            audio2 = record_word(args.microphone, 2)
            w2, c2, p2 = infer_word(model, audio2, DEVICE_STR)
            print(f"  → Palabra 2: {w2:<14} ({c2:.0%})", end="")

            if args.verbose:
                print()
                for i, cls in enumerate(VOICE_CLASSES):
                    bar = "█" * int(float(p2[i]) * 20)
                    pad = "░" * (20 - len(bar))
                    print(f"     {cls:<12} {bar}{pad} {float(p2[i]):5.1%}")
            else:
                print()

            if c2 < MIN_CONFIDENCE:
                print(f"  ✗ Confianza baja ({c2:.0%}) — repite")
                continue

            # ── Lookup ────────────────────────────────────────────────────────
            key = (w1, w2)
            cmd_bytes = LOOKUP.get(key)

            if cmd_bytes is None:
                print(f"  ✗ Combinación no registrada: {w1} + {w2}")
                print("─" * 52)
                continue

            byte1, byte2 = cmd_bytes
            print(f"\n  ► COMANDO: {w1} + {w2}")

            if args.dry_run:
                print(f"  [DRY] → 0x{byte1:02X}  (espera {args.delay}s)  → 0x{byte2:02X}")
            else:
                ok1 = sender.send(byte1)
                print(f"  ✓ CMD1: 0x{byte1:02X}  {'OK' if ok1 else 'ERR'}")
                time.sleep(args.delay)
                ok2 = sender.send(byte2)
                print(f"  ✓ CMD2: 0x{byte2:02X}  {'OK' if ok2 else 'ERR'}")

            print("─" * 52)

        except KeyboardInterrupt:
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comandos compuestos: 2 PTT independientes → lookup → ESP32"
    )
    parser.add_argument("--microphone",   type=int,   default=None)
    parser.add_argument("--confidence",   type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--delay",        type=float, default=0.5,
                        help="Segundos entre CMD1 y CMD2 (default: 0.5)")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Dispositivo PyTorch: {DEVICE_STR}")
    model  = load_voice_model(DEVICE_STR)
    sender = UDPSender()

    try:
        run(args, model, sender)
    finally:
        if not args.dry_run:
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("\n[ptt2] Finalizado.")


if __name__ == "__main__":
    main()

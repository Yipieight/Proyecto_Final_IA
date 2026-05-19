# main_compound.py
"""
Pipeline de control por voz para comandos compuestos de 2 palabras.

Modos:
  VAD (default) : detección automática por energía RMS adaptativa.
                  Di la frase completa de corrido; el sistema detecta inicio
                  y fin automáticamente (silencio de cierre: 700 ms).
  PTT (--ptt)   : mantén ESPACIO, di la frase, suelta.

Flags:
  --ptt          Push-to-Talk con ESPACIO (default: VAD automático)
  --microphone N Índice del micrófono
  --confidence X Confianza mínima del GRU (default: 0.80)
  --threshold X  Sensibilidad VAD base RMS (default: 0.015)
  --delay X      Segundos de duración por movimiento (default: 1.0)
  --dry-run      Muestra predicciones sin enviar al ESP32
  --verbose      Muestra probabilidades por clase
  --list-devices Lista micrófonos y sale

Uso:
    uv run python main_compound.py --microphone 4
    uv run python main_compound.py --microphone 4 --ptt
    uv run python main_compound.py --microphone 4 --dry-run --verbose
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
from model_gru import (
    VoiceGRU,
    compute_mel_sequence,
    COMPOUND_CLASSES,
    COMPOUND_IDX_CLASS,
    COMPOUND_CMD_BYTES,
    NUM_COMPOUND,
    T_MAX,
)

MODEL_GRU_PATH = os.path.join("models", "gru_model.pth")

TARGET_SR        = 16000
CHUNK_DURATION_S = 0.05    # 50 ms por chunk
MAX_RECORD_S     = 3.5     # máximo de la frase compuesta
MIN_RECORD_S     = 0.35

# VAD — más generoso que el de palabras aisladas para no cortar entre palabras
SILENCE_DURATION_S = 0.70  # 700 ms de silencio para cerrar (vs 450 ms en main_voice)
NOISE_ALPHA        = 0.98
VAD_MARGIN         = 4.0

MIN_CONFIDENCE = 0.80

DEVICE_STR = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_gru_model(device: str) -> VoiceGRU:
    if not os.path.exists(MODEL_GRU_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_GRU_PATH}\n"
            "  Ejecuta primero: uv run python train_gru.py"
        )
    model = VoiceGRU()
    state = torch.load(MODEL_GRU_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[gru] VoiceGRU cargada ({NUM_COMPOUND} clases, T_MAX={T_MAX})")
    return model


# ── Inferencia ────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model: VoiceGRU, audio: np.ndarray, device: str) -> tuple[str, float, np.ndarray]:
    mel   = compute_mel_sequence(audio, sr=TARGET_SR, t_max=T_MAX)
    x     = torch.from_numpy(mel[np.newaxis]).to(device)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx   = int(probs.argmax())
    return COMPOUND_IDX_CLASS[idx], float(probs[idx]), probs


# ── UDP sender ────────────────────────────────────────────────────────────────

class UDPSender:
    def __init__(self, ip: str = ESP32_IP, port: int = ESP32_PORT):
        self._addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: int) -> bool:
        try:
            self._sock.sendto(bytes([cmd]), self._addr)
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._sock.close()


# ── Dispatch — inferencia + envío UDP ────────────────────────────────────────

def dispatch(audio: np.ndarray, model: VoiceGRU, sender: UDPSender, args) -> None:
    compound, conf, probs = infer(model, audio, DEVICE_STR)

    if args.verbose:
        print()
        for i, cls in enumerate(COMPOUND_CLASSES):
            p   = float(probs[i])
            bar = "█" * int(p * 30)
            pad = "░" * (30 - len(bar))
            mark = " ◄" if cls == compound else ""
            print(f"  {cls:<22} {bar}{pad} {p:5.1%}{mark}")
        print()

    print(f"  ► GRU: {compound}  ({conf:.0%})")

    if conf < args.confidence:
        print(f"  ✗ Confianza insuficiente ({conf:.0%}) — ignorado")
        print("─" * 52)
        return

    cmd_bytes = COMPOUND_CMD_BYTES.get(compound)
    if cmd_bytes is None:
        print(f"  ✗ Sin mapeo UDP para {compound}")
        print("─" * 52)
        return

    byte1, byte2 = cmd_bytes

    if args.dry_run:
        print(f"  [DRY] → 0x{byte1:02X}  ({args.delay}s)  → 0x{byte2:02X}  ({args.delay}s)  → STOP")
    else:
        ok1 = sender.send(byte1)
        print(f"  ✓ CMD1: 0x{byte1:02X}  {'OK' if ok1 else 'ERR'}")
        time.sleep(args.delay)
        ok2 = sender.send(byte2)
        print(f"  ✓ CMD2: 0x{byte2:02X}  {'OK' if ok2 else 'ERR'}")
        time.sleep(args.delay)
        sender.send(CMD_STOP)
        print(f"  ✓ STOP: 0x00")

    print("─" * 52)


# ── Modo VAD ──────────────────────────────────────────────────────────────────

def run_vad(args, model: VoiceGRU, sender: UDPSender) -> None:
    chunk_samples  = int(TARGET_SR * CHUNK_DURATION_S)
    silence_chunks = int(SILENCE_DURATION_S / CHUNK_DURATION_S)  # 14 chunks = 700ms
    max_chunks     = int(MAX_RECORD_S / CHUNK_DURATION_S)

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    print(f"\n[VAD] Umbral base: {args.threshold}  (adaptativo)")
    print(f"[VAD] Silencio de cierre: {SILENCE_DURATION_S:.0%}s — no cortará entre palabras")
    print(f"[VAD] Ventana máxima: {MAX_RECORD_S}s")
    print(f"[VAD] Clases: {COMPOUND_CLASSES}")
    print("[VAD] Escuchando... (Ctrl+C para salir)\n")

    recording    = False
    buffer       = []
    silent_count = 0
    noise_floor  = args.threshold

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                            blocksize=chunk_samples, device=args.microphone,
                            callback=callback):
            while True:
                chunk = audio_q.get()
                rms   = float(np.sqrt(np.mean(chunk ** 2)))
                threshold = max(args.threshold, noise_floor * VAD_MARGIN)

                if not recording:
                    noise_floor = NOISE_ALPHA * noise_floor + (1 - NOISE_ALPHA) * rms
                    if rms >= threshold:
                        recording    = True
                        buffer       = [chunk]
                        silent_count = 0
                        print("  [VAD] ▶ Frase detectada...", end="", flush=True)
                else:
                    buffer.append(chunk)
                    if rms < threshold * 0.6:
                        silent_count += 1
                    else:
                        silent_count = 0

                    if silent_count >= silence_chunks or len(buffer) >= max_chunks:
                        dur = len(buffer) * CHUNK_DURATION_S
                        print(f" {dur:.2f}s")
                        dispatch(np.concatenate(buffer), model, sender, args)
                        recording    = False
                        buffer       = []
                        silent_count = 0

    except KeyboardInterrupt:
        pass


# ── Modo PTT ──────────────────────────────────────────────────────────────────

def run_ptt(args, model: VoiceGRU, sender: UDPSender) -> None:
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
            release_event.clear()

    def on_release(key):
        if key == kb.Key.space:
            release_event.set()
            pressed_event.clear()

    def audio_cb(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    print(f"\n[PTT] Mantén ESPACIO → di la frase completa → suelta")
    print(f"[PTT] Ventana máxima: {MAX_RECORD_S}s")
    print(f"[PTT] Clases: {COMPOUND_CLASSES}")
    print("[PTT] Listo. (Ctrl+C para salir)\n")

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32",
                            blocksize=chunk_samples, device=args.microphone,
                            callback=audio_cb):
            while True:
                if not pressed_event.wait(timeout=0.1):
                    continue

                n_pre   = audio_q.qsize()
                skipped = 0
                buffer  = []
                print("  [PTT] ● Grabando...", end="", flush=True)

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
                    released = release_event.is_set()
                    if (released and len(buffer) >= min_record_chunks) or len(buffer) >= max_chunks:
                        break

                time.sleep(0.10)
                while not audio_q.empty():
                    try:
                        buffer.append(audio_q.get_nowait())
                    except queue.Empty:
                        break

                if buffer:
                    dur = len(buffer) * CHUNK_DURATION_S
                    print(f" {dur:.2f}s")
                    dispatch(np.concatenate(buffer), model, sender, args)

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comandos compuestos por voz — VAD/PTT → GRU → ESP32"
    )
    parser.add_argument("--ptt",          action="store_true",
                        help="Push-to-Talk: mantén ESPACIO (default: VAD automático)")
    parser.add_argument("--microphone",   type=int,   default=None)
    parser.add_argument("--confidence",   type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--threshold",    type=float, default=0.015,
                        help="Umbral base RMS para VAD (default: 0.015)")
    parser.add_argument("--delay",        type=float, default=1.0,
                        help="Segundos de duración por movimiento (default: 1.0)")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Dispositivo PyTorch: {DEVICE_STR}")
    model  = load_gru_model(DEVICE_STR)
    sender = UDPSender()

    if args.dry_run:
        print("[compound] MODO PRUEBA — sin envío al ESP32")
    print(f"[compound] ESP32: {ESP32_IP}:{ESP32_PORT}")

    try:
        if args.ptt:
            run_ptt(args, model, sender)
        else:
            run_vad(args, model, sender)
    finally:
        if not args.dry_run:
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("\n[compound] Finalizado.")


if __name__ == "__main__":
    main()

# main_compound.py
"""
Pipeline de control por voz para comandos compuestos de 2 palabras.

Flujo (un solo PTT):
  Mantén ESPACIO → di la frase completa de corrido (ej. "adelante izquierda") → suelta
  El audio completo se procesa como una secuencia temporal de mel-spectrogram.
  VoiceGRU predice el comando compuesto y envía 2 bytes UDP al ESP32.

Flags:
  --microphone N   Índice del micrófono
  --confidence X   Confianza mínima del GRU (default: 0.80)
  --dry-run        Muestra predicciones sin enviar al ESP32
  --verbose        Muestra probabilidades por clase
  --delay X        Segundos entre el byte 1 y el byte 2 (default: 0.5)
  --list-devices   Lista micrófonos y sale

Uso:
    uv run python main_compound.py --dry-run --verbose
    uv run python main_compound.py --microphone 1
    uv run python main_compound.py --microphone 1 --delay 0.8
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
CHUNK_DURATION_S = 0.05
MAX_RECORD_S     = 3.5
MIN_RECORD_S     = 0.35
MIN_CONFIDENCE   = 0.80

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
    print(f"[gru] VoiceGRU cargada ({NUM_COMPOUND} clases compuestas, T_MAX={T_MAX})")
    return model


# ── Inferencia ────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model: VoiceGRU, audio: np.ndarray, device: str) -> tuple[str, float, np.ndarray]:
    """
    Convierte el audio completo del enunciado compuesto a mel-sequence
    y lo pasa por VoiceGRU.

    Devuelve: (clase_compuesta, confianza, array_probs)
    """
    mel    = compute_mel_sequence(audio, sr=TARGET_SR, t_max=T_MAX)   # (T_MAX, 64)
    x      = torch.from_numpy(mel[np.newaxis]).to(device)             # (1, T_MAX, 64)
    probs  = torch.softmax(model(x), dim=1)[0].cpu().numpy()          # (N_COMPOUND,)
    idx    = int(probs.argmax())
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


# ── PTT — captura un solo bloque de audio ────────────────────────────────────

def record_ptt(device_idx) -> np.ndarray:
    """
    Mantén ESPACIO → habla la frase completa de corrido → suelta.
    Captura todo el audio desde que se presiona hasta que se suelta
    (máximo MAX_RECORD_S segundos).
    """
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

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n  Mantén ESPACIO y di la frase completa → suelta para inferir")
    pressed_event.wait()
    print("  [●] Grabando...", end="", flush=True)

    buffer  = []
    n_pre   = audio_q.qsize()   # chunks previos al press — descartar
    skipped = 0

    with sd.InputStream(
        samplerate=TARGET_SR, channels=1, dtype="float32",
        blocksize=chunk_samples, device=device_idx, callback=audio_cb,
    ):
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

        # Capturar consonante final tras soltar
        time.sleep(0.10)
        while not audio_q.empty():
            try:
                buffer.append(audio_q.get_nowait())
            except queue.Empty:
                break

    listener.stop()
    dur = len(buffer) * CHUNK_DURATION_S
    print(f" {dur:.2f}s ✓")

    return (
        np.concatenate(buffer)
        if buffer
        else np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)
    )


# ── Loop principal ────────────────────────────────────────────────────────────

def run(args, model: VoiceGRU, sender: UDPSender) -> None:
    min_conf = args.confidence

    print(f"\n[compound] Clases compuestas ({NUM_COMPOUND}):")
    for i, cls in enumerate(COMPOUND_CLASSES):
        print(f"  {i+1}. {cls}")
    print(f"\n[compound] Confianza mínima : {min_conf:.0%}")
    print(f"[compound] Delay entre cmds : {args.delay}s")
    print(f"[compound] Ventana máxima   : {MAX_RECORD_S}s")
    if args.dry_run:
        print("[compound] MODO PRUEBA — sin envío al ESP32")
    print("\n" + "═" * 52)
    print("  Mantén ESPACIO → di la frase completa → suelta")
    print("  Ctrl+C para salir")
    print("═" * 52)

    while True:
        try:
            # ── Captura ───────────────────────────────────────────────────────
            audio = record_ptt(args.microphone)

            # ── Inferencia GRU ────────────────────────────────────────────────
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

            if conf < min_conf:
                print(f"  ✗ Confianza insuficiente — ignorado")
                print("─" * 52)
                continue

            cmd_bytes = COMPOUND_CMD_BYTES.get(compound)
            if cmd_bytes is None:
                print(f"  ✗ Sin mapeo UDP para {compound}")
                print("─" * 52)
                continue

            byte1, byte2 = cmd_bytes

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comandos compuestos por voz — 1 PTT → GRU → ESP32"
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
    model  = load_gru_model(DEVICE_STR)
    sender = UDPSender()

    try:
        run(args, model, sender)
    finally:
        if not args.dry_run:
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("\n[compound] Finalizado.")


if __name__ == "__main__":
    main()

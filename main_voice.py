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
    "DETENER":   0x00,
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


# ── Trim de silencio ─────────────────────────────────────────────────────────

def _trim_silence(audio: np.ndarray,
                  chunk_s: float = CHUNK_DURATION_S,
                  margin: float = 0.6) -> np.ndarray:
    """Recorta silencio al inicio y al final basándose en energía RMS.
    Se usa en PTT para que el audio llegue al modelo igual que en VAD."""
    chunk = int(TARGET_SR * chunk_s)
    if len(audio) < chunk * 2:
        return audio

    # Calcular RMS por chunk y estimar piso de energía
    rms_vals = [float(np.sqrt(np.mean(audio[i:i+chunk]**2)))
                for i in range(0, len(audio) - chunk, chunk)]
    if not rms_vals:
        return audio
    noise_floor = min(rms_vals)
    threshold   = max(noise_floor * 4.0, 0.008)

    # Primer chunk con voz
    start_chunk = 0
    for i, r in enumerate(rms_vals):
        if r >= threshold:
            start_chunk = i
            break

    # Último chunk con voz
    end_chunk = len(rms_vals)
    for i in range(len(rms_vals) - 1, -1, -1):
        if rms_vals[i] >= threshold * margin:
            end_chunk = i + 2   # +2 para no cortar consonante final
            break

    start = max(0, start_chunk * chunk)
    end   = min(len(audio), end_chunk * chunk)
    trimmed = audio[start:end]
    return trimmed if len(trimmed) >= chunk * 2 else audio


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
            model, torch_device, sender: UDPSender,
            no_vad: bool = False) -> None:
    chunk_samples  = int(TARGET_SR * CHUNK_DURATION_S)
    silence_chunks = int(SILENCE_DURATION_S / CHUNK_DURATION_S)
    max_chunks     = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    if no_vad:
        print(f"\n[CONTINUO] Sin detección de voz — predice cada {MAX_UTTERANCE_S:.1f}s")
        print(f"[CONTINUO] Confianza mínima: {MIN_CONFIDENCE:.0%}")
        print(f"[CONTINUO] Clases: {VOICE_CLASSES}")
        print("[CONTINUO] Escuchando... (Ctrl+C para salir)\n")
    else:
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
                chunk = audio_q.get()

                if no_vad:
                    # Modo continuo: acumula chunks fijos y predice sin umbral
                    buffer.append(chunk)
                    if len(buffer) >= max_chunks:
                        audio_data            = np.concatenate(buffer)
                        cls_name, conf, probs = infer(model, audio_data, torch_device)
                        _dispatch(cls_name, conf, probs, sender, dry_run, verbose)
                        buffer = []
                    continue

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
                        audio_data            = np.concatenate(buffer)
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

    chunk_samples     = int(TARGET_SR * CHUNK_DURATION_S)
    max_chunks        = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)
    min_record_chunks = int(0.35 / CHUNK_DURATION_S)   # mínimo 350 ms grabado

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
        audio_q.put(indata[:, 0].copy())

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
                if not press_event.wait(timeout=0.1):
                    continue

                # Descartar solo los chunks anteriores al press (ambiente pre-pulsación).
                # Los chunks nuevos (inicio de la palabra) NO se descartan.
                n_skip  = audio_q.qsize()
                skipped = 0

                print("  [PTT] ● Grabando...", end="", flush=True)
                buffer = []

                # Recoger hasta soltar ESPACIO — mínimo min_record_chunks
                while True:
                    try:
                        chunk = audio_q.get(timeout=0.08)
                    except queue.Empty:
                        if release_event.is_set() and len(buffer) >= min_record_chunks:
                            break
                        continue

                    if skipped < n_skip:
                        skipped += 1   # descartar pre-press
                        continue

                    buffer.append(chunk)

                    released = release_event.is_set()
                    if (released and len(buffer) >= min_record_chunks) or len(buffer) >= max_chunks:
                        break

                # Cola residual tras soltar (consonante final de la palabra)
                time.sleep(0.12)
                while not audio_q.empty():
                    try:
                        buffer.append(audio_q.get_nowait())
                    except queue.Empty:
                        break

                if buffer:
                    audio_data            = _trim_silence(np.concatenate(buffer))
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
    parser.add_argument("--no-vad",        action="store_true",
                        help="Desactivar detección de voz — predice continuamente cada 3s")
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
                    model, torch_device, sender, no_vad=args.no_vad)
    finally:
        if not args.dry_run:
            print("\n[voice] Enviando STOP al ESP32...")
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("[voice] Finalizado.")


if __name__ == "__main__":
    main()

# main_voice.py
"""
Pipeline de control por voz en tiempo real.

Flujo:
  Micrófono → sounddevice → VAD por energía → mel-spectrogram →
  VoiceCNN → argmax → byte UDP → ESP32

Uso:
    uv run python main_voice.py
    uv run python main_voice.py --threshold 0.03   # ajustar sensibilidad
    uv run python main_voice.py --list-devices      # ver micrófonos disponibles

Requiere:
    - models/voice_model.pth  (genera con train_voice.py)
    - ESP32 encendido y con IP actualizada en utils.py
"""

import argparse
import os
import queue
import socket
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
SILENCE_DURATION_S = 0.35    # silencio para cerrar utterance
MAX_UTTERANCE_S    = 2.0     # máx duración de un comando

# Mapa clase → byte UDP (mismo protocolo que la cámara)
VOICE_CMD_MAP = {
    "ALTO":      0x00,
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
def infer(model, audio: np.ndarray, device) -> str:
    mel    = compute_mel_spectrogram(audio)
    tensor = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)  # (1,1,64,64)
    idx    = model(tensor).argmax(1).item()
    return VOICE_IDX_CLASS[idx]


# ── Loop principal ────────────────────────────────────────────────────────────

def run(vad_threshold: float = 0.02, device_idx=None, dry_run: bool = False) -> None:
    model, torch_device = load_voice_model()
    sender = UDPSender()
    if dry_run:
        print("[voice] MODO PRUEBA — no se envían comandos al ESP32")

    chunk_samples  = int(TARGET_SR * CHUNK_DURATION_S)
    silence_chunks = int(SILENCE_DURATION_S / CHUNK_DURATION_S)
    max_chunks     = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    print(f"\n[voice] ESP32: {ESP32_IP}:{ESP32_PORT}")
    print(f"[voice] VAD threshold: {vad_threshold}")
    print(f"[voice] Clases: {VOICE_CLASSES}")
    print("[voice] Escuchando... (Ctrl+C para salir)\n")

    recording    = False
    buffer       = []
    silent_count = 0

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1,
                            dtype="float32", blocksize=chunk_samples,
                            device=device_idx, callback=callback):
            while True:
                chunk = audio_q.get()
                rms   = np.sqrt(np.mean(chunk ** 2))

                if not recording:
                    if rms >= vad_threshold:
                        recording    = True
                        buffer       = [chunk]
                        silent_count = 0
                        print("  [VAD] Voz detectada...", end="", flush=True)
                else:
                    buffer.append(chunk)
                    if rms < vad_threshold:
                        silent_count += 1
                    else:
                        silent_count = 0

                    if silent_count >= silence_chunks or len(buffer) >= max_chunks:
                        audio_data = np.concatenate(buffer)
                        cls_name   = infer(model, audio_data, torch_device)
                        cmd_byte   = VOICE_CMD_MAP.get(cls_name, CMD_STOP)

                        if dry_run:
                            print(f" → DETECTADO: {cls_name}")
                        else:
                            sender.send(cmd_byte)
                            wifi_str = "OK " if sender.wifi_ok else "ERR"
                            print(f" → {cls_name}  (UDP: 0x{cmd_byte:02X}  WiFi:{wifi_str})")

                        recording    = False
                        buffer       = []
                        silent_count = 0

    except KeyboardInterrupt:
        pass
    finally:
        if not dry_run:
            print("\n[voice] Enviando STOP al ESP32...")
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("\n[voice] Finalizado.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control por voz del robot")
    parser.add_argument("--microphone",   type=int, default=None,
                        help="Índice del micrófono (usa --microphone list para ver opciones)")
    parser.add_argument("--threshold",    type=float, default=0.02,
                        help="Umbral de energía RMS para VAD (default: 0.02)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Modo prueba: muestra lo detectado sin enviar al ESP32")
    parser.add_argument("--list-devices", action="store_true",
                        help="Mostrar micrófonos disponibles y salir")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    else:
        run(vad_threshold=args.threshold, device_idx=args.microphone,
            dry_run=args.dry_run)

# main_compound.py
"""
Pipeline de control por voz para comandos compuestos de 2 palabras.

Flujo (PTT doble):
  Pulso 1: mantén ESPACIO → di palabra 1 → suelta
  Pulso 2: mantén ESPACIO → di palabra 2 → suelta
  VoiceGRU predice el comando compuesto y envía 2 bytes UDP al ESP32.

Flags:
  --microphone N  Índice del micrófono
  --confidence X  Confianza mínima del GRU (default: 0.80)
  --dry-run       Muestra predicciones sin enviar al ESP32
  --verbose       Muestra probabilidades detalladas
  --delay X       Segundos entre el byte 1 y el byte 2 (default: 0.5)
  --list-devices  Lista micrófonos y sale

Uso:
    uv run python main_compound.py --dry-run
    uv run python main_compound.py --microphone 1 --verbose
    uv run python main_compound.py --microphone 1 --delay 0.8
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
from voice_dataset import compute_mel_spectrogram, TARGET_SR, VOICE_IDX_CLASS
from model_voice import VoiceCNN
from model_gru import (
    VoiceGRU,
    COMPOUND_CLASSES,
    COMPOUND_IDX_CLASS,
    NUM_COMPOUND,
)

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")
MODEL_GRU_PATH   = os.path.join("models", "gru_model.pth")

# Mapeo compuesto → par de bytes UDP a enviar en secuencia
COMPOUND_CMD_BYTES = {
    "ADELANTE_IZQUIERDA": (0x01, 0x02),
    "ADELANTE_DERECHA":   (0x01, 0x03),
    "ADELANTE_DETENER":   (0x01, 0x00),
    "GIRO_IZQ_ADELANTE":  (0x04, 0x01),
    "GIRO_DER_ADELANTE":  (0x05, 0x01),
    "IZQUIERDA_ADELANTE": (0x02, 0x01),
    "DERECHA_ADELANTE":   (0x03, 0x01),
}

CHUNK_DURATION_S  = 0.05
MAX_UTTERANCE_S   = 3.0
MIN_CONFIDENCE    = 0.80

DEVICE_STR = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ── Carga de modelos ──────────────────────────────────────────────────────────

def load_voice_model(device: str) -> VoiceCNN:
    if not os.path.exists(MODEL_VOICE_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_VOICE_PATH}\n"
            "  Ejecuta primero: uv run python train_voice.py"
        )
    model = VoiceCNN()
    ckpt  = torch.load(MODEL_VOICE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[voice] VoiceCNN cargada (val_acc: {ckpt.get('best_val_acc', 0):.1%})")
    return model


def load_gru_model(device: str) -> VoiceGRU:
    if not os.path.exists(MODEL_GRU_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_GRU_PATH}\n"
            "  Ejecuta primero:\n"
            "    uv run python extract_embeddings.py\n"
            "    uv run python train_gru.py"
        )
    model = VoiceGRU()
    state = torch.load(MODEL_GRU_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[gru]   VoiceGRU cargada ({NUM_COMPOUND} clases compuestas)")
    return model


# ── Hook de embedding ─────────────────────────────────────────────────────────

def make_embedding_hook(voice_model: VoiceCNN):
    """
    Registra un hook en classifier[5] (ReLU tras FC(256→64)).
    Devuelve (store_dict, handle) — el embedding queda en store_dict["last"].
    """
    store = {}

    def hook(module, input, output):
        store["last"] = output.detach().cpu()

    handle = voice_model.classifier[5].register_forward_hook(hook)
    return store, handle


# ── Inferencia individual (VoiceCNN → embedding) ─────────────────────────────

@torch.no_grad()
def infer_word(
    voice_model: VoiceCNN,
    emb_store: dict,
    audio: np.ndarray,
    device: str,
) -> tuple[str, float, np.ndarray]:
    """
    Pasa el audio por VoiceCNN.
    Devuelve (clase_predicha, confianza, embedding_64d).
    """
    mel    = compute_mel_spectrogram(audio)
    x      = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)
    logits = voice_model(x)                            # forward activa el hook
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx    = int(probs.argmax())
    emb    = emb_store["last"][0].numpy()              # (64,)
    return VOICE_IDX_CLASS[idx], float(probs[idx]), emb


# ── Inferencia compuesta (VoiceGRU) ──────────────────────────────────────────

@torch.no_grad()
def infer_compound(
    gru_model: VoiceGRU,
    emb1: np.ndarray,
    emb2: np.ndarray,
    device: str,
) -> tuple[str, float]:
    """Pasa la secuencia [emb1, emb2] por VoiceGRU. Devuelve (clase, confianza)."""
    pair   = np.stack([emb1, emb2], axis=0)[np.newaxis]          # (1, 2, 64)
    x      = torch.from_numpy(pair.astype(np.float32)).to(device)
    probs  = torch.softmax(gru_model(x), dim=1)[0].cpu().numpy()
    idx    = int(probs.argmax())
    return COMPOUND_IDX_CLASS[idx], float(probs[idx])


# ── UDP sender ────────────────────────────────────────────────────────────────

class UDPSender:
    def __init__(self, ip: str = ESP32_IP, port: int = ESP32_PORT):
        self._addr = (ip, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, cmd: int) -> None:
        self._sock.sendto(bytes([cmd]), self._addr)

    def close(self) -> None:
        self._sock.close()


# ── Grabar una palabra con PTT ────────────────────────────────────────────────

def record_word_ptt(device_idx, word_num: int) -> np.ndarray:
    """
    Graba hasta MAX_UTTERANCE_S segundos mientras el usuario mantiene ESPACIO.
    Devuelve el array de audio (float32, 16kHz).
    """
    from pynput import keyboard as kb
    import threading

    chunk_samples     = int(TARGET_SR * CHUNK_DURATION_S)
    max_chunks        = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)
    min_record_chunks = int(0.35 / CHUNK_DURATION_S)

    audio_q       = queue.Queue()
    release_event = threading.Event()

    print(f"\n  Palabra {word_num}: mantén ESPACIO y habla → suelta para confirmar")

    # Esperar que el usuario presione ESPACIO
    pressed = threading.Event()

    def on_press(key):
        if key == kb.Key.space and not pressed.is_set():
            pressed.set()

    def on_release(key):
        if key == kb.Key.space:
            release_event.set()

    def audio_cb(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pressed.wait()                     # bloquea hasta que pulsa ESPACIO
    print(f"  [●] Grabando...", end="", flush=True)

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
            released = release_event.is_set()
            if (released and len(buffer) >= min_record_chunks) or len(buffer) >= max_chunks:
                break

        # Capturar consonante final tras soltar
        time.sleep(0.12)
        while not audio_q.empty():
            try:
                buffer.append(audio_q.get_nowait())
            except queue.Empty:
                break

    listener.stop()
    print(" ✓")
    return np.concatenate(buffer) if buffer else np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)


# ── Trim de silencio ──────────────────────────────────────────────────────────

def trim_silence(audio: np.ndarray) -> np.ndarray:
    chunk = int(TARGET_SR * CHUNK_DURATION_S)
    if len(audio) < chunk * 2:
        return audio
    rms_vals = [float(np.sqrt(np.mean(audio[i:i+chunk]**2)))
                for i in range(0, len(audio) - chunk, chunk)]
    if not rms_vals:
        return audio
    noise_floor = min(rms_vals)
    threshold   = max(noise_floor * 4.0, 0.008)

    start_chunk, end_chunk = 0, len(rms_vals)
    for i, r in enumerate(rms_vals):
        if r >= threshold:
            start_chunk = i
            break
    for i in range(len(rms_vals) - 1, -1, -1):
        if rms_vals[i] >= threshold * 0.6:
            end_chunk = i + 2
            break

    start   = max(0, start_chunk * chunk)
    end     = min(len(audio), end_chunk * chunk)
    trimmed = audio[start:end]
    return trimmed if len(trimmed) >= chunk * 2 else audio


# ── Loop principal ────────────────────────────────────────────────────────────

def run(args, voice_model, gru_model, emb_store, sender):
    min_conf = args.confidence

    print(f"\n[compound] Clases compuestas ({NUM_COMPOUND}):")
    for i, cls in enumerate(COMPOUND_CLASSES):
        print(f"  {i+1}. {cls}")
    print(f"\n[compound] Confianza mínima: {min_conf:.0%}")
    print(f"[compound] Delay entre comandos: {args.delay}s")
    if args.dry_run:
        print("[compound] MODO PRUEBA — no se envían comandos al ESP32")
    print("\n" + "─" * 52)
    print("Instrucciones:")
    print("  1. Pulsa ESPACIO y di la primera palabra, suelta.")
    print("  2. Pulsa ESPACIO y di la segunda palabra, suelta.")
    print("  → El GRU predice y ejecuta el comando compuesto.")
    print("  Ctrl+C para salir.")
    print("─" * 52)

    while True:
        try:
            # ── Palabra 1 ─────────────────────────────────────────────────────
            audio1 = trim_silence(record_word_ptt(args.microphone, 1))
            word1, conf1, emb1 = infer_word(voice_model, emb_store, audio1, DEVICE_STR)
            print(f"  → Palabra 1: {word1:<14} ({conf1:.0%})")

            # ── Palabra 2 ─────────────────────────────────────────────────────
            audio2 = trim_silence(record_word_ptt(args.microphone, 2))
            word2, conf2, emb2 = infer_word(voice_model, emb_store, audio2, DEVICE_STR)
            print(f"  → Palabra 2: {word2:<14} ({conf2:.0%})")

            # ── GRU ───────────────────────────────────────────────────────────
            compound, conf_gru = infer_compound(gru_model, emb1, emb2, DEVICE_STR)
            print(f"\n  ► COMANDO GRU: {compound}  ({conf_gru:.0%})")

            if args.verbose:
                print()

            if conf_gru < min_conf:
                print(f"  ✗ Confianza insuficiente ({conf_gru:.0%} < {min_conf:.0%}) — ignorado")
                print("─" * 52)
                continue

            cmd_bytes = COMPOUND_CMD_BYTES.get(compound)
            if cmd_bytes is None:
                print(f"  ✗ Sin mapeo UDP para {compound}")
                print("─" * 52)
                continue

            byte1, byte2 = cmd_bytes
            if args.dry_run:
                print(f"  [DRY] UDP byte 1: 0x{byte1:02X}  →  {compound.split('_')[0]}")
                print(f"  [DRY] (espera {args.delay}s)")
                print(f"  [DRY] UDP byte 2: 0x{byte2:02X}  →  {compound.split('_')[1] if '_' in compound else ''}")
            else:
                sender.send(byte1)
                print(f"  ✓ UDP: 0x{byte1:02X} enviado")
                time.sleep(args.delay)
                sender.send(byte2)
                print(f"  ✓ UDP: 0x{byte2:02X} enviado")

            print("─" * 52)

        except KeyboardInterrupt:
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Control compuesto por voz — 2 palabras + GRU")
    parser.add_argument("--microphone",  type=int,   default=None)
    parser.add_argument("--confidence",  type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="Segundos entre el cmd 1 y cmd 2 (default: 0.5)")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Dispositivo PyTorch: {DEVICE_STR}")
    voice_model = load_voice_model(DEVICE_STR)
    gru_model   = load_gru_model(DEVICE_STR)
    emb_store, handle = make_embedding_hook(voice_model)
    sender = UDPSender()

    try:
        run(args, voice_model, gru_model, emb_store, sender)
    finally:
        handle.remove()
        if not args.dry_run:
            sender.send(CMD_STOP)
            time.sleep(0.1)
        sender.close()
        print("\n[compound] Finalizado.")


if __name__ == "__main__":
    main()

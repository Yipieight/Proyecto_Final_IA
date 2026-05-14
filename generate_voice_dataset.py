# generate_voice_dataset.py
"""
Genera dataset de audio para control por voz usando Piper TTS (español).

Uso:
    uv run python generate_voice_dataset.py

Descarga el modelo Piper es_MX-ingrid_olga-medium la primera vez (~70 MB).
Genera data/voice/<CLASE>/ con WAVs augmentados.
"""

import os
import wave
import urllib.request
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample

# ── Configuración ────────────────────────────────────────────────────────────

VOICES_DIR   = Path("voices")
DATA_VOICE   = Path("data") / "voice"
SAMPLE_RATE  = 22050          # Piper genera a 22050 Hz; se resamplea a 16000 tras
TARGET_SR    = 16000
SAMPLES_PER_CLASS = 240       # aprox: 8 palabras × 3 augments × 10 repeticiones

# Modelo Piper ES (descarga directa desde HuggingFace)
PIPER_MODEL_URL  = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
)
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
)
MODEL_PATH  = VOICES_DIR / "es_ES-davefx-medium.onnx"
CONFIG_PATH = VOICES_DIR / "es_ES-davefx-medium.onnx.json"

# ── Palabras por clase ────────────────────────────────────────────────────────

VOICE_CLASSES = {
    "STOP":       ["para", "stop", "detente", "alto", "parar", "detener", "frena", "quieto"],
    "ADELANTE":   ["adelante", "avanza", "sigue", "avanzar", "seguir", "hacia adelante", "continua", "recto"],
    "IZQUIERDA":  ["izquierda", "curva izquierda", "ir izquierda", "dobla izquierda",
                   "voltea izquierda", "gira izquierda suave", "tuerce izquierda", "izq"],
    "DERECHA":    ["derecha", "curva derecha", "ir derecha", "dobla derecha",
                   "voltea derecha", "gira derecha suave", "tuerce derecha", "der"],
    "GIRO_IZQ":   ["giro izquierda", "girar izquierda", "pivote izquierda",
                   "noventa izquierda", "vuelta izquierda", "girar a la izquierda",
                   "noventa grados izquierda", "giro noventa izquierda"],
    "GIRO_DER":   ["giro derecha", "girar derecha", "pivote derecha",
                   "noventa derecha", "vuelta derecha", "girar a la derecha",
                   "noventa grados derecha", "giro noventa derecha"],
}


def download_model() -> None:
    VOICES_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        print("Descargando modelo Piper ES (~65 MB)...")
        urllib.request.urlretrieve(PIPER_MODEL_URL, MODEL_PATH)
        print(f"  -> {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        print("Descargando config Piper ES...")
        urllib.request.urlretrieve(PIPER_CONFIG_URL, CONFIG_PATH)
        print(f"  -> {CONFIG_PATH}")


def synthesize_piper(voice, text: str) -> np.ndarray:
    """Sintetiza texto con Piper y devuelve audio float32 a TARGET_SR."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    with wave.open(tmp_path, "w") as wf:
        voice.synthesize_wav(text, wf)
    audio, sr = sf.read(tmp_path, dtype="float32")
    os.unlink(tmp_path)
    # Resamplear a TARGET_SR si es necesario
    if sr != TARGET_SR:
        n_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, n_samples).astype(np.float32)
    return audio


def augment(audio: np.ndarray) -> list:
    """Devuelve [original, speed_slow, speed_fast, noisy, quiet]."""
    variants = [audio.copy()]

    # Velocidad lenta (0.85×)
    slow = resample(audio, int(len(audio) / 0.85)).astype(np.float32)
    variants.append(slow[:len(audio)] if len(slow) >= len(audio)
                    else np.pad(slow, (0, len(audio) - len(slow))))

    # Velocidad rápida (1.15×)
    fast = resample(audio, int(len(audio) / 1.15)).astype(np.float32)
    variants.append(fast[:len(audio)] if len(fast) >= len(audio)
                    else np.pad(fast, (0, len(audio) - len(fast))))

    # Ruido blanco suave
    noisy = audio + np.random.normal(0, 0.005, len(audio)).astype(np.float32)
    variants.append(noisy.clip(-1, 1))

    # Volumen reducido
    quiet = (audio * 0.6).astype(np.float32)
    variants.append(quiet)

    return variants


def generate_dataset() -> None:
    from piper.voice import PiperVoice

    download_model()
    voice = PiperVoice.load(str(MODEL_PATH), config_path=str(CONFIG_PATH))
    print(f"\nModelo Piper cargado: {MODEL_PATH.name}")

    for cls_name, words in VOICE_CLASSES.items():
        out_dir = DATA_VOICE / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = 0
        print(f"\n[{cls_name}] Generando muestras...")

        word_cycle = (words * 30)[:SAMPLES_PER_CLASS // 5 + 1]
        for word in word_cycle:
            try:
                base_audio = synthesize_piper(voice, word)
            except Exception as e:
                print(f"  WARN: fallo '{word}': {e}")
                continue

            for variant in augment(base_audio):
                path = out_dir / f"sample_{idx:04d}.wav"
                sf.write(str(path), variant, TARGET_SR)
                idx += 1
                if idx >= SAMPLES_PER_CLASS:
                    break
            if idx >= SAMPLES_PER_CLASS:
                break

        print(f"  -> {idx} muestras en {out_dir}")

    print("\nDataset generado en data/voice/")


if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset()

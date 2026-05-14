# generate_voice_dataset.py
"""
Genera dataset de audio para control por voz.

Estrategia:
  - 4 voces Piper en español (ES/AR/MX) → diversidad de acento y timbre
  - Augmentación de velocidad y volumen por cada muestra limpia
  - Ruido sintético a 3 niveles SNR (babble, pink, white, car)  ← simula aula real
  - Reverb sintético (eco de sala)

Uso:
    uv run python generate_voice_dataset.py

La primera ejecución descarga 3 modelos extra (~200 MB en total, solo una vez).
Las siguientes ejecuciones usan los modelos cacheados en voices/.

Dataset resultante: ~1500 muestras/clase, 9000 total.
"""

import os
import wave
import urllib.request
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample, butter, sosfilt

# ── Configuración ─────────────────────────────────────────────────────────────

VOICES_DIR        = Path("voices")
DATA_VOICE        = Path("data") / "voice"
TARGET_SR         = 16000
SAMPLES_PER_VOICE = 70    # × 4 voces = 280 muestras limpias por clase
SNR_LEVELS        = [20, 10, 5]   # dB (suave, moderado, fuerte)

BASE_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

PIPER_VOICES = [
    {
        "name":        "davefx",           # España — voz masculina
        "model_path":  VOICES_DIR / "es_ES-davefx-medium.onnx",
        "config_path": VOICES_DIR / "es_ES-davefx-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
    },
    {
        "name":        "sharvard",         # España — segunda voz masculina
        "model_path":  VOICES_DIR / "es_ES-sharvard-medium.onnx",
        "config_path": VOICES_DIR / "es_ES-sharvard-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json",
    },
    {
        "name":        "daniela",          # Argentina — voz femenina alta calidad
        "model_path":  VOICES_DIR / "es_AR-daniela-high.onnx",
        "config_path": VOICES_DIR / "es_AR-daniela-high.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config_url":  f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
    },
    {
        "name":        "ald",              # México — voz masculina
        "model_path":  VOICES_DIR / "es_MX-ald-medium.onnx",
        "config_path": VOICES_DIR / "es_MX-ald-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
    },
]

# ── Palabras por clase (sin ambigüedad entre clases) ──────────────────────────

VOICE_CLASSES = {
    "ALTO":       [
        "alto", "para", "detente", "frena", "basta", "espera", "quieto",
        "detener", "parar", "stop", "detente ahí", "frena ya", "para ya",
        "quieto ahí", "no avances", "alto ahí", "para el carro",
        "no te muevas", "espera ahí",
    ],
    "ADELANTE":   [
        "adelante", "avanza", "sigue", "avanzar", "seguir", "hacia adelante",
        "continua", "recto", "ve recto", "sigue adelante", "continúa",
        "muévete", "avanza ya", "ve hacia adelante", "sigue recto",
        "directo", "ve directo", "ándale", "camina",
    ],
    "IZQUIERDA":  [
        "izquierda", "curva izquierda", "dobla izquierda",
        "voltea izquierda", "tuerce izquierda", "ve a la izquierda",
        "mueve izquierda", "vira izquierda", "hacia la izquierda",
        "dobla a la izquierda", "voltea a la izquierda",
        "tuerce a la izquierda", "ve a la izq", "a la izquierda",
        "muévete a la izquierda", "curva a la izquierda",
    ],
    "DERECHA":    [
        "derecha", "curva derecha", "dobla derecha",
        "voltea derecha", "tuerce derecha", "ve a la derecha",
        "mueve derecha", "vira derecha", "hacia la derecha",
        "dobla a la derecha", "voltea a la derecha",
        "tuerce a la derecha", "ve a la der", "a la derecha",
        "muévete a la derecha", "curva a la derecha",
    ],
    "GIRO_IZQ":   [
        "giro izquierda", "giro a la izquierda", "giro completo izquierda",
        "giro noventa izquierda", "giro noventa grados izquierda",
        "giro noventa", "girar izquierda", "gira izquierda",
        "gira a la izquierda", "vuelta izquierda", "vuelta a la izquierda",
        "media vuelta izquierda", "giro en u izquierda",
        "gira noventa izquierda", "gira noventa grados izquierda",
    ],
    "GIRO_DER":   [
        "giro derecha", "giro a la derecha", "giro completo derecha",
        "giro noventa derecha", "giro noventa grados derecha",
        "girar derecha", "gira derecha",
        "gira a la derecha", "vuelta derecha", "vuelta a la derecha",
        "media vuelta derecha", "giro en u derecha",
        "gira noventa derecha", "gira noventa grados derecha",
        "vuelta completa derecha",
    ],
}


# ── Descarga de voces ─────────────────────────────────────────────────────────

def download_voice(cfg: dict) -> bool:
    VOICES_DIR.mkdir(exist_ok=True)
    ok = True
    for key in ("model_path", "config_path"):
        path = cfg[key]
        url  = cfg[key.replace("path", "url")]
        if path.exists():
            continue
        print(f"  Descargando {path.name} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"  ERROR descargando {path.name}: {e}")
            ok = False
    return ok


# ── Síntesis TTS ──────────────────────────────────────────────────────────────

def synthesize(voice, text: str) -> np.ndarray:
    """Sintetiza texto y devuelve audio float32 mono a TARGET_SR."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "w") as wf:
            voice.synthesize_wav(text, wf)
        audio, sr = sf.read(tmp_path, dtype="float32")
    finally:
        os.unlink(tmp_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)
    return audio


def speed_augment(audio: np.ndarray) -> list:
    """Devuelve 7 variantes: velocidades extremas + volumen. Cubre habla rápida y lenta humana."""
    n = len(audio)
    variants = [audio.copy()]   # normal

    # Velocidades: 0.65× (muy lento) a 1.45× (muy rápido)
    for factor in (0.65, 0.82, 1.20, 1.45):
        stretched = resample(audio, int(n / factor)).astype(np.float32)
        if len(stretched) >= n:
            variants.append(stretched[:n])
        else:
            variants.append(np.pad(stretched, (0, n - len(stretched))))

    variants.append(np.clip(audio * 0.55, -1, 1).astype(np.float32))   # muy suave
    variants.append(np.clip(audio * 1.45, -1, 1).astype(np.float32))   # muy fuerte
    return variants   # 7 variantes


# ── Generadores de ruido sintético (100% offline, NumPy + SciPy) ──────────────

def _pink_noise(n: int) -> np.ndarray:
    """Ruido 1/f via FFT (similar al ruido ambiental real)."""
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1
    power   = 1.0 / np.sqrt(freqs)
    power[0] = 0
    phase   = 2 * np.pi * np.random.rand(len(freqs))
    noise   = np.fft.irfft(power * np.exp(1j * phase), n).astype(np.float32)
    mx = np.abs(noise).max()
    return noise / mx if mx > 1e-8 else noise


def _butter_bandpass(audio: np.ndarray, lo: float, hi: float,
                     sr: int = TARGET_SR) -> np.ndarray:
    nyq  = sr / 2
    low  = np.clip(lo / nyq, 1e-4, 0.999)
    high = np.clip(hi / nyq, 1e-4, 0.999)
    if low >= high:
        return audio
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _butter_lowpass(audio: np.ndarray, cutoff: float,
                    sr: int = TARGET_SR) -> np.ndarray:
    nyq  = sr / 2
    norm = np.clip(cutoff / nyq, 1e-4, 0.999)
    sos  = butter(4, norm, btype="low", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def noise_white(n: int) -> np.ndarray:
    """Ruido blanco uniforme."""
    noise = np.random.randn(n).astype(np.float32)
    return noise / (np.abs(noise).max() + 1e-8)


def noise_pink(n: int) -> np.ndarray:
    """Ruido rosa — suena más natural que el blanco."""
    return _pink_noise(n)


def noise_babble(n: int, sr: int = TARGET_SR) -> np.ndarray:
    """Simula murmullo de personas hablando (6 fuentes en rango de voz)."""
    result = np.zeros(n, dtype=np.float32)
    for _ in range(6):
        src = _pink_noise(n)
        lo  = np.random.uniform(200, 700)
        hi  = np.random.uniform(1200, 3500)
        result += _butter_bandpass(src, lo, hi, sr)
    mx = np.abs(result).max()
    return result / mx if mx > 1e-8 else result


def noise_car(n: int, sr: int = TARGET_SR) -> np.ndarray:
    """Ruido grave de motor / tráfico (< 250 Hz)."""
    return _butter_lowpass(_pink_noise(n), 250, sr)


NOISE_FNS = [noise_babble, noise_pink, noise_white, noise_car]


def mix_snr(speech: np.ndarray, noise_fn, snr_db: float) -> np.ndarray:
    """Mezcla speech + noise al SNR indicado (dB)."""
    noise  = noise_fn(len(speech))
    s_pow  = np.mean(speech ** 2)
    n_pow  = np.mean(noise  ** 2)
    if s_pow < 1e-10 or n_pow < 1e-10:
        return speech
    target = s_pow / (10 ** (snr_db / 10))
    scaled = noise * np.sqrt(target / n_pow)
    return (speech + scaled).clip(-1, 1).astype(np.float32)


# ── Reverb sintético ──────────────────────────────────────────────────────────

def add_reverb(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Convoluciona audio con impulso de sala sintético.
    Simula el eco de un aula o laboratorio (room_size 0.1–0.4 s).
    """
    room_s  = np.random.uniform(0.10, 0.35)
    ir_len  = int(sr * room_s)
    t       = np.linspace(0, 1, ir_len)
    decay   = np.random.uniform(4, 9)
    ir      = np.exp(-decay * t) * np.random.randn(ir_len).astype(np.float32)
    ir[0]   = 1.0   # camino directo
    result  = np.convolve(audio, ir)[:len(audio)]
    mx = np.abs(result).max()
    return (result / mx * 0.90).astype(np.float32) if mx > 1e-8 else audio


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate_dataset() -> None:
    from piper.voice import PiperVoice

    # ── Fase 1: cargar voces ──────────────────────────────────────────────────
    print("Cargando voces Piper...")
    loaded_voices: list[tuple[str, object]] = []
    for cfg in PIPER_VOICES:
        if download_voice(cfg):
            try:
                v = PiperVoice.load(str(cfg["model_path"]),
                                    config_path=str(cfg["config_path"]))
                loaded_voices.append((cfg["name"], v))
                print(f"  ✓ {cfg['name']}")
            except Exception as e:
                print(f"  ✗ {cfg['name']}: {e}")
        else:
            print(f"  ✗ {cfg['name']}: descarga fallida")

    if not loaded_voices:
        raise RuntimeError("No se pudo cargar ninguna voz. Verifica conexión a internet.")

    n_voices = len(loaded_voices)
    print(f"\nVoces activas: {n_voices}  "
          f"(~{SAMPLES_PER_VOICE * n_voices} limpias + "
          f"{SAMPLES_PER_VOICE * n_voices * len(SNR_LEVELS)} con ruido + "
          f"{SAMPLES_PER_VOICE * n_voices} con reverb por clase)\n")

    # ── Fase 2: generar por clase ─────────────────────────────────────────────
    for cls_name, words in VOICE_CLASSES.items():
        out_dir = DATA_VOICE / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_files: list[Path] = []
        global_idx = 0

        print(f"[{cls_name}]")

        # ── 2a: muestras limpias por voz ─────────────────────────────────────
        for v_name, voice in loaded_voices:
            v_count  = 0
            # Ciclar palabras para generar SAMPLES_PER_VOICE muestras
            word_cycle = (words * 20)[: SAMPLES_PER_VOICE // 5 + 2]
            for word in word_cycle:
                try:
                    base = synthesize(voice, word)
                except Exception:
                    continue
                for variant in speed_augment(base):
                    fname = out_dir / f"clean_{v_name}_{global_idx:05d}.wav"
                    sf.write(str(fname), variant, TARGET_SR)
                    clean_files.append(fname)
                    global_idx += 1
                    v_count    += 1
                    if v_count >= SAMPLES_PER_VOICE:
                        break
                if v_count >= SAMPLES_PER_VOICE:
                    break
            print(f"  {v_name}: {v_count} muestras limpias")

        # ── 2b: ruido a 3 SNR ────────────────────────────────────────────────
        noise_count = 0
        for snr in SNR_LEVELS:
            for i, src_path in enumerate(clean_files):
                audio, _ = sf.read(str(src_path), dtype="float32")
                noise_fn  = NOISE_FNS[i % len(NOISE_FNS)]
                mixed     = mix_snr(audio, noise_fn, snr)
                fname     = out_dir / f"noise_snr{snr:02d}_{global_idx:05d}.wav"
                sf.write(str(fname), mixed, TARGET_SR)
                global_idx += 1
                noise_count += 1
        print(f"  ruido (3 SNR × {len(clean_files)} limpias): {noise_count} muestras")

        # ── 2c: reverb ───────────────────────────────────────────────────────
        reverb_count = 0
        for src_path in clean_files:
            audio, _ = sf.read(str(src_path), dtype="float32")
            rev   = add_reverb(audio)
            fname = out_dir / f"reverb_{global_idx:05d}.wav"
            sf.write(str(fname), rev, TARGET_SR)
            global_idx += 1
            reverb_count += 1
        print(f"  reverb: {reverb_count} muestras")

        total = len(list(out_dir.glob("*.wav")))
        print(f"  → TOTAL {cls_name}: {total} muestras\n")

    grand_total = sum(
        len(list((DATA_VOICE / cls).glob("*.wav")))
        for cls in VOICE_CLASSES
    )
    print(f"Dataset completo: {grand_total} muestras en {DATA_VOICE}/")


if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset()

# generate_voice_dataset.py
"""
Genera dataset de audio para control por voz.

Estrategia:
  - 8 voces Piper en español (ES/AR/MX) → máxima diversidad de acento y timbre
  - Augmentación por muestra: velocidad (4 niveles), volumen, pitch (grave/agudo) = 9 variantes
  - Ruido sintético a 3 niveles SNR (babble, pink, white, car)  ← simula aula real
  - Reverb sintético (eco de sala)
  - Filtro de micrófono (300–3400 Hz) — simula mic barato o teléfono

Uso:
    uv run python generate_voice_dataset.py

La primera ejecución descarga los modelos nuevos (~300 MB total, solo una vez).
Las siguientes ejecuciones usan los modelos cacheados en voices/.

Dataset resultante: ~2000 muestras/clase, 12000 total.
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
SAMPLES_PER_VOICE = 45    # × 8 voces = 360 muestras limpias por clase
SNR_LEVELS        = [20, 10, 5]   # dB (suave, moderado, fuerte)

BASE_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

PIPER_VOICES = [
    # ── España ────────────────────────────────────────────────────────────────
    {
        "name":        "davefx",
        "model_path":  VOICES_DIR / "es_ES-davefx-medium.onnx",
        "config_path": VOICES_DIR / "es_ES-davefx-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json",
    },
    {
        "name":        "sharvard",
        "model_path":  VOICES_DIR / "es_ES-sharvard-medium.onnx",
        "config_path": VOICES_DIR / "es_ES-sharvard-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json",
    },
    {
        "name":        "carlfm",           # España — voz compacta x_low
        "model_path":  VOICES_DIR / "es_ES-carlfm-x_low.onnx",
        "config_path": VOICES_DIR / "es_ES-carlfm-x_low.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json",
    },
    {
        "name":        "mls_10246",        # España — voz MLS
        "model_path":  VOICES_DIR / "es_ES-mls_10246-low.onnx",
        "config_path": VOICES_DIR / "es_ES-mls_10246-low.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx.json",
    },
    {
        "name":        "mls_9972",         # España — segunda voz MLS
        "model_path":  VOICES_DIR / "es_ES-mls_9972-low.onnx",
        "config_path": VOICES_DIR / "es_ES-mls_9972-low.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx",
        "config_url":  f"{BASE_HF}/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx.json",
    },
    # ── Argentina ─────────────────────────────────────────────────────────────
    {
        "name":        "daniela",          # Argentina — voz femenina alta calidad
        "model_path":  VOICES_DIR / "es_AR-daniela-high.onnx",
        "config_path": VOICES_DIR / "es_AR-daniela-high.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config_url":  f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
    },
    # ── México ────────────────────────────────────────────────────────────────
    {
        "name":        "ald",              # México — voz masculina
        "model_path":  VOICES_DIR / "es_MX-ald-medium.onnx",
        "config_path": VOICES_DIR / "es_MX-ald-medium.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_MX/ald/medium/es_MX-ald-medium.onnx",
        "config_url":  f"{BASE_HF}/es/es_MX/ald/medium/es_MX-ald-medium.onnx.json",
    },
    {
        "name":        "claude_mx",        # México — voz alta calidad
        "model_path":  VOICES_DIR / "es_MX-claude-high.onnx",
        "config_path": VOICES_DIR / "es_MX-claude-high.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "config_url":  f"{BASE_HF}/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
    },
]

# ── Palabras por clase (3 por clase, fonéticamente distintas) ─────────────────

VOICE_CLASSES = {
    "STOP":      ["stop", "para", "alto"],
    "ADELANTE":  ["adelante", "sigue", "avanza"],
    "IZQUIERDA": ["izquierda", "a la izquierda", "dobla izquierda"],
    "DERECHA":   ["derecha", "a la derecha", "dobla derecha"],
    "GIRO_IZQ":  ["giro izquierda", "gira izquierda", "giro a la izquierda"],
    "GIRO_DER":  ["giro derecha",   "gira derecha",   "giro a la derecha"],
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


# ── Pitch shifting (tono sin cambiar duración) ────────────────────────────────

def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """
    Cambia el tono N semitonos sin alterar la duración.
    Técnica: resampleo doble (cambia pitch+speed, luego restaura speed).
    """
    factor   = 2 ** (semitones / 12)
    n        = len(audio)
    # Paso 1: cambiar velocidad/tono
    pitched  = resample(audio, int(n / factor)).astype(np.float32)
    # Paso 2: restaurar duración (preserva el tono cambiado)
    return resample(pitched, n).astype(np.float32)


# ── Augmentación de variantes por muestra ────────────────────────────────────

def base_augment(audio: np.ndarray) -> list:
    """
    9 variantes por muestra base:
      velocidad × 4 (muy lento → muy rápido)
      volumen   × 2 (suave, fuerte)
      pitch     × 2 (grave −4 st, agudo +4 st)
      + original
    Cubre habla lenta/rápida, voces graves/agudas, micrófonos tímidos/fuertes.
    """
    n        = len(audio)
    variants = [audio.copy()]

    # Velocidades: 0.65× (muy lento) → 1.45× (muy rápido)
    for factor in (0.65, 0.82, 1.20, 1.45):
        stretched = resample(audio, int(n / factor)).astype(np.float32)
        if len(stretched) >= n:
            variants.append(stretched[:n])
        else:
            variants.append(np.pad(stretched, (0, n - len(stretched))))

    # Volumen
    variants.append(np.clip(audio * 0.50, -1, 1).astype(np.float32))   # muy suave
    variants.append(np.clip(audio * 1.50, -1, 1).astype(np.float32))   # muy fuerte

    # Tono
    variants.append(pitch_shift(audio, -4))   # voz grave
    variants.append(pitch_shift(audio, +4))   # voz aguda

    return variants   # 9 variantes


# ── Generadores de ruido sintético (100% offline, NumPy + SciPy) ──────────────

def _pink_noise(n: int) -> np.ndarray:
    """Ruido 1/f via FFT (similar al ruido ambiental real)."""
    freqs    = np.fft.rfftfreq(n)
    freqs[0] = 1
    power    = 1.0 / np.sqrt(freqs)
    power[0] = 0
    phase    = 2 * np.pi * np.random.rand(len(freqs))
    noise    = np.fft.irfft(power * np.exp(1j * phase), n).astype(np.float32)
    mx       = np.abs(noise).max()
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
    noise = np.random.randn(n).astype(np.float32)
    return noise / (np.abs(noise).max() + 1e-8)


def noise_pink(n: int) -> np.ndarray:
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
    """Convoluciona audio con impulso de sala sintético (eco de aula 0.1–0.35 s)."""
    room_s = np.random.uniform(0.10, 0.35)
    ir_len = int(sr * room_s)
    t      = np.linspace(0, 1, ir_len)
    decay  = np.random.uniform(4, 9)
    ir     = np.exp(-decay * t) * np.random.randn(ir_len).astype(np.float32)
    ir[0]  = 1.0
    result = np.convolve(audio, ir)[:len(audio)]
    mx     = np.abs(result).max()
    return (result / mx * 0.90).astype(np.float32) if mx > 1e-8 else audio


# ── Filtro de micrófono ───────────────────────────────────────────────────────

def mic_filter(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Bandpass 300–3400 Hz — simula micrófono barato o teléfono."""
    return _butter_bandpass(audio, 300.0, 3400.0, sr)


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
    n_clean  = SAMPLES_PER_VOICE * n_voices
    print(f"\nVoces activas: {n_voices}  "
          f"(~{n_clean} limpias + {n_clean * len(SNR_LEVELS)} con ruido + "
          f"{n_clean} reverb + {n_clean} mic-filter por clase)\n")

    # ── Fase 2: generar por clase ─────────────────────────────────────────────
    for cls_name, words in VOICE_CLASSES.items():
        out_dir = DATA_VOICE / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_files: list[Path] = []
        global_idx = 0

        print(f"[{cls_name}]")

        # ── 2a: muestras limpias (9 variantes por síntesis) ──────────────────
        for v_name, voice in loaded_voices:
            v_count    = 0
            n_variants = 9   # base_augment produce 9 variantes
            word_cycle = (words * 20)[: SAMPLES_PER_VOICE // n_variants + 2]
            for word in word_cycle:
                try:
                    base = synthesize(voice, word)
                except Exception:
                    continue
                for variant in base_augment(base):
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
                global_idx  += 1
                noise_count += 1
        print(f"  ruido (3 SNR × {len(clean_files)} limpias): {noise_count} muestras")

        # ── 2c: reverb ───────────────────────────────────────────────────────
        reverb_count = 0
        for src_path in clean_files:
            audio, _ = sf.read(str(src_path), dtype="float32")
            rev   = add_reverb(audio)
            fname = out_dir / f"reverb_{global_idx:05d}.wav"
            sf.write(str(fname), rev, TARGET_SR)
            global_idx   += 1
            reverb_count += 1
        print(f"  reverb: {reverb_count} muestras")

        # ── 2d: filtro de micrófono ───────────────────────────────────────────
        mic_count = 0
        for src_path in clean_files:
            audio, _ = sf.read(str(src_path), dtype="float32")
            filtered  = mic_filter(audio)
            fname     = out_dir / f"mic_{global_idx:05d}.wav"
            sf.write(str(fname), filtered, TARGET_SR)
            global_idx += 1
            mic_count  += 1
        print(f"  mic filter: {mic_count} muestras")

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

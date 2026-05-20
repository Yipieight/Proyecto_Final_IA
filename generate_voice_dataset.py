# generate_voice_dataset.py
"""
Genera dataset de audio para control por voz.

Estrategia:
  - 3 voces Piper high (AR/MX) + 3 voces Kokoro ONNX (ES) = 6 voces de alta calidad
    (se eliminaron davefx-medium y sharvard-medium: robóticas y muy similares entre sí)
  - Variantes fonéticas para comandos de 2 palabras (giro izquierda / gira izquierda)
  - Augmentación por muestra: velocidad (4 niveles), volumen, pitch (grave/agudo) = 13 variantes
  - 7 escenarios de ruido: aula, multitud, lluvia, viento, tráfico, rosa, blanco
  - Ruido a 3 niveles SNR (20/10/5 dB)
  - Reverb sintético (eco de sala)
  - Filtro de micrófono (300–3400 Hz) — simula mic barato o teléfono
  - Augmentación compuesta: reverb+ruido, mic+ruido, clipping, EQ, doble ruido, eco, lugar público

Uso:
    uv run python generate_voice_dataset.py
    uv run python generate_voice_dataset.py --only GIRO_IZQ

Dataset resultante: ~5500 muestras/clase, ~33000 total.
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
SAMPLES_PER_VOICE = 65    # × 6 voces (3 Piper high + 3 Kokoro) = 390 limpias por clase

KOKORO_MODEL  = VOICES_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_BIN = VOICES_DIR / "voices-v1.0.bin"
KOKORO_VOICES_ES = ["ef_dora", "em_alex", "em_santa"]   # ♀ + 2♂ español

EDGE_VOICES = [
    {"name": "gt_andres", "voice_id": "es-GT-AndresNeural"},   # Guatemala ♂ — acento centroamericano
    {"name": "mx_dalia",  "voice_id": "es-MX-DaliaNeural"},    # México ♀ — acento mexicano claro
]
SNR_LEVELS        = [20, 10, 5]   # dB (suave, moderado, fuerte)

BASE_HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

PIPER_VOICES = [
    # ── Argentina — high quality ───────────────────────────────────────────────
    {
        "name":        "daniela",          # femenina, alta calidad (la mejor Piper ES)
        "model_path":  VOICES_DIR / "es_AR-daniela-high.onnx",
        "config_path": VOICES_DIR / "es_AR-daniela-high.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "config_url":  f"{BASE_HF}/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
    },
    # ── México — high quality ─────────────────────────────────────────────────
    {
        "name":        "claude_mx",        # masculina, alta calidad, acento mexicano
        "model_path":  VOICES_DIR / "es_MX-claude-high.onnx",
        "config_path": VOICES_DIR / "es_MX-claude-high.onnx.json",
        "model_url":   f"{BASE_HF}/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "config_url":  f"{BASE_HF}/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
    },
    # Voces eliminadas: davefx-medium y sharvard-medium (España)
    # Motivo: calidad medium, sonido robótico/monótono, ambas muy similares entre sí.
    # Reemplazadas aumentando SAMPLES_PER_VOICE de 45 → 65 en las 3 voces restantes.
]

# ── Palabras por clase (3 por clase, fonéticamente distintas) ─────────────────

VOICE_CLASSES = {
    "ALTO":      ["alto"],
    "ADELANTE":  ["adelante"],
    "IZQUIERDA": ["izquierda"],
    "DERECHA":   ["derecha"],
    # Para los comandos de 2 palabras se añaden variantes fonéticas naturales:
    # el TTS los sintetiza con ritmos distintos (sustantivo vs imperativo)
    # lo que amplía la variabilidad temporal y mejora el reconocimiento en vivo.
    "GIRO_IZQ":  ["giro a la izquierda"],
    "GIRO_DER":  ["giro a la derecha"],
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

def _piper_synth_fn(piper_voice):
    """Devuelve función synth(text)->ndarray para una voz Piper."""
    def synth(text: str) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "w") as wf:
                piper_voice.synthesize_wav(text, wf)
            audio, sr = sf.read(tmp_path, dtype="float32")
        finally:
            os.unlink(tmp_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)
        return audio.astype(np.float32)
    return synth


def _kokoro_synth_fn(kokoro, voice_name: str, lang: str = "es"):
    """Devuelve función synth(text)->ndarray para una voz Kokoro ONNX."""
    def synth(text: str) -> np.ndarray:
        samples, sr = kokoro.create(text, voice=voice_name, lang=lang)
        if hasattr(samples, 'numpy'):
            samples = samples.numpy()
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != TARGET_SR:
            samples = resample(samples, int(len(samples) * TARGET_SR / sr)).astype(np.float32)
        return samples
    return synth


def _edge_synth_fn(voice_name: str):
    """Devuelve función synth(text)->ndarray para una voz Edge-TTS (Microsoft Neural)."""
    import asyncio, tempfile, subprocess, os
    def synth(text: str) -> np.ndarray:
        import edge_tts
        async def _run():
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(tmp_path)
            return tmp_path
        tmp_mp3 = asyncio.run(_run())
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        subprocess.run(["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(TARGET_SR), "-ac", "1", wav_path],
                       capture_output=True)
        os.unlink(tmp_mp3)
        audio, sr = sf.read(wav_path, dtype="float32")
        os.unlink(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
    return synth


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
    13 variantes por muestra base:
      velocidad × 4 (muy lento → muy rápido)
      volumen   × 2 (suave, fuerte)
      pitch     × 6 (±2, ±4, ±6 semitonos — cubre todo el rango vocal humano)
      + original
    """
    n        = len(audio)
    variants = [audio.copy()]

    for factor in (0.65, 0.82, 1.20, 1.45):
        stretched = resample(audio, int(n / factor)).astype(np.float32)
        if len(stretched) >= n:
            variants.append(stretched[:n])
        else:
            variants.append(np.pad(stretched, (0, n - len(stretched))))

    variants.append(np.clip(audio * 0.50, -1, 1).astype(np.float32))
    variants.append(np.clip(audio * 1.50, -1, 1).astype(np.float32))

    for st in (-6, -2, +2, +6):          # ±2 y ±6 semitonos adicionales
        variants.append(pitch_shift(audio, st))
    variants.append(pitch_shift(audio, -4))
    variants.append(pitch_shift(audio, +4))

    return variants   # 13 variantes


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


def noise_crowd(n: int, sr: int = TARGET_SR) -> np.ndarray:
    """Multitud densa en público — 15 fuentes de voz superpuestas."""
    result = np.zeros(n, dtype=np.float32)
    for _ in range(15):
        src = _pink_noise(n)
        lo  = np.random.uniform(100, 600)
        hi  = np.random.uniform(1500, 4500)
        result += _butter_bandpass(src, lo, hi, sr)
    mx = np.abs(result).max()
    return result / mx if mx > 1e-8 else result


def noise_rain(n: int, sr: int = TARGET_SR) -> np.ndarray:
    """Lluvia — componente de alta frecuencia + retumbo grave."""
    high   = _butter_bandpass(_pink_noise(n), 2000, 7000, sr)
    rumble = _butter_lowpass(_pink_noise(n), 300, sr)
    result = (0.70 * high + 0.30 * rumble).astype(np.float32)
    mx = np.abs(result).max()
    return result / mx if mx > 1e-8 else result


def noise_wind(n: int, sr: int = TARGET_SR) -> np.ndarray:
    """Viento — ruido de baja-media frecuencia con ráfagas moduladas."""
    base = _butter_bandpass(_pink_noise(n), 80, 900, sr)
    t    = np.linspace(0, n / sr, n)
    gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t + np.random.uniform(0, 2 * np.pi))
    result = (base * gust).astype(np.float32)
    mx = np.abs(result).max()
    return result / mx if mx > 1e-8 else result


# 7 escenarios: aula, multitud, lluvia, viento, tráfico, rosa, blanco
NOISE_FNS = [noise_babble, noise_crowd, noise_rain, noise_wind,
             noise_car, noise_pink, noise_white]


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


# ── Clipping / distorsión ─────────────────────────────────────────────────────

def add_clipping(audio: np.ndarray) -> np.ndarray:
    """Simula micrófono saturado: recorta la señal al 55–80% de su pico."""
    mx = np.abs(audio).max()
    if mx < 1e-8:
        return audio
    threshold = np.random.uniform(0.55, 0.80)
    clipped   = np.clip(audio / mx, -threshold, threshold)
    return (clipped / threshold * mx).astype(np.float32)


# ── EQ aleatorio ─────────────────────────────────────────────────────────────

def random_eq(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Boost/cut aleatorio en 2–3 bandas — simula diferentes salas/micrófonos."""
    result  = audio.copy().astype(np.float64)
    n_bands = np.random.randint(2, 4)
    for _ in range(n_bands):
        lo   = np.random.uniform(150, 4000)
        hi   = min(lo * np.random.uniform(1.5, 4.0), 7000)
        gain = np.random.uniform(0.25, 2.5)
        band = _butter_bandpass(audio, lo, hi, sr).astype(np.float64)
        result += (gain - 1.0) * band
    peak = np.abs(result).max()
    if peak > 1e-8:
        result = result / peak * np.abs(audio).max()
    return result.clip(-1, 1).astype(np.float32)


# ── Ruido doble combinado ─────────────────────────────────────────────────────

NOISE_PAIRS = [
    (noise_babble, noise_rain),
    (noise_crowd,  noise_wind),
    (noise_babble, noise_car),
    (noise_crowd,  noise_rain),
    (noise_rain,   noise_white),
    (noise_wind,   noise_car),
    (noise_babble, noise_pink),
]

def mix_double_noise(speech: np.ndarray, fn1, fn2, snr_db: float) -> np.ndarray:
    """Mezcla speech con dos tipos de ruido simultáneos al SNR indicado."""
    n  = len(speech)
    n1 = fn1(n).astype(np.float64)
    n2 = fn2(n).astype(np.float64)
    combined = (n1 + n2) * 0.5
    s_pow = float(np.mean(speech.astype(np.float64) ** 2))
    n_pow = float(np.mean(combined ** 2))
    if s_pow < 1e-10 or n_pow < 1e-10:
        return speech
    target = s_pow / (10 ** (snr_db / 10))
    scaled = combined * np.sqrt(target / n_pow)
    return (speech.astype(np.float64) + scaled).clip(-1, 1).astype(np.float32)


# ── Eco de pasillo (delay discreto) ──────────────────────────────────────────

def corridor_echo(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """1–3 reflexiones con delay 50–200 ms — simula pasillo o pared lejana."""
    result   = audio.astype(np.float64).copy()
    n_echoes = np.random.randint(1, 4)
    for i in range(1, n_echoes + 1):
        delay_s   = np.random.uniform(0.05, 0.20)
        delay_smp = int(sr * delay_s)
        decay     = np.random.uniform(0.25, 0.55) ** i
        if delay_smp < len(audio):
            delayed           = np.zeros(len(audio), dtype=np.float64)
            delayed[delay_smp:] = audio[:len(audio) - delay_smp] * decay
            result           += delayed
    peak = np.abs(result).max()
    return (result / peak * 0.90).astype(np.float32) if peak > 1e-8 else audio


# ── Lugar público ────────────────────────────────────────────────────────────

def public_place(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Voz baja/media (25–55 %) con multitud fuerte de fondo (SNR 1–5 dB).
    Simula hablar en pasillo, auditorio o salón lleno de gente."""
    vol     = np.random.uniform(0.25, 0.55)
    speech  = (audio * vol).astype(np.float32)
    noise_fn = [noise_babble, noise_crowd][np.random.randint(0, 2)]
    snr_db  = np.random.uniform(1.0, 5.0)
    return mix_snr(speech, noise_fn, snr_db)


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate_dataset(only: str | None = None) -> None:
    from piper.voice import PiperVoice
    from kokoro_onnx import Kokoro

    # ── Fase 1a: cargar voces Piper ───────────────────────────────────────────
    print("Cargando voces Piper (medium/high)...")
    loaded_voices: list[tuple[str, object]] = []
    for cfg in PIPER_VOICES:
        if download_voice(cfg):
            try:
                v = PiperVoice.load(str(cfg["model_path"]),
                                    config_path=str(cfg["config_path"]))
                loaded_voices.append((cfg["name"], _piper_synth_fn(v)))
                print(f"  ✓ piper/{cfg['name']}")
            except Exception as e:
                print(f"  ✗ piper/{cfg['name']}: {e}")
        else:
            print(f"  ✗ piper/{cfg['name']}: descarga fallida")

    # ── Fase 1b: cargar voces Kokoro ONNX ────────────────────────────────────
    print("Cargando voces Kokoro ONNX...")
    if KOKORO_MODEL.exists() and KOKORO_VOICES_BIN.exists():
        try:
            kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES_BIN))
            for v_name in KOKORO_VOICES_ES:
                loaded_voices.append((f"kokoro_{v_name}", _kokoro_synth_fn(kokoro, v_name)))
                print(f"  ✓ kokoro/{v_name}")
        except Exception as e:
            print(f"  ✗ Kokoro no disponible: {e}")
    else:
        print("  ✗ Modelos Kokoro no encontrados en voices/ — solo se usará Piper")

    # ── Fase 1c: cargar voces Edge-TTS (Microsoft Neural) ────────────────────
    print("Cargando voces Edge-TTS (Microsoft Neural)...")
    for cfg in EDGE_VOICES:
        try:
            fn = _edge_synth_fn(cfg["voice_id"])
            loaded_voices.append((f"edge_{cfg['name']}", fn))
            print(f"  ✓ edge/{cfg['name']}")
        except Exception as e:
            print(f"  ✗ edge/{cfg['name']}: {e}")

    if not loaded_voices:
        raise RuntimeError("No se pudo cargar ninguna voz.")

    n_voices = len(loaded_voices)
    n_clean  = SAMPLES_PER_VOICE * n_voices
    print(f"\nVoces activas: {n_voices}  "
          f"(~{n_clean} limpias + {n_clean * len(SNR_LEVELS)} con ruido + "
          f"{n_clean} reverb + {n_clean} mic-filter por clase)\n")

    # ── Fase 2: generar por clase ─────────────────────────────────────────────
    classes_to_gen = {only: VOICE_CLASSES[only]} if only else VOICE_CLASSES
    for cls_name, words in classes_to_gen.items():
        out_dir = DATA_VOICE / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_synths: list[np.ndarray] = []   # audios crudos sin aumentación
        global_idx = 0

        print(f"[{cls_name}]")

        # ── 2a: muestras limpias (13 variantes por síntesis) ─────────────────
        for v_name, synth_fn in loaded_voices:
            v_count    = 0
            n_variants = 13  # base_augment produce 13 variantes
            word_cycle = (words * 20)[: SAMPLES_PER_VOICE // n_variants + 2]
            for word in word_cycle:
                try:
                    base = synth_fn(word)
                except Exception:
                    continue
                raw_synths.append(base)          # guardar audio crudo
                for variant in base_augment(base):
                    fname = out_dir / f"clean_{v_name}_{global_idx:05d}.wav"
                    sf.write(str(fname), variant, TARGET_SR)
                    global_idx += 1
                    v_count    += 1
                    if v_count >= SAMPLES_PER_VOICE:
                        break
                if v_count >= SAMPLES_PER_VOICE:
                    break
            print(f"  {v_name}: {v_count} muestras limpias")

        # Expandir audios crudos × 13 variantes base → base para fases 2b–2k
        # Cada fase aplica su augmentación sobre combinaciones independientes
        # de velocidad, pitch y volumen (no sobre los ya procesados de fase 2a)
        all_variants: list[np.ndarray] = []
        for raw in raw_synths:
            all_variants.extend(list(base_augment(raw)))

        # ── 2b: ruido a 3 SNR ────────────────────────────────────────────────
        noise_count = 0
        for snr in SNR_LEVELS:
            for i, variant in enumerate(all_variants):
                noise_fn = NOISE_FNS[i % len(NOISE_FNS)]
                mixed    = mix_snr(variant, noise_fn, snr)
                fname    = out_dir / f"noise_snr{snr:02d}_{global_idx:05d}.wav"
                sf.write(str(fname), mixed, TARGET_SR)
                global_idx  += 1
                noise_count += 1
        print(f"  ruido (3 SNR × {len(all_variants)} variantes): {noise_count} muestras")

        # ── 2c: reverb ───────────────────────────────────────────────────────
        reverb_count = 0
        for variant in all_variants:
            rev   = add_reverb(variant)
            fname = out_dir / f"reverb_{global_idx:05d}.wav"
            sf.write(str(fname), rev, TARGET_SR)
            global_idx   += 1
            reverb_count += 1
        print(f"  reverb: {reverb_count} muestras")

        # ── 2d: filtro de micrófono ───────────────────────────────────────────
        mic_count = 0
        for variant in all_variants:
            filtered = mic_filter(variant)
            fname    = out_dir / f"mic_{global_idx:05d}.wav"
            sf.write(str(fname), filtered, TARGET_SR)
            global_idx += 1
            mic_count  += 1
        print(f"  mic filter: {mic_count} muestras")

        # ── 2e: reverb + ruido ────────────────────────────────────────────────
        rev_noise_count = 0
        for i, variant in enumerate(all_variants):
            rev      = add_reverb(variant)
            noise_fn = NOISE_FNS[i % len(NOISE_FNS)]
            snr      = SNR_LEVELS[i % len(SNR_LEVELS)]
            compound = mix_snr(rev, noise_fn, snr)
            fname    = out_dir / f"rev_noise_{global_idx:05d}.wav"
            sf.write(str(fname), compound, TARGET_SR)
            global_idx      += 1
            rev_noise_count += 1
        print(f"  reverb+ruido: {rev_noise_count} muestras")

        # ── 2f: mic + ruido ───────────────────────────────────────────────────
        mic_noise_count = 0
        for i, variant in enumerate(all_variants):
            filtered = mic_filter(variant)
            noise_fn = NOISE_FNS[(i + 3) % len(NOISE_FNS)]
            snr      = SNR_LEVELS[(i + 1) % len(SNR_LEVELS)]
            compound = mix_snr(filtered, noise_fn, snr)
            fname    = out_dir / f"mic_noise_{global_idx:05d}.wav"
            sf.write(str(fname), compound, TARGET_SR)
            global_idx      += 1
            mic_noise_count += 1
        print(f"  mic+ruido: {mic_noise_count} muestras")

        # ── 2g: clipping ─────────────────────────────────────────────────────
        clip_count = 0
        for variant in all_variants:
            clipped = add_clipping(variant)
            fname   = out_dir / f"clip_{global_idx:05d}.wav"
            sf.write(str(fname), clipped, TARGET_SR)
            global_idx += 1
            clip_count += 1
        print(f"  clipping: {clip_count} muestras")

        # ── 2h: EQ aleatorio ──────────────────────────────────────────────────
        eq_count = 0
        for variant in all_variants:
            eq_audio = random_eq(variant)
            fname    = out_dir / f"eq_{global_idx:05d}.wav"
            sf.write(str(fname), eq_audio, TARGET_SR)
            global_idx += 1
            eq_count   += 1
        print(f"  EQ aleatorio: {eq_count} muestras")

        # ── 2i: ruido doble combinado ─────────────────────────────────────────
        double_count = 0
        for i, variant in enumerate(all_variants):
            fn1, fn2 = NOISE_PAIRS[i % len(NOISE_PAIRS)]
            snr      = SNR_LEVELS[i % len(SNR_LEVELS)]
            mixed    = mix_double_noise(variant, fn1, fn2, snr)
            fname    = out_dir / f"doublenoise_{global_idx:05d}.wav"
            sf.write(str(fname), mixed, TARGET_SR)
            global_idx   += 1
            double_count += 1
        print(f"  ruido doble: {double_count} muestras")

        # ── 2j: eco de pasillo ────────────────────────────────────────────────
        echo_count = 0
        for variant in all_variants:
            echo  = corridor_echo(variant)
            fname = out_dir / f"echo_{global_idx:05d}.wav"
            sf.write(str(fname), echo, TARGET_SR)
            global_idx += 1
            echo_count += 1
        print(f"  eco pasillo: {echo_count} muestras")

        # ── 2k: lugar público (voz baja + multitud fuerte) ────────────────────
        public_count = 0
        for variant in all_variants:
            pub   = public_place(variant)
            fname = out_dir / f"public_{global_idx:05d}.wav"
            sf.write(str(fname), pub, TARGET_SR)
            global_idx   += 1
            public_count += 1
        print(f"  lugar público: {public_count} muestras")

        total = len(list(out_dir.glob("*.wav")))
        print(f"  → TOTAL {cls_name}: {total} muestras\n")

    grand_total = sum(
        len(list((DATA_VOICE / cls).glob("*.wav")))
        for cls in VOICE_CLASSES
    )
    print(f"Dataset completo: {grand_total} muestras en {DATA_VOICE}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Generar solo una clase (ej: --only DETENER)")
    args = parser.parse_args()
    np.random.seed(42)
    generate_dataset(only=args.only)

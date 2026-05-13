# Voice Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Crear la rama `feature/voice-control` con un pipeline completo de control por voz: Piper TTS genera el dataset en español, un VoiceCNN entrenado desde cero clasifica comandos de audio en tiempo real, y los envía por UDP al ESP32 usando el mismo protocolo y firmware existente.

**Architecture:** Micrófono → sounddevice (captura) → VAD por energía → mel-spectrogram NumPy manual (64×64) → VoiceCNN (PyTorch, 3 conv blocks) → argmax → byte UDP → ESP32 puerto 9999. El dataset se genera offline con Piper TTS español (múltiples voces + augmentación).

**Tech Stack:** PyTorch, NumPy, SciPy, piper-tts, sounddevice, soundfile. Reutiliza `utils.py` (ESP32_IP, ESP32_PORT, CMD_*). Cero cambios a archivos existentes en `main`.

---

## Archivos del proyecto

### Nuevos (solo en `feature/voice-control`)
| Archivo | Responsabilidad |
|---------|----------------|
| `generate_voice_dataset.py` | Descarga modelo Piper ES, genera WAVs por clase con augmentación |
| `voice_dataset.py` | PyTorch Dataset + mel-spectrogram NumPy manual + `_mel_filterbank()` |
| `model_voice.py` | `VoiceCNN` (3 conv + 3 FC) + `build_voice_model()` |
| `train_voice.py` | Entrenamiento, validación, guarda `models/voice_model.pth` |
| `main_voice.py` | Pipeline en tiempo real: mic → VAD → mel → CNN → UDP |

### Modificados
| Archivo | Cambio |
|---------|--------|
| `pyproject.toml` | Agregar dependencias: piper-tts, sounddevice, soundfile, scipy |

### Sin tocar (NUNCA modificar)
`main_robot.py`, `utils.py`, `state_machine.py`, `model_nav.py`, `dataset.py`, `train_kfold.py`, `preprocessing.py`, `esp32_firmware/`, etc.

---

## Task 1: Crear rama y agregar dependencias

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Crear la rama**

```bash
git checkout -b feature/voice-control
```

Expected: `Switched to a new branch 'feature/voice-control'`

- [ ] **Step 2: Agregar dependencias con uv**

```bash
uv add piper-tts sounddevice soundfile scipy
```

Expected: packages added to `pyproject.toml` and `uv.lock` updated.

- [ ] **Step 3: Verificar imports**

```bash
uv run python -c "import sounddevice; import soundfile; import scipy; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(voice): agregar dependencias piper-tts sounddevice soundfile scipy"
```

---

## Task 2: Crear generate_voice_dataset.py

**Files:**
- Create: `generate_voice_dataset.py`

Genera el dataset de audio usando Piper TTS con voz española. Descarga el modelo `es_MX-ingrid_olga-medium` la primera vez. Genera 200–250 WAVs por clase con 3 tipos de augmentación (velocidad, ruido, volumen).

- [ ] **Step 1: Crear el archivo**

```python
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

# Modelo Piper ES (descarga directa desde HuggingFace releases)
PIPER_MODEL_URL  = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    "/es/es_MX/ingrid_olga/medium/es_MX-ingrid_olga-medium.onnx"
)
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    "/es/es_MX/ingrid_olga/medium/es_MX-ingrid_olga-medium.onnx.json"
)
MODEL_PATH  = VOICES_DIR / "es_MX-ingrid_olga-medium.onnx"
CONFIG_PATH = VOICES_DIR / "es_MX-ingrid_olga-medium.onnx.json"

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
        print(f"Descargando modelo Piper ES (~65 MB)...")
        urllib.request.urlretrieve(PIPER_MODEL_URL, MODEL_PATH)
        print(f"  -> {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        print(f"Descargando config Piper ES...")
        urllib.request.urlretrieve(PIPER_CONFIG_URL, CONFIG_PATH)
        print(f"  -> {CONFIG_PATH}")


def synthesize_piper(voice, text: str) -> np.ndarray:
    """Sintetiza texto con Piper y devuelve audio float32 a TARGET_SR."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    with wave.open(tmp_path, "w") as wf:
        voice.synthesize(text, wf)
    audio, sr = sf.read(tmp_path, dtype="float32")
    os.unlink(tmp_path)
    # Resamplear a TARGET_SR si es necesario
    if sr != TARGET_SR:
        n_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, n_samples).astype(np.float32)
    return audio


def augment(audio: np.ndarray, sr: int = TARGET_SR) -> list[np.ndarray]:
    """Devuelve [original, speed_slow, speed_fast, noisy, quiet]."""
    variants = [audio.copy()]

    # Velocidad lenta (0.85×): resamplear a menos samples y pad/trim a original
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

        # Repetir palabras hasta cubrir SAMPLES_PER_CLASS
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
```

- [ ] **Step 2: Crear directorio de datos**

```bash
mkdir -p data/voice voices
```

- [ ] **Step 3: Verificar que el archivo importa sin error**

```bash
uv run python -c "import generate_voice_dataset; print('OK')"
```

Expected: `OK` (no descarga aún, solo chequea imports)

- [ ] **Step 4: Commit**

```bash
git add generate_voice_dataset.py
git commit -m "feat(voice): generate_voice_dataset.py con Piper TTS español y augmentación"
```

---

## Task 3: Crear voice_dataset.py

**Files:**
- Create: `voice_dataset.py`

PyTorch Dataset que carga WAVs desde `data/voice/` y los convierte a mel-spectrograms 64×64 con NumPy puro (sin librosa).

- [ ] **Step 1: Crear el archivo**

```python
# voice_dataset.py
"""
Dataset PyTorch para comandos de voz.

Carga WAVs desde data/voice/<CLASE>/ y devuelve tensores (1, 64, 64)
con el mel-spectrogram calculado manualmente con NumPy.
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from scipy.signal import resample
from scipy.ndimage import zoom

TARGET_SR   = 16000
N_FFT       = 512
HOP_LENGTH  = 160      # 10 ms a 16kHz
N_MELS      = 64
SPEC_SIZE   = 64       # mel-spectrogram redimensionado a 64×64
MAX_AUDIO_S = 2.0      # cortar audios largos a 2 s

DATA_VOICE_DIR = Path("data") / "voice"

VOICE_CLASSES   = ["STOP", "ADELANTE", "IZQUIERDA", "DERECHA", "GIRO_IZQ", "GIRO_DER"]
VOICE_CLASS_IDX = {c: i for i, c in enumerate(VOICE_CLASSES)}
VOICE_IDX_CLASS = {i: c for i, c in enumerate(VOICE_CLASSES)}
NUM_VOICE_CLASSES = len(VOICE_CLASSES)


# ── Mel-spectrogram (NumPy manual) ───────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Banco de filtros triangulares mel (implementación matricial NumPy)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min, f_max   = 0.0, sr / 2.0
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points     = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points      = mel_to_hz(mel_points)
    bins           = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins           = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_l, f_c + 1):
            if f_c > f_l:
                filters[m - 1, k] = (k - f_l) / (f_c - f_l)
        for k in range(f_c, f_r + 1):
            if f_r > f_c:
                filters[m - 1, k] = (f_r - k) / (f_r - f_c)
    return filters


_FILTERBANK = _mel_filterbank(TARGET_SR, N_FFT, N_MELS)


def compute_mel_spectrogram(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Convierte audio (float32, mono) a mel-spectrogram (SPEC_SIZE, SPEC_SIZE).

    Pasos matriciales:
      1. Pre-énfasis
      2. Ventaneo Hann + FFT → espectro de potencia
      3. Filtro mel (banco triangular)
      4. Compresión logarítmica
      5. Resize a SPEC_SIZE×SPEC_SIZE
      6. Normalización Z-score
    """
    # Resamplear si hace falta
    if sr != TARGET_SR:
        audio = resample(audio, int(len(audio) * TARGET_SR / sr)).astype(np.float32)

    # Recortar a MAX_AUDIO_S
    max_samples = int(TARGET_SR * MAX_AUDIO_S)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Padding mínimo para tener al menos un frame
    min_len = N_FFT + HOP_LENGTH
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    # Pre-énfasis
    audio = np.concatenate([[audio[0]], audio[1:] - 0.97 * audio[:-1]])

    # Padding simétrico
    audio = np.pad(audio, N_FFT // 2, mode="reflect")

    # Construir frames: (n_frames, N_FFT)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    indices  = (np.arange(N_FFT)[None, :] +
                HOP_LENGTH * np.arange(n_frames)[:, None])
    frames   = audio[indices]

    # Ventana Hann
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))
    frames = frames * window

    # Espectro de potencia
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2  # (n_frames, N_FFT//2+1)

    # Filtro mel: (N_MELS, N_FFT//2+1) @ (N_FFT//2+1, n_frames) → (N_MELS, n_frames)
    mel = _FILTERBANK @ power.T

    # Log
    mel = np.log(mel + 1e-8)

    # Resize a SPEC_SIZE×SPEC_SIZE
    h, w = mel.shape
    if h != SPEC_SIZE or w != SPEC_SIZE:
        mel = zoom(mel, (SPEC_SIZE / h, SPEC_SIZE / w), order=1)

    # Normalización Z-score
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    return mel.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VoiceDataset(Dataset):
    """
    Carga WAVs desde data/voice/<CLASE>/ y devuelve (tensor(1,64,64), label_int).
    """

    def __init__(self, root: Path = DATA_VOICE_DIR, augment: bool = False):
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment

        for cls_name in VOICE_CLASSES:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            label = VOICE_CLASS_IDX[cls_name]
            for wav in sorted(cls_dir.glob("*.wav")):
                self.samples.append((wav, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No se encontraron WAVs en {root}. "
                "Ejecuta primero: uv run python generate_voice_dataset.py"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        audio, sr   = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)     # estéreo → mono

        if self.augment:
            # Jitter de volumen
            audio = audio * np.random.uniform(0.8, 1.2)
            # Ruido de fondo suave
            audio = audio + np.random.normal(0, 0.003, len(audio)).astype(np.float32)
            audio = audio.clip(-1, 1)

        mel    = compute_mel_spectrogram(audio, sr)
        tensor = torch.from_numpy(mel[np.newaxis])   # (1, 64, 64)
        return tensor, label


def report_balance(root: Path = DATA_VOICE_DIR) -> None:
    for cls in VOICE_CLASSES:
        d = root / cls
        n = len(list(d.glob("*.wav"))) if d.exists() else 0
        print(f"  {cls:<12}: {n:>5} muestras")


if __name__ == "__main__":
    report_balance()
    ds = VoiceDataset()
    x, y = ds[0]
    print(f"\nEjemplo — tensor: {tuple(x.shape)}, clase: {VOICE_IDX_CLASS[y]}")
    print(f"Total muestras: {len(ds)}")
```

- [ ] **Step 2: Verificar imports (sin datos aún)**

```bash
uv run python -c "
from voice_dataset import compute_mel_spectrogram, VOICE_CLASSES
import numpy as np
audio = np.random.randn(16000).astype('float32')
mel = compute_mel_spectrogram(audio)
print(f'mel shape: {mel.shape}')
assert mel.shape == (64, 64), f'Expected (64,64) got {mel.shape}'
print('OK')
"
```

Expected:
```
mel shape: (64, 64)
OK
```

- [ ] **Step 3: Commit**

```bash
git add voice_dataset.py
git commit -m "feat(voice): voice_dataset.py con mel-spectrogram NumPy manual y VoiceDataset"
```

---

## Task 4: Crear model_voice.py

**Files:**
- Create: `model_voice.py`

VoiceCNN de 3 bloques convolucionales siguiendo el estilo de `model_nav.py`. Input: `(batch, 1, 64, 64)`.

- [ ] **Step 1: Crear el archivo**

```python
# model_voice.py
"""
CNN para clasificación de comandos de voz sobre mel-spectrograms.

Entrada: (batch, 1, 64, 64)  ← mel-spectrogram canal único
Salida:  (batch, 6)          ← logits de 6 comandos

Arquitectura (3 bloques + 3 FC):
  Conv(1→32)   → BN → ReLU → MaxPool2×2   → (32, 32, 32)
  Conv(32→64)  → BN → ReLU → MaxPool2×2   → (64, 16, 16)
  Conv(64→128) → BN → ReLU → MaxPool2×2   → (128, 8, 8)
  Flatten → 8192
  FC(8192→256) → ReLU → Dropout(0.5)
  FC(256→64)   → ReLU
  FC(64→6)     ← logits
"""

import torch
import torch.nn as nn
from voice_dataset import SPEC_SIZE, NUM_VOICE_CLASSES


class VoiceCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_VOICE_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        flat = 128 * (SPEC_SIZE // 8) * (SPEC_SIZE // 8)   # 128 * 8 * 8 = 8192

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


def build_voice_model(device: str = "cpu") -> VoiceCNN:
    return VoiceCNN().to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = build_voice_model()
    dummy = torch.zeros(1, 1, 64, 64)
    out   = model(dummy)
    print(f"VoiceCNN — Parámetros: {count_parameters(model):,}")
    print(f"Input: {tuple(dummy.shape)}  →  Output: {tuple(out.shape)}")
```

- [ ] **Step 2: Verificar arquitectura**

```bash
uv run python model_voice.py
```

Expected:
```
VoiceCNN — Parámetros: X,XXX,XXX
Input: (1, 1, 64, 64)  →  Output: (1, 6)
```

(El número exacto de parámetros no importa, lo importante es que Output sea `(1, 6)`)

- [ ] **Step 3: Commit**

```bash
git add model_voice.py
git commit -m "feat(voice): VoiceCNN (3 conv + 3 FC) para clasificación de mel-spectrograms"
```

---

## Task 5: Crear train_voice.py

**Files:**
- Create: `train_voice.py`

Entrena el VoiceCNN, imprime métricas por época y guarda el mejor checkpoint en `models/voice_model.pth`.

- [ ] **Step 1: Crear el archivo**

```python
# train_voice.py
"""
Entrenamiento del VoiceCNN para reconocimiento de comandos de voz.

Uso:
    uv run python train_voice.py --epochs 30

Requiere que data/voice/ esté generado:
    uv run python generate_voice_dataset.py
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from voice_dataset import VoiceDataset, VOICE_CLASSES
from model_voice import build_voice_model, count_parameters

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")


def train(epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
          val_split: float = 0.15) -> None:

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[train_voice] Device: {device}")

    # Dataset completo con augmentación en train
    full_ds = VoiceDataset(augment=False)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Activar augmentación solo en train
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = build_voice_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"Parámetros: {count_parameters(model):,}")
    print(f"Train: {n_train}  Val: {n_val}  Clases: {VOICE_CLASSES}\n")

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1).cpu()
            y_cpu = y.cpu()
            train_correct += (preds == y_cpu).sum().item()
            train_total   += len(y_cpu)
            train_loss    += loss.item() * len(y_cpu)

        train_acc  = train_correct / train_total
        train_loss = train_loss    / train_total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(device), y.to(device)
                preds = model(x).argmax(1).cpu()
                y_cpu = y.cpu()
                val_correct += (preds == y_cpu).sum().item()
                val_total   += len(y_cpu)

        val_acc = val_correct / val_total
        scheduler.step()

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                        "classes": VOICE_CLASSES}, MODEL_VOICE_PATH)
            flag = "  ← guardado"

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_acc:.3f}  val={val_acc:.3f}  "
              f"loss={train_loss:.4f}{flag}")

    print(f"\nMejor val_acc: {best_val_acc:.3f}")
    print(f"Modelo guardado en: {MODEL_VOICE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr)
```

- [ ] **Step 2: Verificar que importa sin error**

```bash
uv run python -c "import train_voice; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train_voice.py
git commit -m "feat(voice): train_voice.py con Adam + LR scheduler + checkpoint"
```

---

## Task 6: Crear main_voice.py

**Files:**
- Create: `main_voice.py`

Pipeline completo en tiempo real. Captura audio del micrófono, detecta voz por energía (VAD), extrae mel-spectrogram, infiere con VoiceCNN y envía el byte UDP al ESP32. Reutiliza `utils.py` para las constantes del protocolo.

- [ ] **Step 1: Crear el archivo**

```python
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
import threading
import time
import numpy as np
import sounddevice as sd
import torch

from utils import ESP32_IP, ESP32_PORT, CMD_STOP, CMD_NAME
from voice_dataset import (
    compute_mel_spectrogram, TARGET_SR,
    VOICE_CLASSES, VOICE_IDX_CLASS, NUM_VOICE_CLASSES
)
from model_voice import build_voice_model

MODEL_VOICE_PATH = os.path.join("models", "voice_model.pth")

# ── Parámetros VAD ───────────────────────────────────────────────────────────
CHUNK_DURATION_S  = 0.05    # 50 ms por chunk
SILENCE_DURATION_S = 0.35   # silencio para cerrar utterance
MAX_UTTERANCE_S    = 2.0    # máx duración de un comando

# Mapa clase → byte UDP (mismo protocolo que la cámara)
VOICE_CMD_MAP = {
    "STOP":       0x00,
    "ADELANTE":   0x01,
    "IZQUIERDA":  0x02,
    "DERECHA":    0x03,
    "GIRO_IZQ":   0x04,
    "GIRO_DER":   0x05,
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

def run(vad_threshold: float = 0.02, device_idx=None) -> None:
    model, torch_device = load_voice_model()
    sender = UDPSender()

    chunk_samples   = int(TARGET_SR * CHUNK_DURATION_S)
    silence_chunks  = int(SILENCE_DURATION_S / CHUNK_DURATION_S)
    max_chunks      = int(MAX_UTTERANCE_S / CHUNK_DURATION_S)

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
                        sender.send(cmd_byte)

                        wifi_str = "OK " if sender.wifi_ok else "ERR"
                        print(f" → {cls_name}  (UDP: 0x{cmd_byte:02X}  WiFi:{wifi_str})")

                        recording    = False
                        buffer       = []
                        silent_count = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[voice] Enviando STOP al ESP32...")
        sender.send(CMD_STOP)
        time.sleep(0.1)
        sender.close()
        print("[voice] Finalizado.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control por voz del robot")
    parser.add_argument("--threshold",    type=float, default=0.02,
                        help="Umbral de energía RMS para VAD (default: 0.02)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Mostrar micrófonos disponibles y salir")
    parser.add_argument("--device",       type=int, default=None,
                        help="Índice del micrófono (ver --list-devices)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    else:
        run(vad_threshold=args.threshold, device_idx=args.device)
```

- [ ] **Step 2: Verificar que importa sin error**

```bash
uv run python -c "import main_voice; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Listar micrófonos disponibles**

```bash
uv run python main_voice.py --list-devices
```

Expected: lista de dispositivos de audio del sistema.

- [ ] **Step 4: Commit**

```bash
git add main_voice.py
git commit -m "feat(voice): main_voice.py — pipeline completo mic→VAD→VoiceCNN→UDP"
```

---

## Task 7: Generar dataset, entrenar y verificar pipeline completo

**Files:** ninguno nuevo

- [ ] **Step 1: Generar dataset con Piper TTS**

```bash
uv run python generate_voice_dataset.py
```

Expected: descarga modelo (~65 MB, solo primera vez) y genera `data/voice/<CLASE>/` con ~240 WAVs por clase.

- [ ] **Step 2: Verificar balance del dataset**

```bash
uv run python voice_dataset.py
```

Expected:
```
  STOP        :   240 muestras
  ADELANTE    :   240 muestras
  IZQUIERDA   :   240 muestras
  DERECHA     :   240 muestras
  GIRO_IZQ    :   240 muestras
  GIRO_DER    :   240 muestras
Total muestras: 1440
```

- [ ] **Step 3: Entrenar el modelo**

```bash
uv run python train_voice.py --epochs 30
```

Expected al final: `Mejor val_acc: 0.XXX` y `Modelo guardado en: models/voice_model.pth`.
Un val_acc > 0.70 en datos sintéticos es aceptable para la demo.

- [ ] **Step 4: Smoke test del pipeline (sin ESP32)**

```bash
uv run python -c "
import numpy as np
from main_voice import load_voice_model, infer
model, device = load_voice_model()
audio = np.random.randn(16000).astype('float32')
cls = infer(model, audio, device)
print(f'Test inference OK → clase: {cls}')
"
```

Expected: imprime una clase (puede ser cualquiera — es ruido aleatorio).

- [ ] **Step 5: Commit final**

```bash
git add data/voice/ models/voice_model.pth
git commit -m "feat(voice): dataset generado y modelo voice_model.pth entrenado"
```

---

## Flujo de trabajo en la presentación

```bash
# Rama cámara (demo principal):
git checkout main
uv run python main_robot.py --camera 1 --display --scale 0.5

# Rama voz (backup si falla la cámara):
git checkout feature/voice-control
uv run python main_voice.py --threshold 0.02
# Si el micrófono no responde bien, ajustar threshold:
uv run python main_voice.py --threshold 0.015
```

---

## Notas para el día de la presentación

1. **Probar con voz real** 2-3 días antes. Si val_acc en datos reales es bajo (< 60%), agregar 10-20 grabaciones reales por clase a `data/voice/<CLASE>/` y reentrenar.
2. **Ajustar VAD_THRESHOLD** según el ruido del aula. En aulas silenciosas: 0.015. Con ruido: 0.03.
3. El ESP32 no cambia — mismo firmware, mismo puerto 9999.
4. La rama `main` (cámara) no se toca: `git stash` / `git checkout main` es suficiente para cambiar de modo.

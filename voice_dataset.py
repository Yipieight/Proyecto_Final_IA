# voice_dataset.py
"""
Dataset PyTorch para comandos de voz.

Carga WAVs desde data/voice/<CLASE>/ y devuelve tensores (1, 64, 64)
con el mel-spectrogram calculado manualmente con NumPy.
"""

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

VOICE_CLASSES     = ["ALTO", "ADELANTE", "IZQUIERDA", "DERECHA", "GIRO_IZQ", "GIRO_DER"]
VOICE_CLASS_IDX   = {c: i for i, c in enumerate(VOICE_CLASSES)}
VOICE_IDX_CLASS   = {i: c for i, c in enumerate(VOICE_CLASSES)}
NUM_VOICE_CLASSES = len(VOICE_CLASSES)


# ── Mel-spectrogram (NumPy manual) ───────────────────────────────────────────

def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Banco de filtros triangulares mel (implementación matricial NumPy)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min, f_max     = 0.0, sr / 2.0
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points       = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points        = mel_to_hz(mel_points)
    bins             = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins             = np.clip(bins, 0, n_fft // 2)

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
        self.samples: list = []
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
            audio = audio * np.random.uniform(0.8, 1.2)
            audio = audio + np.random.normal(0, 0.003, len(audio)).astype(np.float32)
            audio = audio.clip(-1, 1)

        mel    = compute_mel_spectrogram(audio, sr)
        tensor = torch.from_numpy(mel[np.newaxis])   # (1, 64, 64)
        return tensor, label


class CachedVoiceDataset(Dataset):
    """
    Precarga todos los mel-spectrograms en RAM al inicio.
    Una vez cargado, cada __getitem__ es un acceso directo a tensor — sin I/O ni cómputo.

    Uso en entrenamiento:
        ds = CachedVoiceDataset()          # ~2 min de carga, luego muy rápido
        ds.move_to_device(device)          # opcional: todo en MPS/GPU
    """

    def __init__(self, root: Path = DATA_VOICE_DIR):
        import time
        base = VoiceDataset(root, augment=False)
        n    = len(base)

        print(f"[cache] Precargando {n} muestras en RAM...")
        t0 = time.time()

        tensors = torch.zeros(n, 1, SPEC_SIZE, SPEC_SIZE, dtype=torch.float32)
        labels  = torch.zeros(n, dtype=torch.long)

        for i, (wav_path, label) in enumerate(base.samples):
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            mel = compute_mel_spectrogram(audio, sr)
            tensors[i, 0] = torch.from_numpy(mel)
            labels[i]     = label
            if (i + 1) % 8000 == 0:
                print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")

        self.tensors = tensors
        self.labels  = labels
        mem_mb = tensors.numel() * 4 / 1024 / 1024
        print(f"[cache] Listo en {time.time()-t0:.1f}s  (~{mem_mb:.0f} MB en RAM)")

    def move_to_device(self, device: torch.device) -> "CachedVoiceDataset":
        """Mueve tensores al dispositivo — elimina transferencias por batch."""
        self.tensors = self.tensors.to(device)
        self.labels  = self.labels.to(device)
        return self

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.tensors[idx], int(self.labels[idx])


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

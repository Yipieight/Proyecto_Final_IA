# extract_embeddings.py
"""
Pasa todos los WAVs del dataset de voz por VoiceCNN y extrae el embedding
de 64 dimensiones (penúltimo FC, antes de la capa de clasificación final).

Guarda un archivo .npy por clase en embeddings/<CLASE>.npy
  shape: (N_samples, 64)

Uso:
    uv run python extract_embeddings.py
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from voice_dataset import (
    VOICE_CLASSES,
    DATA_VOICE_DIR,
    compute_mel_spectrogram,
)
from model_voice import VoiceCNN

MODEL_PATH     = Path("models") / "voice_model.pth"
EMBEDDINGS_DIR = Path("embeddings")
DEVICE         = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


def load_model(device: str) -> VoiceCNN:
    model = VoiceCNN()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def extract_embedding_hook(model: VoiceCNN):
    """
    Registra un forward hook en el penúltimo ReLU del classifier
    (después de FC(256→64) + ReLU, antes de FC(64→6)).

    El classifier tiene indices:
      0: Flatten
      1: Linear(8192→256)
      2: ReLU
      3: Dropout
      4: Linear(256→64)
      5: ReLU          ← aquí extraemos (64-dim)
      6: Linear(64→6)
    """
    embeddings = {}

    def hook(module, input, output):
        embeddings["last"] = output.detach().cpu()

    handle = model.classifier[5].register_forward_hook(hook)
    return embeddings, handle


@torch.no_grad()
def extract_class(
    model: VoiceCNN,
    cls_name: str,
    embeddings_store: dict,
    hook_handle,
    device: str,
) -> np.ndarray:
    cls_dir = DATA_VOICE_DIR / cls_name
    wavs = sorted(cls_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Sin WAVs en {cls_dir}")

    vecs = []
    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        mel = compute_mel_spectrogram(audio, sr)
        x   = torch.from_numpy(mel[np.newaxis, np.newaxis]).to(device)  # (1,1,64,64)

        model(x)  # forward — el hook captura el embedding
        vecs.append(embeddings_store["last"].numpy())  # (1, 64)

    return np.vstack(vecs)  # (N, 64)


def main():
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelo: {MODEL_PATH}")

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    model = load_model(DEVICE)
    embeddings_store, handle = extract_embedding_hook(model)

    for cls_name in VOICE_CLASSES:
        print(f"  Extrayendo {cls_name}...", end=" ", flush=True)
        arr = extract_class(model, cls_name, embeddings_store, handle, DEVICE)
        out = EMBEDDINGS_DIR / f"{cls_name}.npy"
        np.save(out, arr)
        print(f"{arr.shape[0]} muestras → {out}")

    handle.remove()
    print("\nListo. Embeddings guardados en embeddings/")


if __name__ == "__main__":
    main()

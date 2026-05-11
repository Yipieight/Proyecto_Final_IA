"""
Pipeline de preprocesamiento matricial manual (NumPy puro — sin filtros de OpenCV).

Etapas por frame:
  BGR → escala de grises manual → recorte ROI → blur Gaussiano manual
      → (opcional) bordes Sobel manual → resize → normalización [0, 1]

Cumple el requisito: "al menos un filtro crítico programado manualmente
a nivel matricial usando NumPy/Eigen" (especificación del proyecto).
"""

import numpy as np
import cv2  # solo para cv2.resize (operación geométrica permitida)
from utils import IMG_WIDTH, IMG_HEIGHT, ROI_TOP_FRAC


# ── Conversión a escala de grises ─────────────────────────────────────────────

def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """
    Conversión manual RGB→gris con coeficientes de luminancia BT.601.
    No usa cv2.cvtColor ni ninguna función de alto nivel.
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    return 0.114 * b + 0.587 * g + 0.299 * r


# ── Convolución 2D vectorizada (NumPy stride tricks, sin scipy) ───────────────

def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolución 2D implementada con numpy.lib.stride_tricks.sliding_window_view.

    Complejidad: O(H·W·kH·kW), completamente vectorizada — apta para tiempo real.
    No usa scipy.ndimage ni cv2.filter2D.

    Args:
        img    : array 2D float32 (H, W)
        kernel : array 2D float32 (kH, kW)
    Returns:
        array float32 (H, W)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded  = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    windows = sliding_window_view(padded, (kh, kw))      # (H, W, kH, kW)
    return (windows * kernel).sum(axis=(-2, -1)).astype(np.float32)


# ── Filtro Gaussiano manual ───────────────────────────────────────────────────

def _build_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Construye un kernel Gaussiano 2D normalizado desde cero."""
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    g  = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k  = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Blur Gaussiano manual — convolución NumPy, sin OpenCV.
    img debe estar en [0, 1].
    """
    return _convolve2d(img, _build_gaussian_kernel(size, sigma))


# ── Detección de bordes Sobel manual ─────────────────────────────────────────

_SOBEL_X = np.array([[-1,  0,  1],
                     [-2,  0,  2],
                     [-1,  0,  1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)


def sobel_edges(img: np.ndarray) -> np.ndarray:
    """
    Magnitud del gradiente Sobel — convolución NumPy, sin OpenCV.
    Devuelve imagen normalizada en [0, 1].
    """
    gx  = _convolve2d(img, _SOBEL_X)
    gy  = _convolve2d(img, _SOBEL_Y)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    peak = mag.max()
    return mag / peak if peak > 0.0 else mag


# ── Extracción de ROI ─────────────────────────────────────────────────────────

def extract_roi(gray: np.ndarray) -> np.ndarray:
    """
    Recorta el tercio inferior del frame donde la línea de navegación es visible.
    Descarta ROI_TOP_FRAC del alto desde arriba (techo/pared sin información).
    """
    h   = gray.shape[0]
    top = int(h * ROI_TOP_FRAC)
    return gray[top:, :]


# ── Pipeline completo ─────────────────────────────────────────────────────────

def preprocess_frame(bgr: np.ndarray, use_edges: bool = False) -> np.ndarray:
    """
    Preprocesa un frame BGR crudo de OpenCV.

    Args:
        bgr       : frame BGR de forma (H, W, 3), uint8
        use_edges : si True, reemplaza el blur por magnitud Sobel

    Returns:
        array float32 de forma (IMG_HEIGHT, IMG_WIDTH), valores en [0, 1]
    """
    gray      = to_grayscale(bgr) / 255.0   # [0, 1]
    roi       = extract_roi(gray)
    blurred   = gaussian_blur(roi, size=5, sigma=1.0)

    processed = sobel_edges(blurred) if use_edges else blurred

    # cv2.resize está permitido para operaciones geométricas básicas
    resized = cv2.resize(processed, (IMG_WIDTH, IMG_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)

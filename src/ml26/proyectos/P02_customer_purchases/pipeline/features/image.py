"""
Extracción de features de imágenes de ítems.

Cada ítem tiene una imagen PNG de 128x128 en:
    datasets/customer_purchases/images/{item_img_filename}

La imagen codifica señales visuales latentes que NO aparecen como
columnas en el CSV (color de la prenda, patrón, silueta, etc.).

Extractores disponibles
-----------------------
    extract_mean_color        — color promedio RGB del primer plano (3 features)
    extract_color_histogram   — histograma normalizado por canal (24 features, 8 bins × 3)

Cómo agregar un nuevo extractor
--------------------------------
1. Implementa extract_X(df) -> pd.DataFrame que reciba un DataFrame con
   item_img_filename y devuelva columnas nuevas alineadas por índice.
2. Añade la función a la lista `extractors` en extract_image_features.
3. Registra las columnas nuevas en numeric_features de preprocessing.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_DIR,
)

IMG_DIR = Path(os.path.abspath(DATA_DIR)) / "images"

_HIST_BINS = 8


def _load_image(filename: str) -> np.ndarray:
    """Carga una imagen del directorio de imagenes como array (128, 128, 3)."""
    return np.array(Image.open(IMG_DIR / filename).convert("RGB"))


def _foreground_pixels(arr: np.ndarray) -> np.ndarray:
    """Devuelve los píxeles que no son fondo blanco (todos los canales > 250)."""
    is_bg = (arr[:, :, 0] > 250) & (arr[:, :, 1] > 250) & (arr[:, :, 2] > 250)
    fg = arr[~is_bg]
    return fg if len(fg) > 0 else arr.reshape(-1, 3)


def extract_mean_color(df: pd.DataFrame) -> pd.DataFrame:
    """Color promedio por canal RGB ignorando el fondo blanco.

    Parameters
    ----------
    df : DataFrame con columna item_img_filename.

    Returns
    -------
    pd.DataFrame con columnas img_mean_r, img_mean_g, img_mean_b.
    """
    records = []
    for filename in df["item_img_filename"]:
        arr = _load_image(filename)
        fg = _foreground_pixels(arr)
        mean = fg.mean(axis=0)
        records.append(
            {
                "img_mean_r": float(mean[0]),
                "img_mean_g": float(mean[1]),
                "img_mean_b": float(mean[2]),
            }
        )
    return pd.DataFrame(records, index=df.index)


def extract_color_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Histograma de color normalizado por canal RGB ignorando el fondo blanco.

    Captura la distribución de colores (no solo el promedio): una prenda
    mayormente oscura con detalles brillantes tiene un histograma muy distinto
    al de una prenda de color uniforme, aunque su color promedio sea similar.

    Genera _HIST_BINS columnas por canal → 3 × _HIST_BINS features en total.
    Las columnas se nombran img_hist_{r|g|b}_{0..7}.

    Parameters
    ----------
    df : DataFrame con columna item_img_filename.

    Returns
    -------
    pd.DataFrame con 3 × _HIST_BINS columnas de histograma normalizado.
    """
    records = []
    for filename in df["item_img_filename"]:
        arr = _load_image(filename)
        fg = _foreground_pixels(arr)
        row = {}
        for i, channel in enumerate(["r", "g", "b"]):
            hist, _ = np.histogram(fg[:, i], bins=_HIST_BINS, range=(0, 256))
            hist_norm = hist / (hist.sum() + 1e-9)
            for j, val in enumerate(hist_norm):
                row[f"img_hist_{channel}_{j}"] = float(val)
        records.append(row)
    return pd.DataFrame(records, index=df.index)


# Nombres de columnas para registrar en preprocessing.py
COLOR_HIST_COLS = [f"img_hist_{c}_{i}" for c in ["r", "g", "b"] for i in range(_HIST_BINS)]


def extract_image_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae features visuales por ítem único y devuelve un DataFrame listo para merge.

    Las imágenes se cargan UNA sola vez por ítem (deduplicando por item_id).
    El resultado incluye item_id como clave para hacer merge en orchestration.py.

    Si la carpeta de imágenes no existe, devuelve un DataFrame solo con item_id
    (sin columnas de imagen). El pipeline continúa sin features visuales:
    preprocessing.py las omite automáticamente al filtrar por columnas presentes.

    Parameters
    ----------
    df : DataFrame con columnas item_id e item_img_filename.

    Returns
    -------
    pd.DataFrame con item_id + columnas de imagen (una fila por ítem único).
    """
    if not IMG_DIR.exists():
        print(f"[image.py] AVISO: carpeta de imágenes no encontrada ({IMG_DIR}). "
              "Entrenando sin features visuales.")
        unique_ids = df["item_id"].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame({"item_id": unique_ids})

    # Extractores desactivados — agregar aquí si se dispone de las imágenes:
    #   extract_mean_color,       # img_mean_r, img_mean_g, img_mean_b
    #   extract_color_histogram,  # img_hist_{r,g,b}_{0..7}
    extractors = []

    unique_items = (
        df[["item_id", "item_img_filename"]]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    parts = [fn(unique_items) for fn in extractors]
    img_features = pd.concat(parts, axis=1)
    img_features.insert(0, "item_id", unique_items["item_id"].values)
    return img_features

"""
Capa 2 — Preprocesamiento.

Modifica preprocess() para:
  - Crear features derivadas (e.g. dias desde lanzamiento, match de categoria).
  - Decidir que columnas escalar, codificar o vectorizar.
  - Descartar columnas no disponibles en test (purchase_timestamp, etc.).

No modifiques build_processor — solo define las listas de columnas y las
features derivadas en preprocess().
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def build_processor(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    count_features: list,
    passthrough_features: list = [],
    training: bool = True,
) -> pd.DataFrame:
    """Ajusta o carga un ColumnTransformer y retorna el DataFrame transformado.

    Cuando training=True ajusta el preprocessor y lo guarda en disco.
    Cuando training=False lo carga y aplica — nunca ajustar sobre test.

    Parameters
    ----------
    df                    : DataFrame de entrada (sin label ni IDs).
    numeric_features      : columnas numericas -> StandardScaler.
    categorical_features  : columnas categoricas -> OneHotEncoder.
    count_features        : columnas de texto libre -> CountVectorizer.
    passthrough_features  : columnas que pasan sin transformar.
    training              : True = fit + save; False = load + transform.
    """
    savepath = Path(os.path.abspath(DATA_DIR)) / "preprocessor.pkl"

    if training:
        text_transformers = [(col, CountVectorizer(), col) for col in count_features]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                *text_transformers,
                ("passthrough", "passthrough", passthrough_features),
            ],
            remainder="drop",
        )
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    # CountVectorizer devuelve sparse — convertir a denso para DataFrame uniforme
    if hasattr(processed_array, "toarray"):
        processed_array = processed_array.toarray()

    num_cols = numeric_features
    cat_cols = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )
    )
    # Bug fix: variable era 'text_features', debe ser 'count_features'
    bow_cols = []
    for col in count_features:
        vocab = preprocessor.named_transformers_[col].get_feature_names_out()
        bow_cols.extend([f"{col}_bow_{t}" for t in vocab])

    return pd.DataFrame(
        processed_array,
        columns=list(num_cols) + cat_cols + bow_cols + passthrough_features,
    )


def preprocess(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Define que features usar y como codificarlas.

    Features implementadas
    ----------------------
    Numéricas (StandardScaler):
        Del cliente  : age_years, tenure_months, avg_price, std_price,
                       total_purchases, avg_rating_given, days_since_last
        Del ítem     : days_since_release, price
        Derivadas    : price_vs_customer_avg (precio relativo al perfil del cliente)
                       top_{1,2,3}_match (¿el ítem está en el top-k del cliente?)
        De imagen    : mean_r/g/b, histograma RGB normalizado (8 bins × 3 canales)

    Categóricas (OneHotEncoder):
        customer_preferred_device, item_category, customer_gender

    Passthrough (sin transformar):
        item_days_since_release_cutoff — usado por split_by_days en training.py

    Parameters
    ----------
    df       : DataFrame con todas las features post-merge.
    training : True cuando se llama desde read_train_data (ajusta el
               preprocessor). False cuando se llama desde read_test_data
               (aplica el preprocessor guardado).
    """
    df = df.copy()

    # Columnas a descartar antes del ColumnTransformer
    drop_raw = [
        "purchase_id",
        "item_release_date",        # convertida a numérica abajo
        "item_img_filename",        # reemplazada por features de imagen
        # Columnas no disponibles en test (data leakage si se usan):
        "purchase_timestamp",
        "customer_item_views",
        "purchase_item_rating",
        "purchase_device",
        "item_avg_rating",
        "item_num_ratings",
    ]

    # ── Parseo de fechas ──────────────────────────────────────────────────────
    df["item_release_date"] = pd.to_datetime(df["item_release_date"], format="mixed", dayfirst=True)

    # ── Features derivadas ────────────────────────────────────────────────────

    # item_days_since_release_cutoff: NO escalar — usado por split_by_days.
    df["item_days_since_release_cutoff"] = (
        pd.to_datetime(DATA_COLLECTED_AT) - df["item_release_date"]
    ).dt.days

    # item_days_since_release: copia que sí entra al modelo (se escala).
    # En test será negativo (ítems lanzados después del cutoff) — eso es señal válida.
    df["item_days_since_release"] = df["item_days_since_release_cutoff"]

    # Precio del ítem relativo al gasto promedio del cliente.
    # Captura si el ítem es caro o barato para este cliente específico.
    df["item_price_vs_customer_avg"] = df["item_price"] - df["customer_avg_price"]

    # Match entre la categoría del ítem y las top-k categorías del cliente.
    # Señal directa de afinidad de preferencia.
    for i in range(1, 4):
        col = f"customer_top_{i}_cat"
        if col in df.columns:
            df[f"customer_top_{i}_match"] = (
                df[col] == df["item_category"]
            ).astype(int)

    # Imputar NaN en categóricas antes del OneHotEncoder
    for col in ["customer_gender", "customer_preferred_device"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # ── Definición de grupos de features ─────────────────────────────────────

    # Features numéricas → StandardScaler
    numeric_features = [
        # Demográficas del cliente
        "customer_age_years",
        "customer_tenure_months",
        # Comportamiento de compra
        "customer_avg_price",
        "customer_std_price",
        "customer_total_purchases",
        "customer_avg_rating_given",
        "customer_days_since_last",
        # Del ítem
        "item_days_since_release",
        "item_price",
        "item_price_vs_customer_avg",
        # Afinidad cliente-ítem
        "customer_top_1_match",
        "customer_top_2_match",
        "customer_top_3_match",
        # Features de imagen — desactivadas (sin carpeta de imágenes disponible)
        # "img_mean_r", "img_mean_g", "img_mean_b",
        # *COLOR_HIST_COLS,
    ]

    # Features categóricas → OneHotEncoder
    categorical_features = [
        "customer_preferred_device",   # mobile / desktop / unknown
        "item_category",               # t-shirt, dress, shoes, etc.
        "customer_gender",             # male / female / unknown
    ]

    # Features de texto libre → CountVectorizer
    count_features = [
        # "item_title",  # descomenta si quieres bag-of-words del título
    ]

    # Passthrough: item_days_since_release_cutoff es necesario para split_by_days;
    # se dropea antes de model.fit() en training.py y en read_test_data.
    passthrough_features = ["item_days_since_release_cutoff"]

    # ── Eliminar columnas que no entran al transformer ────────────────────────
    id_cols = ["customer_id", "item_id", "label"]
    cols_to_drop = set(drop_raw) | set(id_cols)
    # También dropear las columnas top_k_cat (reemplazadas por los _match flags)
    top_cat_cols = [f"customer_top_{i}_cat" for i in range(1, 4)]
    cols_to_drop |= set(top_cat_cols)

    df_input = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Solo mantener columnas que existan en df_input (por si algún feature falló)
    numeric_features = [c for c in numeric_features if c in df_input.columns]
    categorical_features = [c for c in categorical_features if c in df_input.columns]
    passthrough_features = [c for c in passthrough_features if c in df_input.columns]

    # ── Aplicar ColumnTransformer ─────────────────────────────────────────────
    return build_processor(
        df_input,
        numeric_features,
        categorical_features,
        count_features,
        passthrough_features,
        training,
    )

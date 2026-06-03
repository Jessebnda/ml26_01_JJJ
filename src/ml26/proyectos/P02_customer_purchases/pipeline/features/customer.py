"""
Ingeniería de features por cliente.

Modifica extract_customer_features para agregar las estadísticas que
quieras calcular por cliente. 
"""

import os

import numpy as np
import pandas as pd

from ml26.proyectos.P02_customer_purchases.pipeline.io import (
    DATA_COLLECTED_AT,
    DATA_DIR,
)


def _nth_top_category(series: pd.Series, n: int) -> str:
    """Devuelve la n-ésima categoría más comprada (0-indexed). 'none' si no existe."""
    vc = series.value_counts()
    return vc.index[n] if len(vc) > n else "none"


def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features agregadas por cliente a partir del historial de compras.

    Esta función se llama UNA SOLA VEZ sobre los datos de entrenamiento.
    El resultado se guarda en customer_features.csv y se reutiliza en test
    porque el conjunto de test no tiene historial de compra para agregar.

    Features generadas
    ------------------
    Demográficas:
        customer_age_years            — edad del cliente en años
        customer_tenure_months        — meses desde registro en la plataforma

    Comportamiento de compra:
        customer_avg_price            — precio promedio de sus compras
        customer_std_price            — desviación estándar de precios (0 si solo 1 compra)
        customer_total_purchases      — número total de compras (indicador de actividad)
        customer_preferred_device     — dispositivo más usado (mobile / desktop)
        customer_days_since_last      — días desde la última compra (recencia)

    Preferencias de categoría:
        customer_top_1_cat            — categoría más comprada
        customer_top_2_cat            — segunda categoría más comprada ('none' si no aplica)
        customer_top_3_cat            — tercera categoría más comprada ('none' si no aplica)

    Satisfacción:
        customer_avg_rating_given     — calificación promedio que el cliente dio a sus compras

    Parameters
    ----------
    train_df : DataFrame completo de compras de entrenamiento (solo positivos).

    Returns
    -------
    pd.DataFrame con una fila por customer_id.
    """
    df = train_df.copy()
    df["item_release_date"] = pd.to_datetime(df["item_release_date"], format="mixed", dayfirst=True)
    df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], format="mixed", dayfirst=True)
    df["customer_date_of_birth"] = pd.to_datetime(df["customer_date_of_birth"], format="mixed", dayfirst=True)
    df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"], format="mixed", dayfirst=True)

    group = df.groupby("customer_id")
    today_ts = pd.to_datetime(DATA_COLLECTED_AT)

    # ── Demográficas ──────────────────────────────────────────────────────────
    customer_age_years = (
        today_ts - group["customer_date_of_birth"].first()
    ).dt.days // 365

    customer_tenure_months = (
        today_ts - group["customer_signup_date"].first()
    ).dt.days // 30

    # ── Comportamiento de compra ──────────────────────────────────────────────
    customer_avg_price = group["item_price"].mean()
    customer_std_price = group["item_price"].std().fillna(0)

    customer_total_purchases = group["item_id"].count()

    # Dispositivo preferido: moda de purchase_device (ignorar NaN)
    customer_preferred_device = group["purchase_device"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 else "unknown"
    )

    # Días desde la última compra (recencia)
    customer_days_since_last = (
        today_ts - group["purchase_timestamp"].max()
    ).dt.days

    # ── Preferencias de categoría ─────────────────────────────────────────────
    # Usamos value_counts().index para obtener las top-k categorías
    customer_top_1_cat = group["item_category"].agg(lambda x: _nth_top_category(x, 0))
    customer_top_2_cat = group["item_category"].agg(lambda x: _nth_top_category(x, 1))
    customer_top_3_cat = group["item_category"].agg(lambda x: _nth_top_category(x, 2))

    # ── Satisfacción ─────────────────────────────────────────────────────────
    global_avg_rating = df["purchase_item_rating"].mean()
    customer_avg_rating_given = group["purchase_item_rating"].mean().fillna(global_avg_rating)

    # ── Construir DataFrame final ─────────────────────────────────────────────
    customer_feat = pd.concat(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_age_years,
            "customer_tenure_months": customer_tenure_months,
            "customer_avg_price": customer_avg_price,
            "customer_std_price": customer_std_price,
            "customer_total_purchases": customer_total_purchases,
            "customer_preferred_device": customer_preferred_device,
            "customer_days_since_last": customer_days_since_last,
            "customer_top_1_cat": customer_top_1_cat,
            "customer_top_2_cat": customer_top_2_cat,
            "customer_top_3_cat": customer_top_3_cat,
            "customer_avg_rating_given": customer_avg_rating_given,
        },
        axis=1,
    ).reset_index(drop=True)

    # Persistir — read_test_data() carga este archivo en lugar de recomputar
    save_path = os.path.abspath(os.path.join(DATA_DIR, "customer_features.csv"))
    customer_feat.to_csv(save_path, index=False)
    print(f"Customer features saved -> {save_path}")
    return customer_feat

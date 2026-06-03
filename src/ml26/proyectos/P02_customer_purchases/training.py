"""
Entrenamiento y validación de modelos de predicción de compras.

Flujo
-----
1. Leer y preprocesar datos via read_train_data().
2. Separar train/val usando split_by_days (temporal split cold-start).
3. Entrenar el modelo con SMOTE.
4. Evaluar en validación: precision, recall, F1, AUC-ROC.
5. Guardar modelo y reporte JSON.
6. Registrar experimento en WandB (si está instalado).

WandB
-----
Cada llamada a run_training() crea un run en el proyecto
"customer_purchases_prediction". Los runs quedan registrados con:
  - config  : tipo de modelo e hiperparámetros
  - métricas: precision, recall, F1, accuracy, AUC-ROC en validación

Para usar WandB: `pip install wandb && wandb login`
Si wandb no está instalado o no hay login, el entrenamiento continúa
sin logging remoto (el logger local sigue funcionando).

Evaluación temporal
-------------------
split_by_days usa item_days_since_release_cutoff para poner en validación
los ítems lanzados recientemente (≤ cutoff_days antes del 14-05-2026).
Esto imita la distribución del test real (ítems lanzados DESPUÉS del cutoff)
mejor que un split aleatorio, ya que el modelo nunca ve información de ítems
nuevos durante el entrenamiento.
"""

import json
from sklearn.metrics import classification_report, roc_auc_score

# WandB es opcional — si no está instalado el entrenamiento continúa igual
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Custom
from ml26.proyectos.P02_customer_purchases.model import PurchaseModel
from ml26.proyectos.P02_customer_purchases.utils import setup_logger
from ml26.proyectos.P02_customer_purchases.pipeline import read_train_data

WANDB_PROJECT = "customer_purchases_prediction"


def split_by_days(X, y, cutoff_days=60):
    """Separa train y validación usando item_days_since_release_cutoff.

    Las filas con ítems lanzados hace <= cutoff_days van a validación;
    el resto va a entrenamiento. Esto imita la distribución cold-start
    del test (ítems nuevos) mejor que un split aleatorio.

    Parameters
    ----------
    X            : pd.DataFrame con columna item_days_since_release_cutoff.
    y            : pd.Series de etiquetas alineada con X.
    cutoff_days  : ítems lanzados hace <= este número de días van a val.

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    if "item_days_since_release_cutoff" not in X.columns:
        raise ValueError("X must contain an 'item_days_since_release_cutoff' column")

    X = X.copy()

    # Split into train/val usando el valor crudo (no escalado)
    val_mask = X["item_days_since_release_cutoff"] <= cutoff_days
    X = X.drop(columns=["item_days_since_release_cutoff"])
    X_val, y_val = X[val_mask], y[val_mask]
    X_train, y_train = X[~val_mask], y[~val_mask]

    # Shuffle training set
    train_idx = X_train.sample(frac=1, random_state=42).index
    X_train, y_train = X_train.loc[train_idx], y_train.loc[train_idx]

    return X_train, X_val, y_train, y_val


def run_training(X, y, classifier: str):
    """Entrena un modelo, evalúa en validación y registra el experimento.

    Parameters
    ----------
    X          : features preprocesadas (incluye item_days_since_release_cutoff).
    y          : etiquetas (0 / 1).
    classifier : tipo de clasificador ("logistic", "rf", "xgb").

    Returns
    -------
    model    : PurchaseModel entrenado.
    run_dir  : Path al directorio del run (modelo + reporte + log).
    """
    model = PurchaseModel(classifier=classifier)

    # Logger escribe directamente en la carpeta del run
    logger = setup_logger(model.name, log_dir=model.run_dir)
    logger.info(f"Run: {model.name}")
    logger.info(f"Model parameters: {model.get_config()}")

    # ── WandB: iniciar run ────────────────────────────────────────────────────
    wandb_run = None
    if _WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=model.name,
                config=model.get_config(),
                reinit=True,
            )
            logger.info(f"WandB run iniciado: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"WandB init falló ({e}). Continuando sin logging remoto.")
            wandb_run = None

    # ── Split temporal ────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = split_by_days(X, y, cutoff_days=60)
    logger.info(f"Split dataset: {len(X_train)} train / {len(X_val)} val")
    logger.info(f"Positivos en val: {y_val.sum()} / {len(y_val)}")

    if wandb_run:
        wandb.config.update({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "val_positive_rate": float(y_val.mean()),
        })

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    logger.info(f"Iniciando entrenamiento ({classifier})...")
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado")

    # ── Evaluación en validación ──────────────────────────────────────────────
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_val, y_proba)

    logger.info(f"Validation metrics: {report}")
    logger.info(f"AUC-ROC: {auc_score:.4f}")

    # ── WandB: registrar métricas ─────────────────────────────────────────────
    if wandb_run:
        try:
            wandb.log({
                "val/precision":  report["weighted avg"]["precision"],
                "val/recall":     report["weighted avg"]["recall"],
                "val/f1":         report["weighted avg"]["f1-score"],
                "val/accuracy":   report["accuracy"],
                "val/auc_roc":    auc_score,
                # Métricas de la clase positiva (compra = 1)
                "val/precision_pos": report.get("1", {}).get("precision", 0),
                "val/recall_pos":    report.get("1", {}).get("recall", 0),
                "val/f1_pos":        report.get("1", {}).get("f1-score", 0),
            })
            wandb.finish()
        except Exception as e:
            logger.warning(f"WandB log falló: {e}")

    # ── Guardar modelo y reporte ──────────────────────────────────────────────
    model.save()
    logger.info(f"Modelo guardado en {model.run_dir}")

    report["auc_roc"] = auc_score
    report_path = model.run_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Reporte guardado en {report_path}")

    return model, model.run_dir


if __name__ == "__main__":
    X, y = read_train_data()
    # Entrenar los tres modelos en secuencia para comparar en WandB
    models_to_train = ["logistic", "rf", "xgb"]
    for clf in models_to_train:
        print(f"\n{'='*50}")
        print(f"Entrenando: {clf}")
        print(f"{'='*50}")
        trained_model, run_dir = run_training(X, y, clf)
        print(f"Run guardado en: {run_dir}")

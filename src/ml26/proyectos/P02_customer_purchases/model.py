# Data management
from datetime import datetime
from pathlib import Path
import joblib

# ML
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

CURRENT_FILE = Path(__file__).resolve()
MODELS_DIR = CURRENT_FILE.parent / "trained_models"

MODELS_DIR.mkdir(exist_ok=True, parents=True)


class PurchaseModel:
    """

    Soporta tres tipos de clasificador:
        "logistic"  — Regresión Logística 
        "rf"        — Random Forest 
        "xgb"       — XGBoost 


    Decisión de diseño — SMOTE vs class_weight
    -------------------------------------------
    Usamos SMOTE (oversampling sintético) en lugar de class_weight="balanced"
    porque queremos que el modelo aprenda ejemplos positivos sintéticos en el
    espacio de features, no solo reponderar los existentes. Esto funciona bien
    cuando el espacio positivo es denso y las features son continuas.
    Para producción se recomienda evaluar ambas opciones.
    """

    def __init__(self, classifier="logistic", solver="lbfgs", max_iter=1000):
        self.model_type = classifier.lower()
        # Logistic Regression params 
        self.solver = solver
        self.max_iter = max_iter

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{self.model_type}_{timestamp}"
        self.run_dir = MODELS_DIR / self.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = self.get_classifier(self.model_type)
        self.smote = SMOTE(random_state=42)

        self.model = Pipeline(
            [
                ("classifier", self.classifier),
            ]
        )

    def get_classifier(self, name: str):
        """Instancia el clasificador según el nombre.

        Parámetros por defecto seleccionados para ser razonables sin tuning:
            logistic  — solver lbfgs es estable; max_iter alto para convergencia.
            rf        — 200 árboles, max_depth=10 para limitar overfitting.
            xgb       — learning_rate bajo + subsampling para generalización.
        """
        if name == "logistic":
            return LogisticRegression(
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=42,
            )
        elif name in ("rf", "randomforest", "random_forest"):
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        elif name == "xgb":
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            raise ValueError(
                f"Clasificador desconocido: '{name}'. Opciones: logistic, rf, xgb"
            )

    def fit(self, X, y):
        """Entrena aplicando SMOTE + clasificador.

        SMOTE genera ejemplos positivos sintéticos interpolando en el espacio
        de features, balanceando el dataset antes de ajustar el clasificador.
        """
        X_res, y_res = self.smote.fit_resample(X, y)
        self.model.fit(X_res, y_res)
        return self

    def predict(self, X):
        """Predice etiquetas binarias (0 / 1)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Devuelve probabilidades [P(class=0), P(class=1)] por ejemplo."""
        return self.model.predict_proba(X)

    def get_config(self) -> dict:
        """Devuelve hiperparámetros del clasificador para logging (WandB, JSON)."""
        config = {"model": self.model_type}
        clf = self.classifier
        # Extraer parámetros relevantes dinámicamente según el tipo
        for attr in [
            "solver", "max_iter",           # Logistic Regression
            "n_estimators", "max_depth",    # RF / XGBoost
            "min_samples_leaf",             # RF
            "learning_rate", "subsample", "colsample_bytree",  # XGBoost
        ]:
            val = getattr(clf, attr, None)
            if val is not None:
                config[attr] = val
        return config

    def save(self):
        """Guarda el modelo como model.pkl en self.run_dir."""
        filepath = self.run_dir / "model.pkl"
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        return filepath

    def load(self, filename: str):
        filepath = Path(MODELS_DIR) / filename
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model

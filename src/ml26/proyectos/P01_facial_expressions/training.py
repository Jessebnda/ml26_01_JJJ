import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ml26.proyectos.P01_facial_expressions.dataset import get_loader
from ml26.proyectos.P01_facial_expressions.network import Network

# Logging
import wandb
from datetime import datetime, timezone

# Numero de muestras por clase en el dataset de entrenamiento (orden segun EMOTIONS_MAP)
# angry=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6
SAMPLES_PER_CLASS = torch.tensor([3995, 436, 4097, 7215, 4830, 3171, 4965], dtype=torch.float32)


def get_class_weights(samples_per_class: torch.Tensor) -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Las clases con menos muestras reciben mayor peso en el loss.
    """
    total = samples_per_class.sum()
    n_classes = len(samples_per_class)
    weights = total / (n_classes * samples_per_class)
    return weights


def init_wandb(cfg):
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")
    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    """
    Realiza un epoch completo en el conjunto de validación.
    args:
    - val_loader (torch.DataLoader): dataloader para los datos de validación
    - net: instancia de red neuronal de clase Network
    - cost_function (torch.nn): Función de costo a utilizar
    returns:
    - val_loss (float): costo promedio por minibatch
    - val_acc (float): accuracy en validación
    """
    val_loss = 0.0
    correct = 0
    total = 0

    net_device = net.device
    for batch in val_loader:
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"].to(net_device)

        with torch.inference_mode():
            logits = net.forward(batch_imgs)          
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()

            # Accuracy: clase predicha = indice con mayor probabilidad
            predictions = logits.argmax(dim=1)        
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    val_acc = correct / total
    return val_loss / len(val_loader), val_acc


def train():
    cfg = {
        "training": {
            "learning_rate": 2e-4,
            "n_epochs": 100,
            "batch_size": 128,
            "weight_decay": 1e-4,       
            "early_stop_patience": 15,  
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg["training"]
    learning_rate       = train_cfg["learning_rate"]
    n_epochs            = train_cfg["n_epochs"]
    batch_size          = train_cfg["batch_size"]
    weight_decay        = train_cfg["weight_decay"]
    early_stop_patience = train_cfg["early_stop_patience"]

    # Datasets y loaders
    train_dataset, train_loader = get_loader("train", batch_size=batch_size, shuffle=True)
    val_dataset,   val_loader   = get_loader("val",   batch_size=batch_size, shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    # Red neuronal
    modelo = Network(input_dim=48, n_classes=7)

    # Funcion de costo con pesos por clase para manejar el desbalance
    class_weights = get_class_weights(SAMPLES_PER_CLASS).to(modelo.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)

    # Optimizador Adam con regularizacion L2
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
    )

    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # ── Entrenamiento ──────────────────────────────────────────────────────
        modelo.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch_imgs   = batch["transformed"]
            batch_labels = batch["label"].to(modelo.device)

            optimizer.zero_grad()
            logits = modelo.forward(batch_imgs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total   += batch_labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc  = correct / total

        # ── Validacion ─────────────────────────────────────────────────────────
        modelo.eval()
        val_loss, val_acc = validation_step(val_loader, modelo, criterion)

        tqdm.write(
            f"Epoch {epoch:3d} | "
            f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.3f} | "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f} | "
        )

        scheduler.step()

        # ── Guardar mejor modelo ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            modelo.save_model("best_model.pth")
            tqdm.write(f"  -> Nuevo mejor modelo guardado (val_loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        # ── Early stopping ─────────────────────────────────────────────────────
        if epochs_without_improvement >= early_stop_patience:
            tqdm.write(f"Early stopping: {early_stop_patience} epochs sin mejora.")
            break

        run.log({
            "epoch":      epoch,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
        })

    run.finish()


if __name__ == "__main__":
    train()

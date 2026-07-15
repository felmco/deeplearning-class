"""El training loop del curso, escrito una vez y bien.

Qué es: la implementación de referencia del ciclo de entrenamiento
y evaluación que se usa en TODOS los laboratorios.
Qué hace: ejecuta epochs de train/eval con la secuencia canónica
    forward → loss → zero_grad → backward → step
más early stopping con restauración del mejor checkpoint.

Este archivo es la respuesta a la pregunta de la Sesión 1:
"¿qué pasa exactamente dentro de model.fit()?" — no hay magia.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class MetricasEpoch:
    """Resultado de una pasada completa por un DataLoader."""

    loss: float
    accuracy: float


@dataclass
class Historia:
    """Acumula las curvas de aprendizaje para graficarlas después."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    binaria: bool = False,
) -> MetricasEpoch:
    """Una pasada completa (train si hay optimizer, eval si no).

    Decisiones que importan y sus porqués:

    - `model.train(training)`: activa/desactiva dropout y pone
      BatchNorm en modo batch/estadísticas acumuladas. Olvidarlo es el
      bug silencioso más común del curso.
    - `torch.inference_mode()` en evaluación: no construye el grafo
      computacional → menos memoria y más velocidad.
    - `optimizer.zero_grad(set_to_none=True)`: PyTorch ACUMULA
      gradientes por diseño; hay que limpiarlos en cada iteración.
    - `loss.item() * batch_size`: la loss viene promediada por batch;
      se re-pondera para que el promedio final sea por MUESTRA.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correctas = 0
    total_muestras = 0

    contexto = torch.enable_grad() if training else torch.inference_mode()
    with contexto:
        for batch in loader:
            # Soporta batches (x, y) y (x, mask, y) — mover TODO al device
            batch = [t.to(device) for t in batch]
            *entradas, labels = batch

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(*entradas)          # 1. forward
            loss = criterion(logits, labels)   # 2. loss

            if training:
                loss.backward()                # 3. backward: llena .grad
                if grad_clip is not None:
                    # Limitar la norma del gradiente: protege contra
                    # exploding gradients (útil en RNNs y LRs altos)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()               # 4. update: θ ← θ − η·∇L

            # ── métricas del batch ──
            n = labels.size(0)
            total_loss += loss.item() * n
            if binaria:
                # Clasificación binaria: un logit → sigmoid → umbral 0.5
                predicciones = (torch.sigmoid(logits) >= 0.5).long()
                total_correctas += (predicciones == labels.long()).sum().item()
            else:
                # Multiclase: el logit más alto gana
                total_correctas += (logits.argmax(dim=1) == labels).sum().item()
            total_muestras += n

    return MetricasEpoch(
        loss=total_loss / total_muestras,
        accuracy=total_correctas / total_muestras,
    )


def entrenar(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 15,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float | None = None,
    binaria: bool = False,
    verbose_cada: int = 10,
) -> tuple[nn.Module, Historia]:
    """Entrenamiento completo con early stopping.

    Early stopping: si la validation loss no mejora durante `patience`
    epochs, se detiene el entrenamiento y se RESTAURA el mejor
    checkpoint. Entrenar más allá de ese punto solo memoriza ruido
    (overfitting).

    Devuelve (modelo con los mejores pesos, historia de curvas).
    """
    model.to(device)
    historia = Historia()

    mejor_val_loss = float("inf")
    mejor_estado = copy.deepcopy(model.state_dict())
    epochs_sin_mejora = 0

    for epoch in range(1, max_epochs + 1):
        metricas_train = run_epoch(model, train_loader, criterion, device,
                                   optimizer, grad_clip, binaria)
        metricas_val = run_epoch(model, val_loader, criterion, device,
                                 binaria=binaria)
        if scheduler is not None:
            scheduler.step()

        historia.train_loss.append(metricas_train.loss)
        historia.val_loss.append(metricas_val.loss)
        historia.train_acc.append(metricas_train.accuracy)
        historia.val_acc.append(metricas_val.accuracy)

        # ¿Mejoró la validation loss? (margen mínimo anti-ruido)
        if metricas_val.loss < mejor_val_loss - 1e-4:
            mejor_val_loss = metricas_val.loss
            mejor_estado = copy.deepcopy(model.state_dict())
            epochs_sin_mejora = 0
        else:
            epochs_sin_mejora += 1

        if verbose_cada and epoch % verbose_cada == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {metricas_train.loss:.4f} acc {metricas_train.accuracy:.3f} | "
                f"val loss {metricas_val.loss:.4f} acc {metricas_val.accuracy:.3f}"
            )

        if epochs_sin_mejora >= patience:
            print(f"Early stopping en epoch {epoch} "
                  f"(sin mejora en {patience} epochs)")
            break

    # Restaurar el mejor modelo visto (¡no el último!)
    model.load_state_dict(mejor_estado)
    return model, historia

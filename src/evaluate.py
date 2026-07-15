"""Evaluación y análisis de errores.

Qué es: las herramientas de la fase que separa un experimento serio
de una demo: métricas correctas, matriz de confusión y la inspección
de los errores de mayor confianza.
Qué hace: predice sobre un DataLoader, calcula métricas por clase y
encuentra los ejemplos donde el modelo se equivoca CON convicción
(los más informativos para diagnosticar).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def predecir(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    binaria: bool = False,
) -> tuple[list[int], list[int], list[float]]:
    """Recorre el loader en modo evaluación y devuelve
    (labels_reales, predicciones, confianzas).

    Reglas de higiene de evaluación:
    - `model.eval()`: desactiva dropout y usa estadísticas acumuladas
      de BatchNorm.
    - `torch.inference_mode()`: sin grafo, sin gradientes.
    - La confianza es la probabilidad de la clase PREDICHA (no de la
      real): mide cuán "seguro" estaba el modelo, correcto o no.
    """
    model.eval()
    y_real: list[int] = []
    y_pred: list[int] = []
    confianzas: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            batch = [t.to(device) for t in batch]
            *entradas, labels = batch
            logits = model(*entradas)

            if binaria:
                probabilidad = torch.sigmoid(logits).squeeze(1)
                predicciones = (probabilidad >= 0.5).long()
                confianza = torch.where(predicciones == 1,
                                        probabilidad, 1 - probabilidad)
            else:
                probabilidades = logits.softmax(dim=1)
                confianza, predicciones = probabilidades.max(dim=1)

            y_real.extend(labels.long().cpu().flatten().tolist())
            y_pred.extend(predicciones.cpu().flatten().tolist())
            confianzas.extend(confianza.cpu().flatten().tolist())

    return y_real, y_pred, confianzas


def reporte_completo(
    y_real: list[int], y_pred: list[int], nombres_clases: list[str] | None = None
) -> None:
    """Imprime classification report + matriz de confusión.

    Por qué no solo accuracy: con clases desbalanceadas, un modelo que
    predice siempre la clase mayoritaria puede tener accuracy alta y
    ser inútil. Macro-F1 promedia el F1 de cada clase por igual y
    delata ese fallo.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_real, y_pred,
                                target_names=nombres_clases, digits=3))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(confusion_matrix(y_real, y_pred))


def errores_alta_confianza(
    y_real: list[int],
    y_pred: list[int],
    confianzas: list[float],
    top_k: int = 10,
) -> list[tuple[int, int, int, float]]:
    """Devuelve los `top_k` errores con mayor confianza:
    (índice, label_real, predicción, confianza).

    Estos ejemplos son oro para el análisis: revelan etiquetas dudosas,
    casos ambiguos, sesgos del dataset o límites reales del modelo.
    La pregunta a responder para cada uno: ¿el error es del DATO o del
    MODELO?
    """
    errores = [
        (i, real, pred, conf)
        for i, (real, pred, conf) in enumerate(zip(y_real, y_pred, confianzas))
        if real != pred
    ]
    errores.sort(key=lambda item: item[3], reverse=True)
    return errores[:top_k]

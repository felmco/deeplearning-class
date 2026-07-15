"""Utilidades transversales del curso.

Qué es: módulo con las funciones que TODOS los laboratorios necesitan.
Qué hace: fija semillas para reproducibilidad, detecta el mejor
dispositivo disponible (CUDA / MPS / CPU), carga configuraciones YAML
y cuenta parámetros de un modelo.

Ejecutar `python -m src.utils` verifica el entorno completo.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fija TODAS las semillas de aleatoriedad relevantes.

    Por qué importa: sin semillas fijas, dos ejecuciones del mismo
    código producen resultados distintos y los experimentos dejan de
    ser comparables. Es el primer paso de cualquier experimento serio.
    """
    random.seed(seed)            # aleatoriedad de Python puro
    np.random.seed(seed)         # aleatoriedad de NumPy
    torch.manual_seed(seed)      # aleatoriedad de PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # y de todas las GPUs


def detectar_dispositivo() -> torch.device:
    """Devuelve el mejor dispositivo disponible.

    Orden de preferencia:
    1. CUDA  → GPU NVIDIA (el estándar en entrenamiento).
    2. MPS   → GPU de Apple Silicon (M1/M2/M3/M4), cuando las
                operaciones son compatibles.
    3. CPU   → siempre disponible; suficiente para los laboratorios
                pequeños de este curso.

    Regla de oro: el MODELO y los DATOS deben vivir en el MISMO
    dispositivo; mover solo uno de los dos es un error clásico.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cargar_config(ruta: str | Path) -> dict:
    """Carga una configuración YAML de `configs/`.

    Mantener la configuración FUERA del código permite versionar cada
    experimento: un cambio de hiperparámetro es un diff de una línea.
    """
    import yaml

    with open(ruta, encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def contar_parametros(modelo: torch.nn.Module) -> tuple[int, int]:
    """Cuenta parámetros (entrenables, totales) de un modelo.

    Útil para comparar arquitecturas y para verificar que 'congelar el
    backbone' en transfer learning realmente congeló lo esperado.
    """
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    totales = sum(p.numel() for p in modelo.parameters())
    return entrenables, totales


def verificar_entorno() -> None:
    """Imprime las versiones instaladas y el dispositivo detectado.

    Es el 'hello world' del curso: si esto corre, el entorno está listo.
    """
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: no instalado (necesario solo en la Sesión 4)")
    print(f"CUDA        : {torch.cuda.is_available()}")
    print(f"MPS         : {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"Dispositivo : {detectar_dispositivo()}")


if __name__ == "__main__":
    verificar_entorno()

"""Smoke tests: ¿está vivo el entorno?

Qué es: las pruebas más baratas posibles.
Qué hace: verifica que las librerías críticas importan, que las
utilidades básicas funcionan y que las configuraciones YAML son
válidas. Si esto falla, nada más del curso va a funcionar.

Ejecutar:  pytest tests/test_smoke.py -q
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_import_torch():
    """PyTorch importa y reporta versión."""
    import torch

    assert torch.__version__


def test_import_transformers():
    """Transformers importa (solo necesario en la Sesión 4)."""
    import pytest

    transformers = pytest.importorskip("transformers")
    assert transformers.__version__


def test_seed_reproducible():
    """Con la misma semilla, PyTorch genera los mismos números."""
    import torch

    from src.utils import seed_everything

    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_dispositivo_valido():
    """El detector devuelve un dispositivo utilizable."""
    from src.utils import detectar_dispositivo

    assert detectar_dispositivo().type in {"cuda", "mps", "cpu"}


def test_configs_validas():
    """Las tres configuraciones YAML del curso cargan y tienen semilla."""
    from src.utils import cargar_config

    for nombre in ["mlp.yaml", "cnn.yaml", "transformer.yaml"]:
        config = cargar_config(RAIZ / "configs" / nombre)
        assert "seed" in config, f"{nombre} debe declarar una semilla"
        assert "experiment_name" in config


def test_tokenizador_basico():
    """El tokenizador simple separa palabras y puntuación."""
    from src.data import tokenizar_basico

    tokens = tokenizar_basico("¡El modelo aprende, no memoriza!")
    assert "el" in tokens
    assert "," in tokens

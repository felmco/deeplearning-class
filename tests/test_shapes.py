"""Pruebas de shapes: el contrato tensorial de cada modelo.

Qué es: la primera línea de defensa contra los errores más comunes
de Deep Learning (shape mismatches).
Qué hace: verifica que cada arquitectura del curso transforma
entradas de shape conocida en salidas de shape esperada, que los
valores son finitos y que las propiedades matemáticas clave se
cumplen (pesos de attention suman 1, máscara causal bloquea el futuro).

Ejecutar:  pytest tests/test_shapes.py -q
"""

import torch

from src.models import (
    FashionCNN,
    MeanEmbeddingClassifier,
    MoonMLP,
    MultiHeadSelfAttention,
    TransformerBlock,
    positional_encoding_sinusoidal,
    scaled_dot_product_attention,
)


def test_mlp_output_shape():
    """La MLP recibe (B, 2) y devuelve (B, 1) logits finitos."""
    model = MoonMLP(hidden_dim=16)
    x = torch.randn(8, 2)
    logits = model(x)
    assert logits.shape == (8, 1)
    assert torch.isfinite(logits).all()


def test_cnn_output_shape():
    """La CNN recibe (B, 1, 28, 28) y devuelve (B, 10) logits."""
    model = FashionCNN()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_attention_shapes_y_suma():
    """Los pesos de attention forman una distribución: cada fila suma 1."""
    torch.manual_seed(42)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (1, 1, 4, 8)
    assert weights.shape == (1, 1, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 4), atol=1e-6)


def test_mascara_causal_bloquea_futuro():
    """Con máscara causal, ningún peso 'mira' posiciones futuras."""
    torch.manual_seed(42)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool)).view(1, 1, 4, 4)

    _, weights = scaled_dot_product_attention(q, k, v, mask=causal)

    # Las posiciones prohibidas (triángulo superior) deben ser ~0
    assert torch.all(weights.masked_select(~causal) < 1e-6)


def test_attention_valida_contratos():
    """Shapes incompatibles deben fallar con un error claro."""
    import pytest

    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 16)   # d_k distinto → error
    v = torch.randn(1, 4, 8)
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v)


def test_multihead_shapes():
    """MHA preserva (B, T, d_model) y expone pesos (B, H, T, T)."""
    mha = MultiHeadSelfAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 7, 64)
    salida, pesos = mha(x, causal=True)
    assert salida.shape == (2, 7, 64)
    assert pesos.shape == (2, 4, 7, 7)


def test_transformer_block_backward():
    """El bloque completo preserva shapes y propaga gradientes."""
    torch.manual_seed(42)
    x = torch.randn(2, 7, 64, requires_grad=True)
    padding_mask = torch.tensor([
        [True, True, True, True, True, False, False],
        [True] * 7,
    ])

    block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
    y, pesos = block(x, causal=True, padding_mask=padding_mask)

    assert y.shape == x.shape
    assert pesos.shape == (2, 4, 7, 7)

    # backpropagation debe funcionar de punta a punta
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_positional_encoding_rango():
    """La PE sinusoidal tiene la shape pedida y valores en [-1, 1]."""
    pe = positional_encoding_sinusoidal(max_len=50, d_model=32)
    assert pe.shape == (50, 32)
    assert pe.min() >= -1.0 and pe.max() <= 1.0


def test_mean_embedding_ignora_padding():
    """El pooling enmascarado no cambia si se agrega padding extra."""
    torch.manual_seed(0)
    model = MeanEmbeddingClassifier(vocab_size=100)
    model.eval()

    ids = torch.tensor([[5, 9, 3]])
    mask = torch.tensor([[True, True, True]])

    # Misma secuencia con dos posiciones de padding al final
    ids_pad = torch.tensor([[5, 9, 3, 0, 0]])
    mask_pad = torch.tensor([[True, True, True, False, False]])

    with torch.inference_mode():
        assert torch.allclose(model(ids, mask), model(ids_pad, mask_pad),
                              atol=1e-6)

"""Arquitecturas del curso, implementadas para ser LEÍDAS.

Qué es: todos los modelos que se construyen a lo largo de las 4 sesiones.
Qué hace: define MLP (Sesión 1), CNN (Sesión 2), attention y bloque
Transformer desde cero (Sesión 3) y el clasificador de texto con
mean pooling del proyecto final. Cada clase documenta sus shapes.

Convención de shapes usada en los comentarios:
    B = batch size · T = sequence length · d = dimensión de features
    C = canales · H, W = alto y ancho de imagen · H_att = número de heads
"""

from __future__ import annotations

import math

import torch
from torch import nn

# ────────────────────────────────────────────────────────────────────
# Sesión 1 — MLP
# ────────────────────────────────────────────────────────────────────


class MoonMLP(nn.Module):
    """MLP para clasificación binaria de make_moons.

    Arquitectura: 2 → hidden → hidden → 1 (logit).
    - Las capas Linear calculan z = xW + b (transformación afín).
    - ReLU introduce la no linealidad: sin ella, apilar capas lineales
      equivale a UNA sola transformación lineal.
    - La salida es UN LOGIT crudo (sin sigmoid): BCEWithLogitsLoss
      aplica sigmoid internamente de forma numéricamente estable.
    """

    def __init__(self, hidden_dim: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),      # (B, 2) → (B, hidden)
            nn.ReLU(),
            nn.Dropout(dropout),           # regularización: apaga neuronas al azar
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),      # (B, hidden) → (B, 1) logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) features estandarizadas → (B, 1) logits."""
        return self.net(x)


# ────────────────────────────────────────────────────────────────────
# Sesión 2 — CNN
# ────────────────────────────────────────────────────────────────────


class FashionCNN(nn.Module):
    """CNN compacta para FashionMNIST (imágenes 1×28×28, 10 clases).

    Patrón clásico: [Conv → BatchNorm → ReLU → Pool] × bloques + head.
    - Conv2d detecta patrones locales con kernels APRENDIDOS.
    - BatchNorm normaliza activaciones por mini-batch → permite LRs
      mayores y estabiliza el entrenamiento.
    - MaxPool reduce resolución espacial: 28 → 14 → 7.
    - AdaptiveAvgPool2d((1,1)) colapsa cada feature map a un número:
      hace la red robusta al tamaño de entrada y evita Flatten frágil.

    Seguimiento de shapes (batch B):
        entrada        (B,   1, 28, 28)
        bloque 1       (B,  32, 14, 14)
        bloque 2       (B,  64,  7,  7)
        bloque 3       (B, 128,  1,  1)   ← AdaptiveAvgPool
        clasificador   (B, 10) logits
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 1 canal de entrada → 32 filtros aprendidos
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # padding=1 preserva 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28 → 14
            # Bloque 2: 32 → 64 filtros
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14 → 7
            # Bloque 3: 64 → 128 filtros, luego pooling global
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                 # (B,128,7,7) → (B,128,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (B,128,1,1) → (B,128)
            nn.Dropout(dropout),
            nn.Linear(128, 10),                           # (B,128) → (B,10) logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) → (B, 10) logits (¡sin softmax!)."""
        return self.classifier(self.features(x))


# ────────────────────────────────────────────────────────────────────
# Sesión 3 — Attention y Transformer desde cero
# ────────────────────────────────────────────────────────────────────


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """La ecuación central de los Transformers, paso a paso.

        Attention(Q,K,V) = softmax(QKᵀ/√d_k + M) · V

    Shapes esperados (los `...` son dimensiones de batch/heads):
        query: (..., T_q, d_k)   →  qué busca cada posición
        key:   (..., T_k, d_k)   →  qué ofrece cada posición
        value: (..., T_k, d_v)   →  la información que se transporta
        mask:  broadcastable a (..., T_q, T_k); True = permitido

    Devuelve (output, weights):
        output:  (..., T_q, d_v) — mezcla ponderada de los values
        weights: (..., T_q, T_k) — cada fila es una distribución (suma 1)
    """
    # Validaciones de contrato: fallar temprano con mensajes claros
    if query.size(-1) != key.size(-1):
        raise ValueError("query y key deben compartir d_k")
    if key.size(-2) != value.size(-2):
        raise ValueError("key y value deben compartir T_k")

    d_k = query.size(-1)

    # 1. Compatibilidad entre cada query y cada key: (..., T_q, T_k)
    scores = query @ key.transpose(-2, -1)

    # 2. Escalar por √d_k: sin esto, con d_k grande los scores crecen,
    #    el softmax se satura y los gradientes se desvanecen.
    scores = scores / math.sqrt(d_k)

    # 3. Máscara: las posiciones prohibidas reciben -inf efectivo,
    #    de modo que el softmax les asigna probabilidad ~0.
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # 4. Softmax por filas: cada query reparte "atención" que suma 1.
    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        weights = torch.nn.functional.dropout(weights, p=dropout_p,
                                              training=training)

    # 5. Mezcla ponderada de values: la información fluye entre tokens.
    output = weights @ value
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention: h "miradas" paralelas sobre la secuencia.

    En lugar de UNA atención con d_model dimensiones, se hacen h
    atenciones con d_model/h dimensiones cada una: distintos heads
    aprenden relaciones complementarias (sintaxis, correferencia, etc.).

        head_i = Attention(Q·Wᵢ^Q, K·Wᵢ^K, V·Wᵢ^V)
        MHA(x) = Concat(head_1 … head_h) · W^O
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model debe ser divisible por num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Una sola proyección produce Q, K y V a la vez (eficiencia)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)   # el W^O de la fórmula
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d_model) → (salida (B, T, d_model), pesos (B, H, T, T)).

        - causal=True impide atender al futuro (decoder GPT-like).
        - padding_mask: (B, T) con True para tokens reales; evita que
          el padding reciba atención.
        """
        batch_size, seq_len, _ = x.shape

        # Proyectar y separar en Q, K, V: cada uno (B, T, d_model)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def dividir_heads(tensor: torch.Tensor) -> torch.Tensor:
            """(B, T, d_model) → (B, H, T, head_dim)."""
            return tensor.view(batch_size, seq_len, self.num_heads,
                               self.head_dim).transpose(1, 2)

        q, k, v = map(dividir_heads, (q, k, v))

        # Construir la máscara combinada (causal ∧ padding)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        if causal:
            mask = torch.tril(mask)          # triángulo inferior: no ver futuro
        mask = mask.view(1, 1, seq_len, seq_len)
        if padding_mask is not None:
            # El padding no puede ser CONSULTADO como key
            mask = mask & padding_mask[:, None, None, :]

        atendido, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
        )

        # Reunir los heads: (B, H, T, head_dim) → (B, T, d_model)
        unido = atendido.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.out_proj(unido), weights


class TransformerBlock(nn.Module):
    """Un bloque Transformer pre-norm completo.

        x' = x + MHA(LayerNorm(x))     ← attention MEZCLA info entre tokens
        y  = x' + FFN(LayerNorm(x'))   ← FFN TRANSFORMA cada token por separado

    Las conexiones residuales (el `x +`) crean una "autopista" para el
    gradiente y permiten apilar decenas de bloques sin que el
    entrenamiento colapse. Pre-norm (LayerNorm ANTES de cada subcapa)
    es la variante más estable, usada en GPT-2 en adelante.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN: expande a d_ff (típicamente 4×d_model), no-linealidad, contrae
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                    # la activación estándar en Transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, d_model) → (misma shape, pesos de atención)."""
        salida_attn, weights = self.attention(
            self.norm1(x), causal=causal, padding_mask=padding_mask
        )
        x = x + salida_attn               # residual 1
        x = x + self.ffn(self.norm2(x))   # residual 2
        return x, weights


def positional_encoding_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    """Codificación posicional sinusoidal del paper original.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Por qué existe: la self-attention es PERMUTACIONALMENTE INVARIANTE
    (no distingue el orden de los tokens). Sumar esta "firma" de ondas
    a los embeddings inyecta la noción de posición.

    Devuelve: (max_len, d_model).
    """
    posicion = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(posicion * div_term)   # dimensiones pares
    pe[:, 1::2] = torch.cos(posicion * div_term)   # dimensiones impares
    return pe


# ────────────────────────────────────────────────────────────────────
# Proyecto final — clasificador neuronal propio
# ────────────────────────────────────────────────────────────────────


class MeanEmbeddingClassifier(nn.Module):
    """Clasificador de texto: Embedding → mean pooling con máscara → MLP.

    Es el "modelo neuronal propio" del proyecto final: demuestra
    embeddings, padding masks y pooling sin depender de un modelo
    preentrenado. Sus limitaciones (ignora el orden de las palabras)
    son parte de la lección.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 64, num_classes: int = 2,
                 pad_id: int = 0) -> None:
        super().__init__()
        # padding_idx congela el embedding de <pad> en ceros
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_id)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) enteros · mask: (B, T) bool → (B, C) logits.

        El mean pooling ENMASCARADO promedia solo los tokens reales:
        dividir por mask.sum() y no por T evita que las secuencias
        cortas queden diluidas por su padding.
        """
        embeddings = self.embedding(input_ids)            # (B, T, d)
        enmascarado = embeddings * mask.unsqueeze(-1)     # anula el padding
        pooled = enmascarado.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)                    # (B, num_classes)

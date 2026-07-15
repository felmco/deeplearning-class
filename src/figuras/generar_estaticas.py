"""Generador de figuras estáticas del curso.

Qué es: script reproducible que crea todas las visualizaciones PNG
del material didáctico.
Qué hace: cada función `fig_*` produce una figura y la guarda en
`docs/assets/figuras/`. Ejecutar con:

    python -m src.figuras.generar_estaticas
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guardar archivos

import matplotlib.pyplot as plt
import numpy as np

from .estilo import (
    AMARILLO,
    AZUL,
    CELESTE,
    GRIS,
    MORADO,
    NARANJA,
    ROJO,
    VERDE,
    aplicar_estilo,
)

# Carpeta destino de todas las figuras (relativa a la raíz del repo)
DESTINO = Path(__file__).resolve().parents[2] / "docs" / "assets" / "figuras"


# ────────────────────────────────────────────────────────────────────
# Sesión 1 — Fundamentos
# ────────────────────────────────────────────────────────────────────

def fig_activaciones() -> None:
    """Funciones de activación y sus derivadas, lado a lado.

    Por qué importa: la derivada es lo que fluye hacia atrás en
    backpropagation; donde la derivada es ~0, el aprendizaje se detiene
    (saturación de sigmoid/tanh, zona muerta de ReLU).
    """
    z = np.linspace(-6, 6, 500)

    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)
    # GELU aproximada con tanh (la usada por defecto en muchos Transformers)
    gelu = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

    d_sigmoid = sigmoid * (1 - sigmoid)
    d_tanh = 1 - tanh**2
    d_relu = (z > 0).astype(float)
    d_gelu = np.gradient(gelu, z)  # derivada numérica (suficiente para ilustrar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for valores, nombre in [(sigmoid, "sigmoid"), (tanh, "tanh"),
                            (relu, "ReLU"), (gelu, "GELU")]:
        ax1.plot(z, valores, label=nombre, linewidth=2.2)
    ax1.set(title="Funciones de activación", xlabel="z", ylabel="φ(z)")
    ax1.axhline(0, color=GRIS, linewidth=0.8)
    ax1.axvline(0, color=GRIS, linewidth=0.8)
    ax1.legend()

    for valores, nombre in [(d_sigmoid, "sigmoid′"), (d_tanh, "tanh′"),
                            (d_relu, "ReLU′"), (d_gelu, "GELU′")]:
        ax2.plot(z, valores, label=nombre, linewidth=2.2)
    ax2.set(title="Derivadas: lo que fluye en backprop", xlabel="z", ylabel="φ′(z)")
    ax2.axhline(0, color=GRIS, linewidth=0.8)
    ax2.annotate("saturación:\ngradiente ≈ 0", xy=(-5, 0.02), xytext=(-5.8, 0.45),
                 fontsize=9, color=ROJO,
                 arrowprops=dict(arrowstyle="->", color=ROJO))
    ax2.legend()

    fig.savefig(DESTINO / "activaciones.png")
    plt.close(fig)


def fig_softmax_temperatura() -> None:
    """Efecto de la temperatura sobre la distribución softmax.

    softmax(z/T): T<1 concentra la probabilidad (más determinista),
    T>1 la aplana (más diversa). Es la base del sampling en LLMs.
    """
    logits = np.array([2.0, 1.0, 0.5, 0.2, -0.5])
    etiquetas = ["gato", "perro", "león", "lobo", "pez"]
    temperaturas = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, T in zip(axes, temperaturas):
        # Softmax numéricamente estable: restar el máximo antes de exponenciar
        z = logits / T
        p = np.exp(z - z.max())
        p /= p.sum()
        colores = [AZUL if i == p.argmax() else "#9DC3E6" for i in range(len(p))]
        ax.bar(etiquetas, p, color=colores)
        ax.set(title=f"T = {T}", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    axes[0].set_ylabel("probabilidad")
    fig.suptitle("softmax(z/T): la temperatura controla la 'confianza' de la distribución",
                 fontweight="bold")
    fig.savefig(DESTINO / "softmax_temperatura.png")
    plt.close(fig)


def fig_curvas_aprendizaje() -> None:
    """Los tres regímenes clásicos: underfitting, good fit y overfitting."""
    epochs = np.arange(1, 101)
    rng = np.random.default_rng(42)
    ruido = lambda s: rng.normal(0, s, len(epochs))  # noqa: E731

    escenarios = {
        "Underfitting": (
            1.1 - 0.25 * (1 - np.exp(-epochs / 40)) + ruido(0.008),
            1.15 - 0.25 * (1 - np.exp(-epochs / 40)) + ruido(0.008),
            "ambas curvas altas:\nfalta capacidad o entrenamiento",
        ),
        "Good fit": (
            1.1 * np.exp(-epochs / 22) + 0.18 + ruido(0.008),
            1.1 * np.exp(-epochs / 22) + 0.24 + ruido(0.010),
            "brecha pequeña y estable",
        ),
        "Overfitting": (
            1.1 * np.exp(-epochs / 15) + 0.06 + ruido(0.006),
            1.1 * np.exp(-epochs / 18) + 0.22 + 0.0035 * np.maximum(epochs - 35, 0)
            + ruido(0.012),
            "validation sube mientras\ntrain sigue bajando",
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (titulo, (tr, va, nota)) in zip(axes, escenarios.items()):
        ax.plot(epochs, tr, label="train", color=AZUL, linewidth=2)
        ax.plot(epochs, va, label="validation", color=NARANJA, linewidth=2)
        ax.set(title=titulo, xlabel="epoch")
        ax.text(0.5, 0.95, nota, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="center", color=GRIS, style="italic")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    fig.suptitle("Diagnóstico por curvas de aprendizaje", fontweight="bold")
    fig.savefig(DESTINO / "curvas_aprendizaje.png")
    plt.close(fig)


def fig_learning_rate() -> None:
    """Efecto real del learning rate: descenso de gradiente sobre una
    cuadrática. LR bajo avanza lento, LR adecuado converge, LR alto diverge.
    """
    def descenso(lr: float, pasos: int = 40) -> np.ndarray:
        """Minimiza f(θ)=θ² con actualizaciones θ ← θ − lr·∇f."""
        theta, historia = 2.5, []
        for _ in range(pasos):
            historia.append(theta**2)          # pérdida actual
            theta = theta - lr * 2 * theta      # gradiente de θ² es 2θ
        return np.array(historia)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for lr, color, nombre in [(0.02, AZUL, "LR bajo (0.02): lento"),
                              (0.30, VERDE, "LR adecuado (0.30): converge"),
                              (1.02, ROJO, "LR alto (1.02): diverge")]:
        ax.plot(descenso(lr), label=nombre, color=color, linewidth=2.2)
    ax.set(title="El learning rate decide el destino del entrenamiento",
           xlabel="paso de optimización", ylabel="loss  f(θ)=θ²", yscale="log")
    ax.legend()
    fig.savefig(DESTINO / "learning_rate.png")
    plt.close(fig)


def fig_moons_frontera() -> None:
    """Entrena una MLP mínima EN NUMPY sobre make_moons y dibuja la
    frontera de decisión con distintas capacidades (4 vs 64 neuronas).

    Es la evidencia visual central del Laboratorio 1.
    """
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
    X = StandardScaler().fit_transform(X)

    def entrenar_mlp(hidden: int, epochs: int = 3000, lr: float = 0.5, seed: int = 0):
        """MLP 2-hidden-1 entrenada con gradient descent manual (numpy).

        Devuelve una función que predice probabilidades sobre una malla.
        Implementación pedagógica: forward y backward explícitos.
        """
        rng = np.random.default_rng(seed)
        # Inicialización tipo He para la capa ReLU
        W1 = rng.normal(0, np.sqrt(2 / 2), (2, hidden))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1))
        b2 = np.zeros(1)
        Y = y.reshape(-1, 1)

        for _ in range(epochs):
            # ── forward ──
            Z1 = X @ W1 + b1
            H1 = np.maximum(0, Z1)              # ReLU
            logits = H1 @ W2 + b2
            P = 1 / (1 + np.exp(-logits))       # sigmoid
            # ── backward (derivadas de BCE con sigmoid) ──
            dlogits = (P - Y) / len(X)
            dW2 = H1.T @ dlogits
            db2 = dlogits.sum(0)
            dH1 = dlogits @ W2.T
            dZ1 = dH1 * (Z1 > 0)                # derivada de ReLU
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(0)
            # ── update ──
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        def predecir(G: np.ndarray) -> np.ndarray:
            H = np.maximum(0, G @ W1 + b1)
            return 1 / (1 + np.exp(-(H @ W2 + b2)))

        return predecir

    xx, yy = np.meshgrid(np.linspace(-2.6, 2.6, 300), np.linspace(-2.6, 2.6, 300))
    malla = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, hidden in zip(axes, [4, 64]):
        prob = entrenar_mlp(hidden)(malla).reshape(xx.shape)
        ax.contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.65)
        ax.contour(xx, yy, prob, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k",
                   s=18, linewidths=0.4)
        ax.set(title=f"hidden_dim = {hidden}", xlabel="x₁", ylabel="x₂")
        ax.grid(False)
    fig.suptitle("La capacidad de la MLP moldea la frontera de decisión (make_moons)",
                 fontweight="bold")
    fig.savefig(DESTINO / "moons_frontera.png")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Sesión 2 — CNN
# ────────────────────────────────────────────────────────────────────

def _imagen_ejemplo() -> np.ndarray:
    """Devuelve una imagen en escala de grises (sklearn sample image)."""
    from sklearn.datasets import load_sample_image
    img = load_sample_image("china.jpg").mean(axis=2)  # RGB → gris
    img = img[60:260, 150:350]                          # recorte cuadrado
    return img / 255.0


def _conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución 2D 'valid' implementada a mano (pedagógica, sin librerías)."""
    kh, kw = kernel.shape
    h, w = img.shape
    salida = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(salida.shape[0]):
        for j in range(salida.shape[1]):
            salida[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return salida


def fig_kernels() -> None:
    """Kernels clásicos aplicados a una imagen real: la antesala de los
    filtros APRENDIDOS por una CNN."""
    img = _imagen_ejemplo()
    kernels = {
        "bordes horizontales": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),
        "bordes verticales": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("original")
    for ax, (nombre, k) in zip(axes[1:], kernels.items()):
        ax.imshow(_conv2d(img, k), cmap="gray")
        ax.set_title(nombre)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Kernels diseñados a mano — una CNN APRENDE los suyos por backprop",
                 fontweight="bold")
    fig.savefig(DESTINO / "kernels_conv.png")
    plt.close(fig)


def fig_padding_stride() -> None:
    """Efecto de padding y stride sobre el tamaño de salida.

    Fórmula: H_out = floor((H + 2P − K) / S) + 1
    """
    casos = [
        ("K=3, P=0, S=1", 8, 3, 0, 1),
        ("K=3, P=1, S=1 ('same')", 8, 3, 1, 1),
        ("K=3, P=1, S=2", 8, 3, 1, 2),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
    for ax, (titulo, H, K, P, S) in zip(axes, casos):
        H_out = (H + 2 * P - K) // S + 1
        total = H + 2 * P
        # Dibujar la malla de entrada (con padding en gris)
        for i in range(total):
            for j in range(total):
                es_pad = i < P or j < P or i >= P + H or j >= P + H
                color = "#DDDDDD" if es_pad else "#9DC3E6"
                ax.add_patch(plt.Rectangle((j, total - 1 - i), 0.94, 0.94,
                                           facecolor=color, edgecolor="white"))
        # Resaltar la primera ventana del kernel
        ax.add_patch(plt.Rectangle((0, total - K), K - 0.06, K - 0.06,
                                   facecolor="none", edgecolor=ROJO, linewidth=2.5))
        ax.set(title=f"{titulo}\nentrada {H}×{H} → salida {H_out}×{H_out}",
               xlim=(-0.5, total + 0.5), ylim=(-0.5, total + 0.5))
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("H_out = ⌊(H + 2P − K)/S⌋ + 1   (gris = padding, rojo = kernel)",
                 fontweight="bold")
    fig.savefig(DESTINO / "padding_stride.png")
    plt.close(fig)


def fig_receptive_field() -> None:
    """Crecimiento del receptive field con la profundidad: una activación
    profunda 've' una región cada vez mayor de la imagen original."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    capas = [(1, "capa 1: ve 3×3", AZUL), (2, "capa 2: ve 5×5", VERDE),
             (3, "capa 3: ve 7×7", NARANJA)]
    N = 9
    for i in range(N):
        for j in range(N):
            ax.add_patch(plt.Rectangle((j, N - 1 - i), 0.94, 0.94,
                                       facecolor="#EEEEEE", edgecolor="white"))
    for profundidad, etiqueta, color in reversed(capas):
        rf = 2 * profundidad + 1                    # RF con kernels 3×3, stride 1
        off = (N - rf) / 2
        ax.add_patch(plt.Rectangle((off, off), rf - 0.06, rf - 0.06,
                                   facecolor="none", edgecolor=color, linewidth=2.6,
                                   label=etiqueta))
    ax.set(title="Receptive field: kernels 3×3 apilados", xlim=(-0.5, N + 4.5),
           ylim=(-0.5, N + 0.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="center right")
    fig.savefig(DESTINO / "receptive_field.png")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Sesión 3 — Secuencias y Transformers
# ────────────────────────────────────────────────────────────────────

def fig_rnn_gradientes() -> None:
    """Vanishing/exploding gradients en una RNN: la norma del gradiente
    a través del tiempo es un producto de Jacobianos; si su norma media
    es <1 se desvanece, si es >1 explota."""
    T = 40
    pasos = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for factor, color, nombre in [(0.90, AZUL, "‖J‖≈0.90 → vanishing"),
                                  (1.00, VERDE, "‖J‖≈1.00 → estable (ideal)"),
                                  (1.10, ROJO, "‖J‖≈1.10 → exploding")]:
        ax.plot(pasos, factor ** pasos, label=nombre, color=color, linewidth=2.2)
    ax.set(title="El gradiente atraviesa T pasos: ‖∂L/∂h₁‖ ∝ ‖J‖ᵀ",
           xlabel="distancia temporal T", ylabel="magnitud relativa del gradiente",
           yscale="log")
    ax.legend()
    fig.savefig(DESTINO / "rnn_gradientes.png")
    plt.close(fig)


def fig_atencion() -> None:
    """Pipeline visual de scaled dot-product attention con máscara causal:
    scores → escala → máscara → softmax. Calculado con numpy real."""
    rng = np.random.default_rng(3)
    tokens = ["El", "gato", "come", "pescado"]
    T, d_k = len(tokens), 8
    Q = rng.normal(size=(T, d_k))
    K = rng.normal(size=(T, d_k))

    scores = Q @ K.T
    escalado = scores / np.sqrt(d_k)
    mascara = np.triu(np.ones((T, T), bool), k=1)   # True = posición futura
    enmascarado = np.where(mascara, -1e9, escalado)
    # Softmax por filas, numéricamente estable
    e = np.exp(enmascarado - enmascarado.max(axis=1, keepdims=True))
    pesos = e / e.sum(axis=1, keepdims=True)

    matrices = [(scores, "1. scores = QKᵀ"),
                (escalado, "2. ÷ √d_k"),
                (np.where(mascara, np.nan, escalado), "3. máscara causal"),
                (pesos, "4. softmax → pesos")]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (M, titulo) in zip(axes, matrices):
        im = ax.imshow(M, cmap="viridis")
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks(range(T), tokens, rotation=45, fontsize=8)
        ax.set_yticks(range(T), tokens, fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
        # Anotar los valores en la matriz final de pesos
        if "softmax" in titulo:
            for i in range(T):
                for j in range(T):
                    if pesos[i, j] > 0.001:
                        ax.text(j, i, f"{pesos[i, j]:.2f}", ha="center",
                                va="center", fontsize=7, color="white")
    fig.suptitle("Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)·V — cada fila suma 1",
                 fontweight="bold")
    fig.savefig(DESTINO / "atencion_pipeline.png")
    plt.close(fig)


def fig_positional_encoding() -> None:
    """Positional encoding sinusoidal: cada posición recibe una 'firma'
    única formada por ondas de distintas frecuencias."""
    max_len, d_model = 100, 64
    pos = np.arange(max_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    div = np.exp(i * (-np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    im = ax1.imshow(pe, aspect="auto", cmap="RdBu")
    ax1.set(title="Matriz PE (posición × dimensión)",
            xlabel="dimensión del embedding", ylabel="posición en la secuencia")
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.03)

    for dim in [0, 8, 20]:
        ax2.plot(pe[:, dim], label=f"dim {dim}", linewidth=1.8)
    ax2.set(title="Cada dimensión oscila a distinta frecuencia",
            xlabel="posición", ylabel="valor")
    ax2.legend(fontsize=9)
    fig.suptitle("PE(pos, 2i) = sin(pos/10000^{2i/d}) · PE(pos, 2i+1) = cos(…)",
                 fontweight="bold")
    fig.savefig(DESTINO / "positional_encoding.png")
    plt.close(fig)


def fig_embeddings_2d() -> None:
    """Intuición de semántica distribuida: palabras similares quedan
    cercanas en el espacio de embeddings (proyección 2D ilustrativa)."""
    grupos = {
        "animales": (["gato", "perro", "león", "lobo"], (2.0, 2.0), AZUL),
        "comida": (["pizza", "taco", "sopa", "arroz"], (-2.0, 1.8), NARANJA),
        "verbos": (["correr", "saltar", "nadar", "volar"], (0.2, -2.2), VERDE),
    }
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7, 5))
    for nombre, (palabras, centro, color) in grupos.items():
        puntos = rng.normal(0, 0.42, (len(palabras), 2)) + np.array(centro)
        ax.scatter(puntos[:, 0], puntos[:, 1], s=60, color=color, label=nombre,
                   edgecolors="k", linewidths=0.5, zorder=3)
        for (px, py), palabra in zip(puntos, palabras):
            ax.annotate(palabra, (px, py), xytext=(6, 4),
                        textcoords="offset points", fontsize=10)
    ax.set(title="Embeddings: la similitud semántica se vuelve cercanía geométrica",
           xlabel="componente 1 (proyección)", ylabel="componente 2 (proyección)")
    ax.legend()
    fig.savefig(DESTINO / "embeddings_2d.png")
    plt.close(fig)


def fig_complejidad() -> None:
    """Costo computacional conceptual: self-attention O(T²·d) frente a
    RNN O(T·d²) secuencial."""
    T = np.linspace(1, 512, 200)
    d = 64
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T, T**2 * d, label="self-attention  O(T²·d) — paralelo", color=MORADO,
            linewidth=2.2)
    ax.plot(T, T * d**2, label="RNN  O(T·d²) — secuencial", color=VERDE,
            linewidth=2.2)
    ax.set(title="Costo vs longitud de secuencia (d=64, esquemático)",
           xlabel="longitud de secuencia T", ylabel="operaciones (escala log)",
           yscale="log")
    ax.axvline(64, color=GRIS, linewidth=0.9, linestyle="--")
    ax.annotate("T = d: punto de cruce", xy=(64, 3e5), xytext=(110, 1.1e5),
                fontsize=9, color=GRIS, arrowprops=dict(arrowstyle="->", color=GRIS))
    ax.legend()
    fig.savefig(DESTINO / "complejidad_attention.png")
    plt.close(fig)


def fig_neurona_vs_perceptron() -> None:
    """Anatomía comparada: neurona biológica (izquierda) vs perceptrón (derecha).
    Diagrama didáctico dibujado con patches; define cada componente y la
    correspondencia entre ambos modelos."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    fig, (ax_bio, ax_art) = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax in (ax_bio, ax_art):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.grid(False)

    # ─────────────── Panel izquierdo: neurona biológica ───────────────
    ax_bio.set_title("Neurona biológica", fontsize=15, fontweight="bold", pad=14)

    y_soma = 5.6   # centro vertical del soma; alinea el panel con el del perceptrón

    # Soma (cuerpo celular) y núcleo
    ax_bio.add_patch(Circle((3.3, y_soma), 1.15, facecolor=CELESTE, alpha=0.45,
                            edgecolor=AZUL, linewidth=2.0, zorder=3))
    ax_bio.add_patch(Circle((3.3, y_soma), 0.42, facecolor=AZUL, alpha=0.85,
                            edgecolor="white", linewidth=1.2, zorder=4))

    # Dendritas: ramas que ENTRAN al soma (las "features" del mundo)
    for ang in (150, 172, 195, 128, 218):
        rad = np.deg2rad(ang)
        x0, y0 = 3.3 + 2.55 * np.cos(rad), y_soma + 2.55 * np.sin(rad)
        x1, y1 = 3.3 + 1.05 * np.cos(rad), y_soma + 1.05 * np.sin(rad)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2 + 0.16
        ax_bio.plot([x0, xm, x1], [y0, ym, y1], color=AZUL, linewidth=2.2,
                    solid_capstyle="round", zorder=2)
        ax_bio.add_patch(Circle((x0, y0), 0.12, facecolor=AZUL, zorder=2))

    # Axón: el canal de salida, con vainas de mielina
    ax_bio.plot([4.45, 8.0], [y_soma, y_soma], color=NARANJA, linewidth=3.4,
                solid_capstyle="round", zorder=2)
    for xm in (5.2, 6.1, 7.0):
        ax_bio.add_patch(FancyBboxPatch((xm - 0.22, y_soma - 0.18), 0.44, 0.36,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor=AMARILLO, edgecolor=NARANJA,
                                        linewidth=1.4, zorder=3))
    # Terminales axónicos (sinapsis)
    for dy in (0.62, 0.0, -0.62):
        ax_bio.plot([8.0, 8.75], [y_soma, y_soma + dy], color=NARANJA, linewidth=2.4,
                    solid_capstyle="round", zorder=2)
        ax_bio.add_patch(Circle((8.85, y_soma + dy), 0.16, facecolor=ROJO, zorder=3))

    anotaciones_bio = [
        ("Dendritas\nreciben señales de entrada", (0.55, 8.4), (1.5, 6.95), AZUL),
        ("Soma (cuerpo celular)\nintegra las señales recibidas", (1.9, 2.6), (3.0, 4.5), AZUL),
        ("Axón\ntransmite el impulso si se\nsupera el umbral de disparo", (5.0, 8.4), (5.9, 5.85), NARANJA),
        ("Sinapsis\nentrega la señal a la\nsiguiente neurona", (7.5, 2.7), (8.6, 5.1), ROJO),
    ]
    for texto, xy_text, xy_punta, color in anotaciones_bio:
        ax_bio.annotate(texto, xy=xy_punta, xytext=xy_text, fontsize=9.5,
                        color="#222222", ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=color,
                                        linewidth=1.4, shrinkA=2, shrinkB=4))

    ax_bio.text(5.0, 0.55,
                "La neurona dispara (todo o nada) cuando la suma de estímulos\n"
                "supera un umbral. La fuerza de cada sinapsis se aprende con la experiencia.",
                ha="center", va="center", fontsize=9.5, style="italic", color=GRIS)

    # ─────────────── Panel derecho: perceptrón ───────────────
    ax_art.set_title("Perceptrón (neurona artificial)", fontsize=15,
                     fontweight="bold", pad=14)

    entradas = [("$x_1$", 7.4), ("$x_2$", 5.9), ("$x_3$", 4.4)]
    x_in, x_sum, x_act, x_out = 1.15, 4.6, 6.5, 8.8
    y_centro = 5.9

    # Nodos de entrada + aristas ponderadas
    for i, (etq, y) in enumerate(entradas, start=1):
        ax_art.add_patch(Circle((x_in, y), 0.42, facecolor="white",
                                edgecolor=AZUL, linewidth=2.0, zorder=3))
        ax_art.text(x_in, y, etq, ha="center", va="center", fontsize=12, zorder=4)
        ax_art.add_patch(FancyArrowPatch((x_in + 0.42, y), (x_sum - 0.62, y_centro),
                                         arrowstyle="->", mutation_scale=13,
                                         color=AZUL, linewidth=1.7,
                                         shrinkA=0, shrinkB=0, zorder=2))
        # Etiqueta del peso sobre la arista
        xw, yw = x_in + 1.5, y + (y_centro - y) * 0.42 + 0.16
        ax_art.text(xw, yw, f"$w_{i}$", fontsize=12, color=AZUL,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.92), zorder=4)

    # Bias: entrada constante = 1
    ax_art.add_patch(Circle((x_in, 2.9), 0.42, facecolor="#F4F4F4",
                            edgecolor=GRIS, linewidth=1.8, linestyle="--", zorder=3))
    ax_art.text(x_in, 2.9, "$1$", ha="center", va="center", fontsize=12, zorder=4)
    ax_art.add_patch(FancyArrowPatch((x_in + 0.42, 2.9), (x_sum - 0.62, y_centro - 0.3),
                                     arrowstyle="->", mutation_scale=13, color=GRIS,
                                     linewidth=1.6, linestyle="--",
                                     shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text(x_in + 1.5, 3.35, "$b$", fontsize=12, color=GRIS, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=4)

    # Nodo suma (Σ)
    ax_art.add_patch(Circle((x_sum, y_centro), 0.62, facecolor=CELESTE, alpha=0.5,
                            edgecolor=AZUL, linewidth=2.0, zorder=3))
    ax_art.text(x_sum, y_centro, r"$\Sigma$", ha="center", va="center",
                fontsize=17, zorder=4)

    # Nodo activación (φ)
    ax_art.add_patch(FancyArrowPatch((x_sum + 0.62, y_centro), (x_act - 0.62, y_centro),
                                     arrowstyle="->", mutation_scale=13, color=AZUL,
                                     linewidth=1.8, shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text((x_sum + x_act) / 2, y_centro + 0.42, "$z$", fontsize=12,
                color=AZUL, ha="center", va="center", zorder=4)
    ax_art.add_patch(Circle((x_act, y_centro), 0.62, facecolor=AMARILLO, alpha=0.65,
                            edgecolor=NARANJA, linewidth=2.0, zorder=3))
    ax_art.text(x_act, y_centro, r"$\varphi$", ha="center", va="center",
                fontsize=16, zorder=4)

    # Salida
    ax_art.add_patch(FancyArrowPatch((x_act + 0.62, y_centro), (x_out - 0.12, y_centro),
                                     arrowstyle="->", mutation_scale=13, color=NARANJA,
                                     linewidth=2.2, shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text(x_out + 0.35, y_centro, r"$\hat{y}$", fontsize=14,
                fontweight="bold", ha="center", va="center", zorder=4)

    anotaciones_art = [
        ("Entradas $x_j$\n(features)", (0.25, 9.0), (x_in, 7.85), AZUL),
        ("Pesos $w_j$\nimportancia de cada\nentrada (se aprenden)", (1.95, 9.0), (2.5, 6.85), AZUL),
        ("Bias $b$ — entrada constante\nque desplaza el umbral", (0.15, 2.05), (x_in - 0.25, 2.6), GRIS),
        ("Suma ponderada", (3.6, 7.55), (x_sum, 6.6), AZUL),
        ("Activación $\\varphi$\nintroduce la no linealidad", (5.75, 8.75), (x_act, 6.6), NARANJA),
        ("Salida $\\hat{y}$\npredicción", (8.6, 7.45), (x_out + 0.15, 6.35), NARANJA),
    ]
    for texto, xy_text, xy_punta, color in anotaciones_art:
        ax_art.annotate(texto, xy=xy_punta, xytext=xy_text, fontsize=9.5,
                        color="#222222", ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=color,
                                        linewidth=1.4, shrinkA=2, shrinkB=4))

    # Fórmulas del perceptrón, en una caja limpia bajo el diagrama
    ax_art.add_patch(FancyBboxPatch((0.55, 0.18), 8.9, 1.42,
                                    boxstyle="round,pad=0.12,rounding_size=0.18",
                                    facecolor="#F7FAFD", edgecolor=AZUL,
                                    linewidth=1.4, zorder=1))
    ax_art.text(5.0, 1.2,
                r"$z \;=\; \sum_{j=1}^{n} w_j x_j \;+\; b \;=\; "
                r"\mathbf{w}^\top \mathbf{x} + b$",
                ha="center", va="center", fontsize=14, zorder=2)
    ax_art.text(5.0, 0.48, r"$\hat{y} \;=\; \varphi(z)$",
                ha="center", va="center", fontsize=14, zorder=2)

    # ─────────────── Correspondencia entre ambos modelos ───────────────
    fig.subplots_adjust(bottom=0.16, top=0.90, wspace=0.06)
    pares = [("dendritas", r"entradas $x_j$"), ("sinapsis", r"pesos $w_j$"),
             ("soma", r"suma $\Sigma$"), ("umbral de disparo", r"activación $\varphi$"),
             ("axón", r"salida $\hat{y}$")]
    x0, ancho = 0.075, 0.855 / len(pares)
    for i, (bio, art) in enumerate(pares):
        fig.text(x0 + ancho * (i + 0.5), 0.075, f"{bio}  →  {art}",
                 ha="center", va="center", fontsize=10.5, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.42", facecolor="#F2F6FA",
                           edgecolor=CELESTE, linewidth=1.1))
    fig.text(0.5, 0.018,
             "La analogía es inspiración histórica, no equivalencia: una neurona real "
             "es mucho más compleja que esta suma ponderada.",
             ha="center", va="center", fontsize=9, style="italic", color=GRIS)
    fig.savefig(DESTINO / "neurona_vs_perceptron.png")


def main() -> None:
    """Genera todas las figuras estáticas."""
    aplicar_estilo()
    DESTINO.mkdir(parents=True, exist_ok=True)
    tareas = [
        fig_activaciones, fig_softmax_temperatura, fig_curvas_aprendizaje,
        fig_learning_rate, fig_moons_frontera, fig_kernels, fig_padding_stride,
        fig_receptive_field, fig_rnn_gradientes, fig_atencion,
        fig_positional_encoding, fig_embeddings_2d, fig_complejidad,
        fig_neurona_vs_perceptron,
    ]
    for tarea in tareas:
        tarea()
        print(f"✔ {tarea.__name__}")


if __name__ == "__main__":
    main()

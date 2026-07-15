"""Generador de animaciones GIF del curso.

Qué es: script reproducible que crea las animaciones embebidas en las
sesiones del aula.
Qué hace: cada función `gif_*` construye una animación con matplotlib
y la exporta como GIF a `docs/assets/figuras/`. Ejecutar con:

    python -m src.figuras.generar_gifs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .estilo import AZUL, ROJO, VERDE, aplicar_estilo

DESTINO = Path(__file__).resolve().parents[2] / "docs" / "assets" / "figuras"
FPS = 10


# ────────────────────────────────────────────────────────────────────
# GIF 1 — Descenso de gradiente sobre una superficie de pérdida
# ────────────────────────────────────────────────────────────────────

def gif_descenso_gradiente() -> None:
    """Tres optimizadores bajando la misma superficie no convexa:
    LR bajo (lento), LR adecuado (converge) y LR alto (rebota).
    La superficie es f(x,y) = x² + 6y² rotada: un valle alargado, el
    caso clásico donde el LR alto oscila perpendicular al valle.
    """
    def f(x, y):
        return 0.5 * x**2 + 3.0 * y**2

    def grad(x, y):
        return np.array([x, 6.0 * y])

    # Simular las trayectorias completas por adelantado
    def trayectoria(lr: float, pasos: int = 60) -> np.ndarray:
        p = np.array([-3.6, 1.8])
        puntos = [p.copy()]
        for _ in range(pasos):
            p = p - lr * grad(*p)
            p = np.clip(p, -4.5, 4.5)  # evitar que la divergencia salga del lienzo
            puntos.append(p.copy())
        return np.array(puntos)

    configs = [(0.02, AZUL, "LR = 0.02 (lento)"),
               (0.15, VERDE, "LR = 0.15 (adecuado)"),
               (0.32, ROJO, "LR = 0.32 (oscila)")]
    rutas = [trayectoria(lr) for lr, _, _ in configs]

    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-2.6, 2.6, 200))
    zz = f(xx, yy)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.contourf(xx, yy, zz, levels=28, cmap="Blues_r", alpha=0.85)
    ax.contour(xx, yy, zz, levels=14, colors="white", linewidths=0.5)
    ax.plot(0, 0, "*", color="gold", markersize=16, zorder=5,
            markeredgecolor="k", label="mínimo")
    lineas, puntos = [], []
    for (_, color, nombre) in configs:
        linea, = ax.plot([], [], color=color, linewidth=2, label=nombre)
        punto, = ax.plot([], [], "o", color=color, markersize=7,
                         markeredgecolor="k", zorder=6)
        lineas.append(linea)
        puntos.append(punto)
    ax.set(title="θ ← θ − η·∇L(θ): el learning rate define la trayectoria",
           xlabel="θ₁", ylabel="θ₂")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)

    def actualizar(frame: int):
        for ruta, linea, punto in zip(rutas, lineas, puntos):
            linea.set_data(ruta[:frame + 1, 0], ruta[:frame + 1, 1])
            punto.set_data([ruta[frame, 0]], [ruta[frame, 1]])
        return lineas + puntos

    anim = FuncAnimation(fig, actualizar, frames=61, blit=True)
    anim.save(DESTINO / "descenso_gradiente.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# GIF 2 — Kernel de convolución deslizándose sobre una imagen
# ────────────────────────────────────────────────────────────────────

def gif_convolucion() -> None:
    """Un kernel 3×3 detector de bordes recorre una imagen 8×8 y
    construye el feature map 6×6 celda por celda: la operación
    fundamental de una CNN, vista en cámara lenta.
    """
    # Imagen sintética 8×8: un cuadrado brillante sobre fondo oscuro
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1.0
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float)  # bordes horiz.

    salida = np.full((6, 6), np.nan)
    posiciones = [(i, j) for i in range(6) for j in range(6)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.2))
    ax1.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("entrada 8×8 + kernel 3×3")
    rect = plt.Rectangle((-0.5, -0.5), 3, 3, facecolor="none",
                         edgecolor=ROJO, linewidth=2.6)
    ax1.add_patch(rect)
    im2 = ax2.imshow(salida, cmap="RdBu_r", vmin=-3, vmax=3)
    ax2.set_title("feature map 6×6 (respuesta del filtro)")
    for ax in (ax1, ax2):
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    texto = fig.suptitle("", fontweight="bold")

    def actualizar(frame: int):
        i, j = posiciones[frame]
        # Producto elemento a elemento + suma = un valor del feature map
        valor = np.sum(img[i:i + 3, j:j + 3] * kernel)
        salida[i, j] = valor
        rect.set_xy((j - 0.5, i - 0.5))
        im2.set_data(salida)
        texto.set_text(f"Y[{i},{j}] = Σ ventana ⊙ kernel = {valor:+.0f}")
        return [im2, rect]

    anim = FuncAnimation(fig, actualizar, frames=len(posiciones), blit=False)
    anim.save(DESTINO / "convolucion.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# GIF 3 — Evolución de la frontera de decisión durante el entrenamiento
# ────────────────────────────────────────────────────────────────────

def gif_frontera_evolucion() -> None:
    """Una MLP (numpy) aprende make_moons: se captura la frontera de
    decisión cada N pasos para VER el aprendizaje ocurriendo.
    """
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=400, noise=0.20, random_state=42)
    X = StandardScaler().fit_transform(X)
    Y = y.reshape(-1, 1)

    rng = np.random.default_rng(0)
    hidden = 16
    W1 = rng.normal(0, 1.0, (2, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
    b2 = np.zeros(1)

    xx, yy2 = np.meshgrid(np.linspace(-2.6, 2.6, 160), np.linspace(-2.6, 2.6, 160))
    malla = np.c_[xx.ravel(), yy2.ravel()]

    def paso(lr: float = 0.8) -> float:
        """Un paso de gradient descent completo; devuelve la BCE loss."""
        nonlocal W1, b1, W2, b2
        Z1 = X @ W1 + b1
        H1 = np.maximum(0, Z1)
        logits = H1 @ W2 + b2
        P = 1 / (1 + np.exp(-logits))
        eps = 1e-9
        loss = -np.mean(Y * np.log(P + eps) + (1 - Y) * np.log(1 - P + eps))
        dlogits = (P - Y) / len(X)
        dW2 = H1.T @ dlogits
        db2 = dlogits.sum(0)
        dZ1 = (dlogits @ W2.T) * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        return loss

    def predecir(G):
        H = np.maximum(0, G @ W1 + b1)
        return 1 / (1 + np.exp(-(H @ W2 + b2)))

    # Capturar 36 fotogramas a lo largo de 1800 pasos de entrenamiento
    fotogramas = []
    for k in range(36):
        for _ in range(50):
            loss = paso()
        fotogramas.append((predecir(malla).reshape(xx.shape), (k + 1) * 50, loss))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.grid(False)

    def actualizar(frame: int):
        prob, pasos_n, loss = fotogramas[frame]
        ax.clear()
        ax.contourf(xx, yy2, prob, levels=20, cmap="RdBu_r", alpha=0.65)
        ax.contour(xx, yy2, prob, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k",
                   s=16, linewidths=0.4)
        ax.set_title(f"paso {pasos_n} — BCE loss = {loss:.3f}", fontweight="bold")
        ax.grid(False)
        return []

    anim = FuncAnimation(fig, actualizar, frames=len(fotogramas), blit=False)
    anim.save(DESTINO / "frontera_evolucion.gif", writer=PillowWriter(fps=6))
    plt.close(fig)


def main() -> None:
    """Genera todas las animaciones GIF."""
    aplicar_estilo()
    DESTINO.mkdir(parents=True, exist_ok=True)
    for tarea in [gif_descenso_gradiente, gif_convolucion, gif_frontera_evolucion]:
        tarea()
        print(f"✔ {tarea.__name__}")


if __name__ == "__main__":
    main()

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

TINTA_LINEA = "#1a2332"   # color de las fronteras de decisión dibujadas

# Carpeta destino de todas las figuras (relativa a la raíz del repo)
DESTINO = Path(__file__).resolve().parents[2] / "docs" / "assets" / "figuras"


# ────────────────────────────────────────────────────────────────────
# Sesión 1 — Fundamentos
# ────────────────────────────────────────────────────────────────────

def fig_tensores() -> None:
    """La evolución escalar → vector → matriz → tensor 3D.

    Cada paso agrega UN eje: esa es toda la historia. Se dibuja con celdas
    2D y capas desplazadas en diagonal (2.5D) para el tensor, con el shape
    y un ejemplo del curso bajo cada etapa.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)

    LADO = 0.85          # lado de cada celda
    y_base = 4.4         # base vertical de los dibujos

    def celda(x, y, color=CELESTE, alpha=0.55, texto=None, zorder=3):
        ax.add_patch(Rectangle((x, y), LADO, LADO, facecolor=color, alpha=alpha,
                               edgecolor=AZUL, linewidth=1.4, zorder=zorder))
        if texto is not None:
            ax.text(x + LADO / 2, y + LADO / 2, texto, ha="center", va="center",
                    fontsize=9, zorder=zorder + 1)

    def rotulo(xc, nombre, shape, ejemplo):
        ax.text(xc, 2.6, nombre, ha="center", fontsize=13, fontweight="bold")
        ax.text(xc, 1.9, shape, ha="center", fontsize=11, family="monospace",
                color=AZUL, fontweight="bold")
        ax.text(xc, 1.15, ejemplo, ha="center", fontsize=9, color=GRIS,
                style="italic")

    def flecha(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, y_base + 1.2), (x1, y_base + 1.2),
                                     arrowstyle="->", mutation_scale=16,
                                     color=NARANJA, linewidth=2.2))
        ax.text((x0 + x1) / 2, y_base + 1.75, "+1 eje", ha="center",
                fontsize=10, color=NARANJA, fontweight="bold")

    # ── Escalar: un solo número ──
    celda(1.6, y_base + 0.8, texto="3.7")
    rotulo(2.0, "Escalar", "()", "una loss: 0.693")

    flecha(3.2, 4.6)

    # ── Vector: fila de números (1 eje) ──
    for j in range(5):
        celda(5.0 + j * LADO, y_base + 0.8, texto=f"{0.2 * (j + 1):.1f}")
    rotulo(7.1, "Vector", "(5,)", "un embedding")

    flecha(9.9, 11.3)

    # ── Matriz: cuadrícula (2 ejes) ──
    for i in range(4):
        for j in range(5):
            celda(11.7 + j * LADO, y_base - 0.9 + (3 - i) * LADO)
    rotulo(13.8, "Matriz", "(4, 5)", "batch de 4 muestras\ncon 5 features")

    flecha(16.6, 18.0)

    # ── Tensor 3D: capas apiladas en diagonal (3 ejes) ──
    DX, DY = 0.42, 0.34            # desplazamiento diagonal por capa
    colores_capa = ["#BBDEF5", "#A7D9C9", "#E5C7DB"]   # frontal, media, trasera
    for k in range(2, -1, -1):     # capa trasera primero; las frontales la tapan
        ox, oy = 18.4 + k * DX, y_base - 0.9 + k * DY
        for i in range(4):
            for j in range(5):
                celda(ox + j * LADO, oy + (3 - i) * LADO,
                      color=colores_capa[k], alpha=1.0, zorder=3 + (2 - k))
    ax.text(23.9, y_base + 2.4, "3 capas\n(canales, tiempo, …)",
            fontsize=9, color=GRIS, ha="left", va="center")
    rotulo(20.9, "Tensor 3D", "(3, 4, 5)", "3 canales de una imagen,\no 3 pasos de una secuencia")

    ax.text(13.5, 9.4,
            "Cada paso agrega UN eje — un tensor es solo la generalización: "
            "una caja de números con tantos ejes como necesites",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(13.5, 8.55,
            "El shape cuenta cuántos elementos hay por eje; leerlo en voz alta "
            "es entender el dato   ·   (3, 4, 5) = \"3 capas de 4 filas × 5 columnas\"",
            ha="center", fontsize=10, color=GRIS)

    fig.savefig(DESTINO / "tensores_evolucion.png")
    plt.close(fig)


def fig_neurona_frontera() -> None:
    """La geometría de una neurona en 3 paneles: (1) z=0 es una línea que
    parte el plano en dos mitades; (2) el bias b DESPLAZA esa línea sin
    girarla; (3) los pesos w la GIRAN. Puntos sintéticos clasificados con
    la propia neurona, para que la figura sea literal y no un esquema."""
    rng = np.random.default_rng(7)

    w = np.array([1.0, 0.8])          # pesos de la neurona de ejemplo
    b0 = -0.4                          # bias del panel 1

    def linea(ax, w, b, **kw):
        """Dibuja w·x + b = 0 dentro del recuadro [-3, 3]."""
        xs = np.linspace(-3, 3, 2)
        # w1*x + w2*y + b = 0  →  y = -(w1*x + b) / w2
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── Panel 1: la neurona parte el plano en dos mitades ──
    ax = axes[0]
    X = rng.uniform(-2.7, 2.7, size=(90, 2))
    z = X @ w + b0                     # la propia neurona clasifica los puntos
    ax.scatter(*X[z > 0].T, s=26, color=AZUL, alpha=0.75, label="z > 0 → clase 1")
    ax.scatter(*X[z <= 0].T, s=26, color=NARANJA, alpha=0.75, label="z < 0 → clase 0")
    # Sombrear las dos mitades evaluando z sobre una malla
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    zz = w[0] * xx + w[1] * yy + b0
    ax.contourf(xx, yy, zz, levels=[-99, 0, 99], colors=[NARANJA, AZUL], alpha=0.10)
    linea(ax, w, b0, color=TINTA_LINEA, linewidth=2.4)
    # w es perpendicular a la línea y apunta hacia la mitad z>0
    x_anc = np.array([0.4, -(w[0] * 0.4 + b0) / w[1]])
    wn = w / np.hypot(*w)
    ax.annotate("", xy=x_anc + wn * 1.1, xytext=x_anc,
                arrowprops=dict(arrowstyle="->", color=VERDE, linewidth=2.4))
    ax.text(*(x_anc + wn * 1.35), r"$\mathbf{w}$", color=VERDE, fontsize=13,
            fontweight="bold", ha="center")
    ax.set_title("z = w·x + b = 0 es una línea:\nparte el plano en dos mitades")
    ax.legend(loc="lower left", fontsize=8.5)

    # ── Panel 2: el bias desplaza la línea (sin girarla) ──
    ax = axes[1]
    wn = w / np.hypot(*w)              # dirección perpendicular a las líneas
    ang = np.degrees(np.arctan2(-w[0], w[1]))   # inclinación real de las líneas
    etiquetas_b = [(1.6, "--", 1.8, -2.2), (b0, "-", 2.4, -0.7), (-2.4, "--", 1.8, 0.9)]
    for b, estilo, ancho, x_etq in etiquetas_b:
        linea(ax, w, b, color=TINTA_LINEA, linestyle=estilo, linewidth=ancho)
        # Etiqueta anclada SOBRE su línea, desplazada un poco en perpendicular
        y_etq = -(w[0] * x_etq + b) / w[1]
        signo = "+" if b > 0 else "−"
        ax.text(x_etq + wn[0] * 0.34, y_etq + wn[1] * 0.34,
                f"b = {signo}{abs(b)}", fontsize=10, rotation=ang,
                ha="center", va="center",
                fontweight="bold" if b == b0 else "normal")
    # Doble flecha perpendicular: de la línea b=+1.6 a la línea b=−2.4
    p1 = -1.6 * wn / np.hypot(*w)
    p2 = 2.4 * wn / np.hypot(*w)
    ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                arrowprops=dict(arrowstyle="<->", color=ROJO, linewidth=2.0))
    ax.set_title("El bias b DESPLAZA la línea\n(mismos pesos: no gira)")

    # ── Panel 3: los pesos giran la línea ──
    ax = axes[2]
    for ang, estilo, ancho in [(35, "--", 1.8), (65, "-", 2.4), (115, "--", 1.8)]:
        rad = np.deg2rad(ang)
        linea(ax, np.array([np.cos(rad), np.sin(rad)]), b0,
              color=TINTA_LINEA, linestyle=estilo, linewidth=ancho)
    theta = np.linspace(np.deg2rad(-35), np.deg2rad(35), 50)
    ax.plot(1.7 * np.cos(theta), 1.7 * np.sin(theta), color=ROJO, linewidth=2.0)
    ax.annotate("", xy=(1.7 * np.cos(theta[-1]), 1.7 * np.sin(theta[-1])),
                xytext=(1.7 * np.cos(theta[-3]), 1.7 * np.sin(theta[-3])),
                arrowprops=dict(arrowstyle="->", color=ROJO, linewidth=2.0))
    ax.set_title("Los pesos w GIRAN la línea\n(cambian su orientación)")

    fig.suptitle("La geometría de una neurona: una frontera recta que w orienta y b desplaza",
                 fontweight="bold", y=1.04)
    fig.savefig(DESTINO / "neurona_bias_hiperplano.png")
    plt.close(fig)


def fig_xor() -> None:
    """XOR en dos paneles: (1) los cuatro puntos y varias líneas candidatas,
    cada una fallando en al menos un punto; (2) la solución de una MLP con
    dos neuronas ocultas: una BANDA diagonal (dos líneas cooperando), que es
    exactamente lo que una neurona sola no puede dibujar."""
    pts0 = np.array([[0, 0], [1, 1]])   # clase 0
    pts1 = np.array([[0, 1], [1, 0]])   # clase 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax in (ax1, ax2):
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-0.55, 1.55)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        # Los 4 puntos, grandes y con su etiqueta al lado
        ax.scatter(*pts1.T, s=260, color=AZUL, zorder=5)
        ax.scatter(*pts0.T, s=260, color=NARANJA, zorder=5)
        for x, y in pts0:
            ax.text(x, y, "0", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=6)
        for x, y in pts1:
            ax.text(x, y, "1", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=6)

    # ── Panel 1: cada línea recta deja al menos un punto mal ──
    xs = np.linspace(-0.55, 1.55, 2)
    ax1.plot(xs, 0.5 + 0 * xs, "--", color=GRIS, linewidth=2.0)
    ax1.text(1.52, 0.56, "✗", fontsize=15, color=ROJO, fontweight="bold")
    ax1.plot(0.5 + 0 * xs, xs, "--", color=GRIS, linewidth=2.0)
    ax1.text(0.56, 1.44, "✗", fontsize=15, color=ROJO, fontweight="bold")
    ax1.plot(xs, xs - 0.5, "--", color=GRIS, linewidth=2.0)
    ax1.text(1.30, 0.92, "✗", fontsize=15, color=ROJO, fontweight="bold")
    ax1.plot(xs, 1.5 - xs, "--", color=GRIS, linewidth=2.0)
    ax1.text(0.12, 1.44, "✗", fontsize=15, color=ROJO, fontweight="bold")
    ax1.set_title("Ninguna recta funciona: toda línea\ndeja algún punto del lado equivocado", fontsize=11.5)

    # ── Panel 2: dos neuronas ocultas = una banda diagonal ──
    xx, yy = np.meshgrid(np.linspace(-0.55, 1.55, 300),
                         np.linspace(-0.55, 1.55, 300))
    # MLP mínima hecha a mano: h1 = "x+y > 0.5" (OR), h2 = "x+y > 1.5" (AND),
    # salida = h1 AND (NO h2) → la banda 0.5 < x+y < 1.5, que es XOR.
    banda = ((xx + yy > 0.5) & (xx + yy < 1.5)).astype(float)
    ax2.contourf(xx, yy, banda, levels=[0.5, 1.5], colors=[AZUL], alpha=0.16)
    ax2.plot(xs, 0.5 - xs, "--", color=TINTA_LINEA, linewidth=2.0)
    ax2.plot(xs, 1.5 - xs, "--", color=TINTA_LINEA, linewidth=2.0)
    ax2.text(0.5, 0.62, "zona clase 1", fontsize=10, color=AZUL,
             rotation=-45, ha="center", va="center", fontweight="bold")
    ax2.text(-0.30, 0.62, "neurona A:\nx+y > 0.5", fontsize=8.5, color=GRIS,
             rotation=-45, ha="center", va="center")
    ax2.text(1.28, 0.42, "neurona B:\nx+y > 1.5", fontsize=8.5, color=GRIS,
             rotation=-45, ha="center", va="center")
    ax2.set_title("Dos neuronas ocultas SÍ pueden: juntas\ndibujan una banda — eso ya no es una recta", fontsize=11.5)

    fig.subplots_adjust(wspace=0.3)
    fig.suptitle("XOR: clase 1 (azul) si exactamente UNA de las dos entradas está activa",
                 fontweight="bold", y=1.04)
    fig.savefig(DESTINO / "xor_no_separable.png")
    plt.close(fig)


def fig_capa_densa() -> None:
    """La capa densa como multiplicación de matrices, celda por celda.

    H (B, d_in) por W (d_in, d_out) más b produce Z (B, d_out); φ se aplica
    elemento a elemento. Se resalta UNA fila de H (una muestra) y UNA
    columna de W (una neurona): su producto punto es UNA celda de Z — la
    neurona j evaluada en la muestra i. Números concretos incluidos.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    ax.grid(False)

    L = 0.9                    # lado de celda
    FILA_H, COL_W = 1, 0       # fila de H y columna de W resaltadas
    vals_h = ["2", "1", "0"]   # la muestra resaltada
    vals_w = ["1", "−1", "3"]  # la neurona resaltada
    B, DIN, DOUT = 4, 3, 2

    def grid(x0, y0, filas, cols, color, textos=None):
        """Cuadrícula con origen (x0, y0) arriba-izquierda; devuelve centro."""
        for i in range(filas):
            for j in range(cols):
                ax.add_patch(Rectangle((x0 + j * L, y0 - (i + 1) * L), L, L,
                                       facecolor=color, edgecolor=AZUL,
                                       linewidth=1.2, zorder=2))
                if textos and textos.get((i, j)):
                    ax.text(x0 + j * L + L / 2, y0 - (i + 1) * L + L / 2,
                            textos[(i, j)], ha="center", va="center",
                            fontsize=10, zorder=4)
        return x0 + cols * L / 2

    def marco(x0, y0, filas, cols, color):
        """Resalta un bloque (fila o columna o celda) con borde grueso."""
        ax.add_patch(Rectangle((x0, y0 - filas * L), cols * L, filas * L,
                               facecolor="none", edgecolor=color,
                               linewidth=3.0, zorder=3))

    def etiqueta(xc, nombre, shape, nota=None):
        ax.text(xc, 2.55, nombre, ha="center", fontsize=12.5, fontweight="bold")
        ax.text(xc, 1.95, shape, ha="center", fontsize=10.5,
                family="monospace", color=AZUL, fontweight="bold")
        if nota:
            ax.text(xc, 1.35, nota, ha="center", fontsize=9, color=GRIS,
                    style="italic")

    y_top = 7.0                # borde superior de H y Z

    # ── H⁽ˡ⁻¹⁾: (B, d_in), cada fila una muestra ──
    tx = {(FILA_H, j): v for j, v in enumerate(vals_h)}
    xc = grid(0.9, y_top, B, DIN, "#D6E9F8", tx)
    marco(0.9, y_top - FILA_H * L, 1, DIN, NARANJA)
    etiqueta(xc, "H⁽ˡ⁻¹⁾", "(B, d_in)", "cada FILA es\nuna muestra")
    ax.text(0.55, y_top - FILA_H * L - L / 2, "muestra i →", ha="right",
            va="center", fontsize=9, color=NARANJA, fontweight="bold")

    ax.text(4.25, y_top - B * L / 2, "×", fontsize=22, ha="center", va="center")

    # ── W⁽ˡ⁾: (d_in, d_out), cada columna una neurona ──
    y_w = y_top - (B - DIN) * L / 2    # centrar verticalmente respecto a H
    tx = {(i, COL_W): v for i, v in enumerate(vals_w)}
    xc = grid(5.0, y_w, DIN, DOUT, "#CBE8DC", tx)
    marco(5.0 + COL_W * L, y_w, DIN, 1, VERDE)
    etiqueta(xc, "W⁽ˡ⁾", "(d_in, d_out)", "cada COLUMNA es\nuna neurona")
    ax.text(5.0 + COL_W * L + L / 2, y_w + 0.35, "neurona j\n↓", ha="center",
            va="bottom", fontsize=9, color=VERDE, fontweight="bold")

    ax.text(7.6, y_top - B * L / 2, "+", fontsize=22, ha="center", va="center")

    # ── b⁽ˡ⁾: un bias por neurona ──
    y_b = y_top - B * L / 2 + L / 2
    xc = grid(8.1, y_b, 1, DOUT, "#EFEFEF", {(0, COL_W): "0.5"})
    etiqueta(xc, "b⁽ˡ⁾", "(d_out,)", "un bias\npor neurona")

    ax.text(10.7, y_top - B * L / 2, "=", fontsize=22, ha="center", va="center")

    # ── Z⁽ˡ⁾: (B, d_out) con la celda resultado resaltada ──
    xc = grid(11.2, y_top, B, DOUT, "#FDF3D0", {(FILA_H, COL_W): "1.5"})
    marco(11.2 + COL_W * L, y_top - FILA_H * L, 1, 1, ROJO)
    etiqueta(xc, "Z⁽ˡ⁾", "(B, d_out)")

    # ── φ elemento a elemento → H⁽ˡ⁾ ──
    ax.add_patch(FancyArrowPatch((13.35, y_top - B * L / 2),
                                 (14.15, y_top - B * L / 2),
                                 arrowstyle="->", mutation_scale=15,
                                 color=NARANJA, linewidth=2.2))
    ax.text(13.75, y_top - B * L / 2 + 0.55, "φ", fontsize=15, ha="center",
            color=NARANJA, fontweight="bold")
    xc = grid(14.4, y_top, B, DOUT, "#F9E3C8")
    etiqueta(xc, "H⁽ˡ⁾ = φ(Z⁽ˡ⁾)", "(B, d_out)", "entrada de la\ncapa siguiente")

    # ── La celda roja, explicada con los números de la figura ──
    ax.text(8.3, 0.45,
            "celda roja = (fila naranja) · (columna verde) + bias "
            "= 2·1 + 1·(−1) + 0·3 + 0.5 = 1.5   —   la neurona j evaluada en la muestra i",
            ha="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDF3D0",
                      edgecolor=ROJO, linewidth=1.4))

    ax.text(8.3, 9.15,
            "Una capa densa es UNA multiplicación de matrices: evalúa las "
            "d_out neuronas sobre las B muestras de un solo golpe",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(8.3, 8.45,
            "El superíndice (l) numera la capa · φ se aplica número a número "
            "· por esto la GPU vuela: todo es un solo producto de matrices",
            ha="center", fontsize=10, color=GRIS)

    fig.savefig(DESTINO / "capa_densa_matmul.png")
    plt.close(fig)


def fig_softmax_pasos() -> None:
    """El mecanismo de softmax en 3 pasos con números reales: logits
    (pueden ser negativos) → e^z (todo positivo, diferencias agrandadas)
    → dividir por la suma (probabilidades que suman 1)."""
    clases = ["gato", "perro", "pez"]
    z = np.array([2.0, 1.0, -1.0])
    ez = np.exp(z)
    p = ez / ez.sum()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    datos = [
        (z, "1 · Logits z\n(salida cruda de la red)",
         "pueden ser negativos\ny no suman nada especial"),
        (ez, "2 · Exponenciar: e^z\n(todo positivo)",
         "las diferencias\nse agrandan"),
        (p, "3 · Dividir por la suma\n(probabilidades)",
         f"suman {p.sum():.2f} ✓\n¡una distribución!"),
    ]
    for ax, (vals, titulo, nota) in zip(axes, datos):
        colores = [AZUL if v == vals.max() else CELESTE for v in vals]
        barras = ax.bar(clases, vals, color=colores)
        ax.set_title(titulo, fontsize=11.5)
        ax.axhline(0, color=GRIS, linewidth=0.9)
        for barra, v in zip(barras, vals):
            va = "bottom" if v >= 0 else "top"
            dy = 0.04 * max(abs(vals.max()), 1) * (1 if v >= 0 else -1)
            ax.text(barra.get_x() + barra.get_width() / 2, v + dy,
                    f"{v:.2f}", ha="center", va=va, fontsize=10,
                    fontweight="bold")
        ax.text(0.97, 0.86, nota, transform=ax.transAxes, fontsize=9,
                ha="right", color=GRIS, style="italic")
        ax.margins(y=0.22)

    # Flechas entre paneles, con la operación que se aplica
    for x_fig, texto in [(0.365, "e^z"), (0.655, "÷ suma")]:
        fig.text(x_fig, 0.5, "→", fontsize=24, ha="center", va="center",
                 color=NARANJA, fontweight="bold")
        fig.text(x_fig, 0.60, texto, fontsize=11, ha="center",
                 color=NARANJA, fontweight="bold")

    fig.suptitle("softmax paso a paso: de puntajes sueltos a porciones de una torta",
                 fontweight="bold", y=1.04)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(DESTINO / "softmax_pasos.png")
    plt.close(fig)


def fig_losses() -> None:
    """Las dos losses del curso como CURVAS DE CASTIGO: MSE castiga el error
    en cuadrático (errores grandes duelen desproporcionadamente); BCE castiga
    con −log(p) la probabilidad asignada a la respuesta correcta (estar
    confiado y equivocado explota)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # ── MSE: parábola del error ──
    err = np.linspace(-3, 3, 300)
    ax1.plot(err, err**2, color=AZUL, linewidth=2.6)
    for e, nota, xytext in [
        (0.5, "error 0.5 → castigo 0.25", (0.9, 0.9)),
        (2.0, "error 2.0 → castigo 4.0\n(4× el error, 16× el castigo)", (-2.6, 5.6)),
    ]:
        ax1.scatter([e], [e**2], s=70, color=ROJO, zorder=5)
        ax1.annotate(nota, xy=(e, e**2), xytext=xytext,
                     fontsize=9.5, arrowprops=dict(arrowstyle="->", color=ROJO))
    ax1.set(title="MSE: el castigo crece al CUADRADO del error",
            xlabel="error  (ŷ − y)", ylabel="castigo  (ŷ − y)²")
    ax1.axhline(0, color=GRIS, linewidth=0.8)
    ax1.axvline(0, color=GRIS, linewidth=0.8)

    # ── BCE: −log(p) sobre la probabilidad de la respuesta correcta ──
    p = np.linspace(0.01, 1.0, 300)
    ax2.plot(p, -np.log(p), color=AZUL, linewidth=2.6)
    for pi, nota, xytext in [
        (0.9, "p=0.9 → castigo 0.11\n(casi seguro y correcto: apenas duele)", (0.42, 1.1)),
        (0.1, "p=0.1 → castigo 2.3\n(confiado y equivocado: duele MUCHO)", (0.22, 3.1)),
    ]:
        ax2.scatter([pi], [-np.log(pi)], s=70, color=ROJO, zorder=5)
        ax2.annotate(nota, xy=(pi, -np.log(pi)), xytext=xytext, fontsize=9.5,
                     arrowprops=dict(arrowstyle="->", color=ROJO))
    ax2.set(title="Cross-entropy: −log(p) explota cuando p → 0",
            xlabel="probabilidad asignada a la respuesta CORRECTA",
            ylabel="castigo  −log(p)", ylim=(0, 5))

    fig.suptitle("Una loss es una curva de castigo: dime cuánto te equivocaste "
                 "y te digo cuánto duele", fontweight="bold", y=1.05)
    fig.savefig(DESTINO / "losses_castigo.png")
    plt.close(fig)


def fig_paisaje_perdida() -> None:
    """El "paisaje montañoso" de la loss, literal: una superficie 3D con
    montañas y un valle sobre dos parámetros (θ₁, θ₂), la bolita del
    descenso bajando en contra del gradiente, y el MISMO paisaje visto
    desde arriba como mapa de contornos — el puente hacia cómo se dibuja
    el paisaje en 2D en el resto del curso y en el simulador."""

    def f(x, y):
        # Un cuenco suave (el valle) + dos colinas gaussianas (las montañas)
        base = 0.12 * (x**2 + 1.4 * y**2)
        colina1 = 3.2 * np.exp(-((x + 1.7) ** 2 + (y - 1.4) ** 2) / 1.5)
        colina2 = 2.6 * np.exp(-((x - 2.1) ** 2 + (y + 1.5) ** 2) / 2.0)
        return base + colina1 + colina2

    def grad(x, y, h=1e-4):
        """Gradiente numérico por diferencias centradas."""
        return np.array([
            (f(x + h, y) - f(x - h, y)) / (2 * h),
            (f(x, y + h) - f(x, y - h)) / (2 * h),
        ])

    # La ruta del descenso: θ ← θ − η∇L desde una ladera
    p = np.array([-2.7, 2.5])
    ruta = [p.copy()]
    for _ in range(260):
        p = p - 0.14 * grad(*p)
        ruta.append(p.copy())
    ruta = np.array(ruta)
    z_ruta = f(ruta[:, 0], ruta[:, 1])

    xx, yy = np.meshgrid(np.linspace(-3.8, 3.8, 200),
                         np.linspace(-3.1, 3.1, 200))
    zz = f(xx, yy)

    fig = plt.figure(figsize=(13.5, 5.6))

    # ── Panel 1: el paisaje en 3D ──
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # Sin esto, mplot3d ordena por profundidad y la superficie tapa a la
    # bolita, la estrella y la ruta; con zorder manual quedan encima.
    ax1.computed_zorder = False
    ax1.plot_surface(xx, yy, zz, cmap="Blues_r", rstride=2, cstride=2,
                     linewidth=0, antialiased=True, alpha=0.92, zorder=1)
    # La ruta sobre la superficie (levemente elevada para que no se hunda)
    ax1.plot(ruta[:, 0], ruta[:, 1], z_ruta + 0.06, color=ROJO,
             linewidth=2.6, zorder=10)
    ax1.scatter(*ruta[0], z_ruta[0] + 0.10, s=90, color=ROJO,
                edgecolor="white", linewidth=1.5, zorder=11,
                depthshade=False)
    ax1.scatter(*ruta[-1], z_ruta[-1] + 0.10, s=160, color="#f5c400",
                marker="*", edgecolor="#333", linewidth=0.8, zorder=11,
                depthshade=False)
    g0 = grad(*ruta[0])
    g0n = g0 / np.linalg.norm(g0)
    ax1.text(ruta[0][0], ruta[0][1] + 0.25, z_ruta[0] + 1.0,
             "inicio", color=ROJO, fontsize=10, fontweight="bold",
             ha="center")
    ax1.text(ruta[-1][0] + 0.3, ruta[-1][1] - 0.4, z_ruta[-1] + 0.7,
             "mínimo\n(el valle)", fontsize=10, fontweight="bold",
             color="#7a6200")
    ax1.set_xlabel("θ₁ (un parámetro)", fontsize=9.5, labelpad=2)
    ax1.set_ylabel("θ₂ (otro parámetro)", fontsize=9.5, labelpad=2)
    # El zlabel nativo de mplot3d queda fuera del bbox "tight": anotarlo
    # como texto 2D anclado a los ejes, que sí se incluye en el recorte.
    ax1.text2D(-0.04, 0.60, "L(θ) = altura", transform=ax1.transAxes,
               rotation=90, fontsize=10, ha="center", va="center")
    ax1.tick_params(labelsize=7.5, pad=1)
    ax1.view_init(elev=33, azim=-120)
    ax1.set_title("El paisaje en 3D: montañas, valle\ny la ruta θ ← θ − η·∇L",
                  fontsize=11.5)

    # ── Panel 2: el mismo paisaje visto desde arriba ──
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(xx, yy, zz, levels=26, cmap="Blues_r", alpha=0.9)
    ax2.contour(xx, yy, zz, levels=13, colors="white", linewidths=0.6)
    ax2.plot(ruta[:, 0], ruta[:, 1], color=ROJO, linewidth=2.4)
    ax2.scatter(*ruta[0], s=80, color=ROJO, edgecolor="white",
                linewidth=1.5, zorder=5)
    ax2.scatter(*ruta[-1], s=200, color="#f5c400", marker="*",
                edgecolor="#333", linewidth=0.8, zorder=5)
    # Las mismas dos flechas, ahora en el mapa
    ax2.annotate("", xy=ruta[0] + g0n * 1.0, xytext=ruta[0],
                 arrowprops=dict(arrowstyle="->", color=VERDE, linewidth=2.4))
    ax2.text(*(ruta[0] + g0n * 1.0 + [0.15, 0.2]), "∇L", color=VERDE,
             fontsize=11, fontweight="bold")
    ax2.annotate("", xy=ruta[0] - g0n * 1.0, xytext=ruta[0],
                 arrowprops=dict(arrowstyle="->", color=ROJO, linewidth=2.4))
    ax2.text(*(ruta[0] - g0n * 1.0 + [-1.35, -0.15]), "−η·∇L", color=ROJO,
             fontsize=11, fontweight="bold")
    ax2.text(-1.7, 1.35, "montaña", fontsize=9, color="#334",
             ha="center", style="italic")
    ax2.text(2.0, -1.5, "montaña", fontsize=9, color="#334",
             ha="center", style="italic")
    ax2.text(ruta[-1][0] + 0.3, ruta[-1][1] - 0.55, "el valle",
             fontsize=10, fontweight="bold", color="white")
    ax2.set_aspect("equal")
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("θ₁", fontsize=10)
    ax2.set_ylabel("θ₂", fontsize=10)
    ax2.set_title("El MISMO paisaje visto desde arriba:\nun mapa de contornos "
                  "(cada anillo = curva de nivel)", fontsize=11.5)

    fig.suptitle("La loss es un paisaje: el gradiente apunta cuesta arriba "
                 "y el descenso camina en contra", fontweight="bold", y=1.02)
    fig.savefig(DESTINO / "paisaje_perdida.png")
    plt.close(fig)


def fig_matriz_confusion() -> None:
    """Matriz de confusión 2×2 con números concretos, y precision, recall y
    F1 calculados de ELLA, con las celdas que usa cada métrica señaladas."""
    from matplotlib.patches import Rectangle

    # Ejemplo: detector de spam sobre 100 correos
    TP, FN, FP, TN = 40, 10, 5, 45
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * prec * rec / (prec + rec)

    fig, (ax, axm) = plt.subplots(1, 2, figsize=(12, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.25]})

    # ── La matriz ──
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    celdas = [
        (0, 1, TP, "TP = 40", "spam marcado spam ✓", "#CBE8DC"),
        (1, 1, FN, "FN = 10", "spam que se escapó ✗", "#F6D9CE"),
        (0, 0, FP, "FP = 5", "bueno marcado spam ✗", "#F6D9CE"),
        (1, 0, TN, "TN = 45", "bueno dejado pasar ✓", "#CBE8DC"),
    ]
    for cx, cy, n, sigla, nota, color in celdas:
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=color,
                               edgecolor=TINTA_LINEA, linewidth=1.6))
        ax.text(cx + 0.5, cy + 0.62, sigla, ha="center", fontsize=13,
                fontweight="bold")
        ax.text(cx + 0.5, cy + 0.30, nota, ha="center", fontsize=8.2,
                color="#333333")
    ax.text(0.5, 2.12, "predijo: spam", ha="center", fontsize=10.5,
            fontweight="bold")
    ax.text(1.5, 2.12, "predijo: no spam", ha="center", fontsize=10.5,
            fontweight="bold")
    ax.text(-0.10, 1.5, "real:\nspam", ha="right", va="center", fontsize=10.5,
            fontweight="bold")
    ax.text(-0.10, 0.5, "real:\nno spam", ha="right", va="center", fontsize=10.5,
            fontweight="bold")
    ax.text(1.0, -0.22, "100 correos · la diagonal verde son los aciertos",
            ha="center", fontsize=10, color=GRIS, style="italic")

    # ── Las métricas calculadas de la matriz ──
    axm.axis("off")
    axm.grid(False)
    lineas = [
        ("Precision", "de lo que marqué spam, ¿cuánto era spam?",
         f"TP / (TP + FP) = {TP} / {TP + FP} = {prec:.2f}", VERDE),
        ("Recall", "del spam real, ¿cuánto atrapé?",
         f"TP / (TP + FN) = {TP} / {TP + FN} = {rec:.2f}", AZUL),
        ("F1", "promedio armónico: solo es alto si AMBAS lo son",
         f"2·P·R / (P + R) = {f1:.2f}", NARANJA),
        ("Accuracy", "aciertos totales — engañosa si hay desbalance",
         f"(TP + TN) / 100 = {(TP + TN) / 100:.2f}", GRIS),
    ]
    for i, (nombre, pregunta, formula, color) in enumerate(lineas):
        y = 0.88 - i * 0.24
        axm.text(0.02, y, nombre, fontsize=13, fontweight="bold", color=color,
                 transform=axm.transAxes)
        axm.text(0.30, y, pregunta, fontsize=9.5, style="italic",
                 color="#333333", transform=axm.transAxes)
        axm.text(0.30, y - 0.09, formula, fontsize=10.5, family="monospace",
                 transform=axm.transAxes)
    axm.set_title("Las métricas salen de la matriz", fontsize=11.5)

    fig.suptitle("Matriz de confusión → precision, recall y F1 (detector de spam)",
                 fontweight="bold", y=1.02)
    fig.savefig(DESTINO / "matriz_confusion_metricas.png")
    plt.close(fig)


def fig_lora() -> None:
    """LoRA a escala: la matriz W congelada (d×d) versus las dos matrices
    flacas B (d×r) y A (r×d) que aprenden la corrección. Con d=768 y r=8,
    los rectángulos a escala real hacen visible el ahorro (~2%)."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    d, r = 768, 8
    D = 7.0                              # lado de W en unidades de lienzo
    R = max(D * r / d, 0.30)             # grosor de B y A (mínimo visible)
    y0 = 1.6                             # base de los rectángulos
    ymed = y0 + D / 2                    # línea media vertical

    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 28.5)
    ax.set_ylim(0, 11.3)
    ax.axis("off")
    ax.grid(False)

    # ── W congelada ──
    ax.add_patch(Rectangle((0.8, y0), D, D, facecolor="#E3E6EA",
                           edgecolor=GRIS, linewidth=1.8, hatch="///", zorder=2))
    ax.text(0.8 + D / 2, ymed, "W\n(congelada:\nno se toca)",
            ha="center", va="center", fontsize=12, color="#444444", zorder=3)
    ax.text(0.8 + D / 2, y0 - 0.6, f"{d}×{d} = {d * d:,} parámetros",
            ha="center", fontsize=9.5, family="monospace")

    ax.text(8.7, ymed, "+", fontsize=26, ha="center", va="center")

    # ── B (flaca vertical) × A (flaca horizontal), centradas en la línea media ──
    ax.add_patch(Rectangle((9.7, y0), R, D, facecolor="#CBE8DC",
                           edgecolor=VERDE, linewidth=1.8, zorder=2))
    ax.text(9.7 + R / 2, y0 + D + 0.42, "B", ha="center", fontsize=13,
            fontweight="bold", color=VERDE)
    ax.text(9.7 + R / 2, y0 - 0.6, f"{d}×{r}", ha="center", fontsize=9.5,
            family="monospace")

    ax.text(11.4, ymed, "×", fontsize=20, ha="center", va="center")

    ax.add_patch(Rectangle((12.4, ymed - R / 2), D, R, facecolor="#D6E9F8",
                           edgecolor=AZUL, linewidth=1.8, zorder=2))
    ax.text(12.4 + D + 0.45, ymed, "A", va="center", fontsize=13,
            fontweight="bold", color=AZUL)
    ax.text(12.4 + D / 2, ymed - R / 2 - 0.6, f"{r}×{d}", ha="center",
            fontsize=9.5, family="monospace")

    # ── ΔW: mismo tamaño que W, pero descrito con pocos números ──
    ax.add_patch(FancyArrowPatch((20.4, ymed), (21.2, ymed),
                                 arrowstyle="->", mutation_scale=15,
                                 color=NARANJA, linewidth=2.2))
    ax.text(20.8, ymed + 0.6, "=", fontsize=15, ha="center",
            color=NARANJA, fontweight="bold")
    ax.add_patch(Rectangle((21.5, y0), D, D, facecolor="#FDF3D0",
                           edgecolor=NARANJA, linewidth=1.8, zorder=2))
    ax.text(21.5 + D / 2, ymed, "ΔW = B·A\n(la corrección:\ndel tamaño de W,\n"
            "pero descrita con\nmuy pocos números)",
            ha="center", va="center", fontsize=10.5, zorder=3)

    n_lora = r * (d + d)
    ax.text(14.25, 10.8,
            "LoRA: W queda congelada y la corrección ΔW = B·A se aprende "
            "con dos matrices flacas",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(14.25, 10.05,
            f"entrenables: {d}×{r} + {r}×{d} = {n_lora:,} parámetros "
            f"= {100 * n_lora / (d * d):.1f}% de los {d * d:,} de W "
            f"(grosores a escala real)",
            ha="center", fontsize=10.5, color=GRIS)
    ax.text(14.25, 0.3,
            'r = 8 es el "rango": el ancho de las matrices flacas. '
            "Subir r da más capacidad de corrección a cambio de más parámetros.",
            ha="center", fontsize=9.5, style="italic", color=GRIS)

    fig.savefig(DESTINO / "lora_matrices.png")
    plt.close(fig)


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
                 fontweight="bold", y=1.06)
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
    # loc fijo: el 'best' automático la ponía encima de la nota del panel
    axes[0].legend(loc="lower left")
    fig.suptitle("Diagnóstico por curvas de aprendizaje", fontweight="bold", y=1.04)
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
                 y=1.06,
                 fontweight="bold")
    fig.savefig(DESTINO / "padding_stride.png")
    plt.close(fig)


def fig_cnn_anatomia() -> None:
    """La anatomía completa de una CNN en una sola imagen: entrada →
    bloques [conv+ReLU → pooling] que extraen features → flatten →
    capas densas (la MLP de la Sesión 1) → softmax → probabilidades.
    Es el mapa que abre la Sesión 2: las dos mitades (extraer features /
    clasificar) quedan delimitadas con llaves abajo."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(14.5, 6.0))
    ax.set_xlim(0, 30)
    ax.set_ylim(1.4, 12)
    ax.axis("off")
    ax.grid(False)

    y_mid = 6.6          # eje vertical de la línea de ensamblaje
    Y_TITULO = 9.55      # banda de títulos de etapa
    Y_SUB = 4.15         # banda de sub-etiquetas

    def titulo_etapa(xc, texto):
        ax.text(xc, Y_TITULO, texto, ha="center", fontsize=10.5,
                fontweight="bold", color="#1a2332")

    def sub_etapa(xc, texto):
        ax.text(xc, Y_SUB, texto, ha="center", va="top", fontsize=8.5,
                color=GRIS, style="italic")

    def flecha(x0, x1, texto=None):
        ax.add_patch(FancyArrowPatch((x0, y_mid), (x1, y_mid),
                                     arrowstyle="->", mutation_scale=14,
                                     color=NARANJA, linewidth=2.0, zorder=2))
        if texto:
            ax.text((x0 + x1) / 2, y_mid + 0.55, texto, ha="center",
                    fontsize=8, color=NARANJA, fontweight="bold")

    def pila(x0, n, lado, color, off=0.28):
        """n cuadrados apilados en diagonal (feature maps); devuelve
        (x_centro, x_fin)."""
        for k in range(n - 1, -1, -1):    # de atrás hacia adelante
            ax.add_patch(Rectangle(
                (x0 + k * off, y_mid - lado / 2 + k * off * 0.75),
                lado, lado, facecolor=color, edgecolor=AZUL,
                linewidth=1.1, zorder=3 + (n - k)))
        ancho = lado + (n - 1) * off
        return x0 + ancho / 2, x0 + ancho

    # ── Entrada: una "camiseta" en pixel-art ──
    rng = np.random.default_rng(3)
    img = np.full((14, 14), 0.06)
    img[2:4, 1:13] = 0.92                     # hombros y mangas
    img[4:6, 1:4] = 0.92                      # manga izquierda
    img[4:6, 10:13] = 0.92                    # manga derecha
    img[4:12, 4:10] = 0.92                    # cuerpo
    img[2, 6:8] = 0.06                        # cuello
    img += rng.uniform(0, 0.06, img.shape)    # textura leve
    ax.imshow(img, extent=(0.7, 3.7, y_mid - 1.5, y_mid + 1.5),
              cmap="gray", vmin=0, vmax=1, zorder=2)
    titulo_etapa(2.2, "Entrada")
    sub_etapa(2.2, "imagen 28×28\n(1 canal)")
    # kernel recorriendo la imagen
    ax.add_patch(Rectangle((1.15, y_mid - 1.0), 0.64, 0.64, facecolor="none",
                           edgecolor="#f5c400", linewidth=2.2, zorder=4))
    ax.plot([1.79, 4.95], [y_mid - 0.68, y_mid - 1.1], linestyle="--",
            color="#b89000", linewidth=1.1, zorder=2)
    ax.text(1.45, y_mid - 1.75, "kernel 3×3", fontsize=8, color="#8a6d00",
            ha="center", fontweight="bold")

    # ── Bloque 1: conv + ReLU → pooling ──
    flecha(3.85, 4.85)
    c1, fin = pila(4.95, 4, 2.5, "#BBDEF5")
    titulo_etapa(c1, "Convolución + ReLU")
    sub_etapa(c1, "feature maps\n(uno por filtro)")
    flecha(fin + 0.2, fin + 1.15)
    p1, fin = pila(fin + 1.25, 4, 1.7, "#A7D9C9", off=0.26)
    titulo_etapa(p1, "MaxPool 2×2")
    sub_etapa(p1, "mitad de\nresolución")

    # ── Bloque 2: conv + ReLU → pooling (más filtros, más chicos) ──
    flecha(fin + 0.2, fin + 1.05)
    c2, fin = pila(fin + 1.15, 7, 1.5, "#BBDEF5", off=0.19)
    titulo_etapa(c2, "Conv + ReLU")
    sub_etapa(c2, "más filtros, patrones\nmás abstractos")
    flecha(fin + 0.2, fin + 1.05)
    p2, fin = pila(fin + 1.15, 7, 1.0, "#A7D9C9", off=0.15)
    titulo_etapa(p2, "MaxPool")

    # ── Flatten: los mapas se estiran a un vector ──
    flecha(fin + 0.2, fin + 1.05, "flatten")
    x_flat = fin + 1.35
    ys_flat = np.linspace(y_mid - 1.9, y_mid + 1.9, 9)
    for y in ys_flat:
        ax.add_patch(Rectangle((x_flat, y - 0.19), 0.42, 0.38,
                               facecolor="#EFEFEF", edgecolor=AZUL,
                               linewidth=1.0, zorder=3))
    titulo_etapa(x_flat + 0.21, "Flatten")
    sub_etapa(x_flat + 0.21, "vector")

    # ── Capas densas (la MLP de la Sesión 1) ──
    x_d1, x_d2 = x_flat + 2.0, x_flat + 3.6
    ys_d1 = np.linspace(y_mid - 2.0, y_mid + 2.0, 6)
    ys_d2 = np.linspace(y_mid - 1.2, y_mid + 1.2, 4)
    for y0 in ys_flat:                        # conexiones flatten → capa 1
        for y1 in ys_d1:
            ax.plot([x_flat + 0.42, x_d1], [y0, y1], color="#ccd4de",
                    linewidth=0.4, zorder=1)
    for y0 in ys_d1:                          # conexiones capa 1 → capa 2
        for y1 in ys_d2:
            ax.plot([x_d1, x_d2], [y0, y1], color="#ccd4de",
                    linewidth=0.4, zorder=1)
    for x, ys in [(x_d1, ys_d1), (x_d2, ys_d2)]:
        for y in ys:
            ax.add_patch(Circle((x, y), 0.26, facecolor=CELESTE,
                                edgecolor=AZUL, linewidth=1.2, zorder=3))
    xc_dense = (x_d1 + x_d2) / 2
    titulo_etapa(xc_dense, "Capas densas")
    sub_etapa(xc_dense, "la MLP de la\nSesión 1")

    # ── Salida: softmax → una probabilidad por clase ──
    x_bar = x_d2 + 2.6
    ax.add_patch(FancyArrowPatch((x_d2 + 0.45, y_mid), (x_bar - 1.55, y_mid),
                                 arrowstyle="->", mutation_scale=14,
                                 color=NARANJA, linewidth=2.0, zorder=2))
    ax.text(x_d2 + 1.0, y_mid - 0.62, "softmax", ha="center", fontsize=8,
            color=NARANJA, fontweight="bold")
    clases = [("camiseta", 0.72, NARANJA), ("zapato", 0.19, CELESTE),
              ("bolso", 0.09, CELESTE)]
    for fila, (nombre, p, color) in enumerate(clases):
        y = y_mid + 1.0 - fila * 1.0
        ax.add_patch(Rectangle((x_bar, y - 0.28), 2.6 * p, 0.56,
                               facecolor=color, edgecolor="none", zorder=3))
        ax.text(x_bar - 0.12, y, nombre, ha="right", va="center", fontsize=9)
        ax.text(x_bar + 2.6 * p + 0.12, y, f"{p:.2f}", ha="left",
                va="center", fontsize=9, fontweight="bold")
    titulo_etapa(x_bar + 1.1, "Probabilidades")
    sub_etapa(x_bar + 1.1, "suman 1\n(softmax, Sesión 1)")

    # ── Las dos mitades, delimitadas abajo ──
    def llave(x0, x1, texto):
        ax.plot([x0, x0, x1, x1], [2.95, 2.7, 2.7, 2.95],
                color="#1a2332", linewidth=1.4)
        ax.text((x0 + x1) / 2, 2.25, texto, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="#1a2332")

    llave(4.7, x_flat - 0.6, "EXTRACCIÓN DE FEATURES — aprende QUÉ mirar (§3)")
    llave(x_flat - 0.2, x_bar + 2.9, "CLASIFICACIÓN — decide QUÉ ES (Sesión 1)")

    ax.text(14.5, 11.3,
            "La anatomía de una CNN: de píxeles a probabilidades",
            ha="center", fontsize=13.5, fontweight="bold")
    fig.savefig(DESTINO / "cnn_anatomia.png")
    plt.close(fig)


def fig_pooling() -> None:
    """Max pooling vs average pooling sobre la MISMA entrada 4×4:
    cada ventana 2×2 de color produce UNA celda de la salida 2×2."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    X = np.array([[1, 3, 2, 0],
                  [5, 6, 1, 2],
                  [7, 2, 9, 4],
                  [3, 1, 4, 8]], float)
    tintes = [CELESTE, VERDE, NARANJA, MORADO]   # TL, TR, BL, BR

    def cuadrante(i, j):
        return (i // 2) * 2 + (j // 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (nombre, reduce) in zip(axes, [
        ("Max pooling 2×2 — se queda el pico", np.max),
        ("Average pooling 2×2 — el promedio", np.mean),
    ]):
        ax.set_xlim(-0.5, 9.2)
        ax.set_ylim(-0.6, 4.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.grid(False)

        # Entrada 4×4 con tinte por ventana
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((j, 3 - i), 1, 1,
                                       facecolor=tintes[cuadrante(i, j)],
                                       alpha=0.30, edgecolor="#888",
                                       linewidth=0.8))
                ax.text(j + 0.5, 3 - i + 0.5, f"{X[i, j]:.0f}", ha="center",
                        va="center", fontsize=11)
        # Bordes gruesos de las 4 ventanas
        for qi in range(2):
            for qj in range(2):
                ax.add_patch(Rectangle((qj * 2, 2 - qi * 2), 2, 2,
                                       facecolor="none",
                                       edgecolor=tintes[qi * 2 + qj],
                                       linewidth=2.6))
        ax.text(2, 4.25, "feature map 4×4", ha="center", fontsize=9.5,
                color=GRIS)

        ax.add_patch(FancyArrowPatch((4.5, 2), (5.6, 2), arrowstyle="->",
                                     mutation_scale=16, color=NARANJA,
                                     linewidth=2.2))

        # Salida 2×2: una celda por ventana, mismo color
        for qi in range(2):
            for qj in range(2):
                valor = reduce(X[qi * 2:qi * 2 + 2, qj * 2:qj * 2 + 2])
                ax.add_patch(Rectangle((6.0 + qj * 1.3, 1.05 + (1 - qi) * 1.3),
                                       1.3, 1.3,
                                       facecolor=tintes[qi * 2 + qj],
                                       alpha=0.45,
                                       edgecolor=tintes[qi * 2 + qj],
                                       linewidth=2.2))
                ax.text(6.65 + qj * 1.3, 1.7 + (1 - qi) * 1.3,
                        f"{valor:.2f}".rstrip("0").rstrip("."),
                        ha="center", va="center", fontsize=12,
                        fontweight="bold")
        ax.text(7.3, 4.25, "salida 2×2", ha="center", fontsize=9.5,
                color=GRIS)
        ax.set_title(nombre, fontsize=11.5)

    fig.suptitle("Pooling: resumir y encoger — cada ventana se reduce a UN "
                 "número (sin pesos que aprender)", fontweight="bold", y=1.04)
    fig.savefig(DESTINO / "pooling.png")
    plt.close(fig)


def fig_layernorm_batchnorm() -> None:
    """El único punto donde difieren BatchNorm y LayerNorm: el EJE sobre
    el que calculan media y desviación. Misma matriz de activaciones
    (B muestras × d features): BatchNorm resalta una columna, LayerNorm
    una fila."""
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(5)
    B, D = 5, 7
    vals = rng.normal(0, 1, (B, D))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    escenarios = [
        ("BatchNorm — típico en CNN", "columna", VERDE,
         "una μ, σ POR FEATURE,\ncalculadas a través del BATCH"),
        ("LayerNorm — estándar en Transformers", "fila", NARANJA,
         "una μ, σ POR MUESTRA,\ncalculadas a través de sus FEATURES"),
    ]
    for ax, (titulo, modo, color, nota) in zip(axes, escenarios):
        ax.set_xlim(-1.8, D + 3.6)
        ax.set_ylim(-1.9, B + 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.grid(False)

        for i in range(B):
            for j in range(D):
                resaltada = (j == 2) if modo == "columna" else (i == 1)
                ax.add_patch(Rectangle((j, B - 1 - i), 1, 1,
                                       facecolor=color if resaltada else "#EDEFF3",
                                       alpha=0.55 if resaltada else 1.0,
                                       edgecolor="#999", linewidth=0.7))
                ax.text(j + 0.5, B - 1 - i + 0.5, f"{vals[i, j]:+.1f}",
                        ha="center", va="center", fontsize=8,
                        color="#333")
        # marco grueso del grupo que comparte μ, σ
        if modo == "columna":
            ax.add_patch(Rectangle((2, 0), 1, B, facecolor="none",
                                   edgecolor=color, linewidth=3))
            ax.text(2.5, B + 0.55, "↓ este grupo comparte μ, σ", ha="center",
                    fontsize=9, color=color, fontweight="bold")
        else:
            ax.add_patch(Rectangle((0, B - 2), D, 1, facecolor="none",
                                   edgecolor=color, linewidth=3))
            ax.text(D + 0.25, B - 1.5, "← este grupo\ncomparte μ, σ",
                    ha="left", va="center", fontsize=9, color=color,
                    fontweight="bold")

        ax.text(-0.45, B / 2, "muestras del batch (B)", rotation=90,
                ha="center", va="center", fontsize=9, color=GRIS)
        ax.text(D / 2, -0.55, "features / canales (d)", ha="center",
                va="top", fontsize=9, color=GRIS)
        ax.text(D / 2, -1.45, nota, ha="center", va="top", fontsize=9.5,
                color="#1a2332", style="italic")
        ax.set_title(titulo, fontsize=11.5)

    fig.suptitle("BatchNorm y LayerNorm: la misma fórmula, distinto eje — "
                 "por eso una depende del batch y la otra no",
                 fontweight="bold", y=1.05)
    fig.savefig(DESTINO / "layernorm_batchnorm.png")
    plt.close(fig)


def fig_lr_schedules() -> None:
    """El learning rate como función del tiempo: constante, step decay,
    cosine y warmup+cosine, sobre el mismo presupuesto de 60 epochs."""
    T = 60
    t = np.arange(T + 1)
    base, eta_min = 1e-3, 1e-5

    constante = np.full_like(t, base, dtype=float)
    step = base * 0.1 ** np.minimum(t // 20, 2)
    cosine = eta_min + 0.5 * (base - eta_min) * (1 + np.cos(np.pi * t / T))
    calent = 5
    warmup = np.where(
        t <= calent,
        base * t / calent,
        eta_min + 0.5 * (base - eta_min) * (1 + np.cos(np.pi * (t - calent) / (T - calent))),
    )
    warmup[0] = base / 50   # evitar 0 en escala log

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(t, constante, color=GRIS, linewidth=2, linestyle=":",
            label="constante (punto de partida)")
    ax.plot(t, step, color=MORADO, linewidth=2.2,
            label="step decay (recortes ×0.1)")
    ax.plot(t, cosine, color=AZUL, linewidth=2.4,
            label="cosine (el del Lab 2)")
    ax.plot(t, warmup, color=NARANJA, linewidth=2.4,
            label="warmup + cosine (Transformers, Sesión 4)")
    ax.axvspan(0, calent, color=NARANJA, alpha=0.10)
    ax.annotate("warmup: crecer desde ~0\ncon la red recién inicializada",
                xy=(2.5, 4e-4), xytext=(12, 1.1e-4), fontsize=8.5,
                color="#8a5a00",
                arrowprops=dict(arrowstyle="->", color="#8a5a00"))
    ax.set(title="El learning rate como función del tiempo (schedules)",
           xlabel="epoch", ylabel="learning rate η (escala log)",
           yscale="log")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.savefig(DESTINO / "lr_schedules.png")
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
                 y=1.06,
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


def fig_transformer_anatomia() -> None:
    """La anatomía de un Transformer tipo GPT en una sola imagen:
    texto → tokens/IDs → embeddings + posición → N bloques (MHA + FFN)
    → logits → softmax → probabilidad del siguiente token. Es el mapa
    que abre la Sesión 3, gemelo de cnn_anatomia para la Sesión 2."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(14.5, 6.0))
    ax.set_xlim(0, 30)
    ax.set_ylim(1.4, 12)
    ax.axis("off")
    ax.grid(False)

    y_mid = 6.8
    Y_TITULO, Y_SUB = 9.7, 4.0

    def titulo_etapa(xc, texto):
        ax.text(xc, Y_TITULO, texto, ha="center", fontsize=10.5,
                fontweight="bold", color="#1a2332")

    def sub_etapa(xc, texto):
        ax.text(xc, Y_SUB, texto, ha="center", va="top", fontsize=8.5,
                color=GRIS, style="italic")

    def flecha(x0, x1, texto=None, y_texto=None):
        ax.add_patch(FancyArrowPatch((x0, y_mid), (x1, y_mid),
                                     arrowstyle="->", mutation_scale=14,
                                     color=NARANJA, linewidth=2.0, zorder=2))
        if texto:
            ax.text((x0 + x1) / 2, y_texto or y_mid + 0.55, texto,
                    ha="center", fontsize=8, color=NARANJA,
                    fontweight="bold")

    def caja(x, y, w, h, color, texto="", fs=9, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor=AZUL,
                                    linewidth=1.2, **kw))
        if texto:
            ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
                    fontsize=fs)

    # ── 1 · Texto crudo ──
    caja(0.5, y_mid - 0.9, 2.7, 1.8, "#F3F3F3", '"El gato\ncome…"', fs=10)
    titulo_etapa(1.85, "Texto")
    sub_etapa(1.85, "la entrada\ncruda")

    # ── 2 · Tokens → IDs ──
    flecha(3.4, 4.3, "tokenizer", y_texto=8.55)
    tokens = [("El", "12"), ("gato", "843"), ("come", "7")]
    for k, (tok, idd) in enumerate(tokens):
        y = y_mid + 0.95 - k * 0.95
        ax.add_patch(Rectangle((4.5, y - 0.36), 1.35, 0.72,
                               facecolor="#F3F3F3", edgecolor=AZUL,
                               linewidth=1.0))
        ax.text(5.17, y, tok, ha="center", va="center", fontsize=9)
        ax.add_patch(Rectangle((5.85, y - 0.36), 0.95, 0.72,
                               facecolor="#DCE9F6", edgecolor=AZUL,
                               linewidth=1.0))
        ax.text(6.32, y, idd, ha="center", va="center", fontsize=9,
                family="monospace")
    titulo_etapa(5.65, "Tokens → IDs")
    sub_etapa(5.65, "el vocabulario asigna\nun entero a cada token")

    # ── 3 · Embeddings + posición ──
    flecha(7.05, 7.95, "lookup en E", y_texto=8.55)
    rng = np.random.default_rng(7)
    for k in range(3):
        y = y_mid + 0.95 - k * 0.95
        for j in range(6):
            ax.add_patch(Rectangle((8.15 + j * 0.4, y - 0.3), 0.4, 0.6,
                                   facecolor=AZUL,
                                   alpha=0.15 + 0.75 * rng.random(),
                                   edgecolor="white", linewidth=0.5))
    # la señal de posición que se SUMA
    xs = np.linspace(8.15, 10.55, 80)
    ax.plot(xs, y_mid - 2.1 + 0.28 * np.sin(2.2 * np.pi *
            (xs - 8.15) / 2.4), color=MORADO, linewidth=1.8)
    ax.text(9.35, y_mid - 2.75, "⊕ posición (PE)", ha="center", fontsize=8,
            color=MORADO, fontweight="bold")
    titulo_etapa(9.35, "Embeddings")
    sub_etapa(9.35, "cada ID → su vector\n+ la señal de posición")

    # ── 4 · N bloques Transformer ──
    flecha(10.85, 11.75)
    for k in range(2, -1, -1):     # pila con offset
        caja(11.95 + k * 0.3, y_mid - 1.7 + k * 0.25, 4.6, 3.4,
             "#EAF2FA" if k else "#DCE9F6", zorder=3 + (2 - k))
    caja(12.35, y_mid + 0.15, 3.8, 1.05, "#BBDEF5",
         "Multi-Head Attention\nmezcla ENTRE tokens", fs=8, zorder=7)
    caja(12.35, y_mid - 1.35, 3.8, 1.05, "#A7D9C9",
         "FFN\ntransforma cada token", fs=8, zorder=7)
    ax.text(17.35, 8.75, "×N", fontsize=12, fontweight="bold",
            color=AZUL)
    titulo_etapa(14.15, "Bloques Transformer")
    sub_etapa(14.15, "N copias apiladas\n(GPT-2 small: N = 12)")

    # ── 5 · Logits del último token ──
    flecha(17.1, 18.0)
    alturas = [2.6, 1.6, 1.0, 0.6, 0.35]
    for k, hbar in enumerate(alturas):
        ax.add_patch(Rectangle((18.25 + k * 0.5, y_mid - 1.4), 0.36, hbar,
                               facecolor=AZUL if k == 0 else CELESTE,
                               edgecolor="none"))
    ax.plot([18.1, 20.8], [y_mid - 1.4, y_mid - 1.4], color=GRIS,
            linewidth=0.8)
    titulo_etapa(19.45, "Logits")
    sub_etapa(19.45, "un puntaje por token\ndel vocabulario")

    # ── 6 · softmax → siguiente token ──
    flecha(21.1, 22.0, "softmax")
    clases = [("pescado", 0.62, NARANJA), ("croquetas", 0.21, CELESTE),
              ("siesta", 0.07, CELESTE)]
    for fila, (nombre, p, color) in enumerate(clases):
        y = y_mid + 1.0 - fila * 1.0
        ax.add_patch(Rectangle((23.7, y - 0.28), 3.4 * p, 0.56,
                               facecolor=color, edgecolor="none"))
        ax.text(23.58, y, nombre, ha="right", va="center", fontsize=9)
        ax.text(23.82 + 3.4 * p, y, f"{p:.2f}", ha="left", va="center",
                fontsize=9, fontweight="bold")
    titulo_etapa(25.3, "Siguiente token")
    sub_etapa(25.3, "generar texto = elegir\nuno y volver a empezar")

    # ── Las tres zonas ──
    def llave(x0, x1, texto):
        ax.plot([x0, x0, x1, x1], [2.95, 2.7, 2.7, 2.95],
                color="#1a2332", linewidth=1.4)
        ax.text((x0 + x1) / 2, 2.25, texto, ha="center", va="top",
                fontsize=10, fontweight="bold", color="#1a2332")

    llave(0.5, 10.6, "DE TEXTO A VECTORES (§2)")
    llave(11.6, 16.9, "EL TRANSFORMER (§4–§6)")
    llave(17.9, 27.6, "PREDICCIÓN (Sesión 1: logits → softmax)")

    ax.text(15, 11.35,
            "La anatomía de un Transformer (tipo GPT): del texto a la "
            "probabilidad del siguiente token",
            ha="center", fontsize=13.5, fontweight="bold")
    fig.savefig(DESTINO / "transformer_anatomia.png")
    plt.close(fig)


def fig_lstm_compuertas() -> None:
    """La celda LSTM como cinta transportadora de memoria: la compuerta
    f borra (⊗), la i escribe (⊕) y la o decide qué se lee hacia h_t.
    La cinta cruza el tiempo con SUMAS — la autopista del gradiente."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 11.5)
    ax.axis("off")
    ax.grid(False)

    def caja(x, y, w, h, color, texto, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="#666",
                                    linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
                fontsize=fs)

    def valvula(x, simbolo, color):
        ax.add_patch(plt.Circle((x, 8.4), 0.45, facecolor="white",
                                edgecolor=color, linewidth=2.6, zorder=5))
        ax.text(x, 8.4, simbolo, ha="center", va="center", fontsize=15,
                color=color, fontweight="bold", zorder=6)

    def flecha(p0, p1, color="#666", ls="-"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->",
                                     mutation_scale=13, color=color,
                                     linewidth=1.6, linestyle=ls, zorder=4))

    # ── La cinta de memoria ──
    ax.add_patch(FancyArrowPatch((1.2, 8.4), (22.8, 8.4), arrowstyle="->",
                                 mutation_scale=22, color=AZUL,
                                 linewidth=5, zorder=3))
    ax.text(1.0, 9.15, "cₜ₋₁", fontsize=13, fontweight="bold", color=AZUL)
    ax.text(22.0, 9.15, "cₜ", fontsize=13, fontweight="bold", color=AZUL)
    ax.text(11.5, 10.0, "la cinta de memoria atraviesa el tiempo casi sin tocarse",
            ha="center", fontsize=9.5, color=AZUL, style="italic")

    # ── Compuerta 1: olvidar ──
    valvula(6, "⊗", ROJO)
    caja(4.4, 5.4, 3.2, 1.15, "#F6D9CE", "fₜ = σ( W·[xₜ, hₜ₋₁] )")
    flecha((6, 6.55), (6, 7.9), ROJO)
    ax.text(6, 4.85, "OLVIDAR: ¿qué borro?", ha="center",
            fontsize=9, color=ROJO, fontweight="bold")

    # ── Compuerta 2: escribir ──
    valvula(11, "⊕", VERDE)
    caja(9.3, 5.4, 3.4, 1.15, "#CBE8DC", "iₜ ⊙ c̃ₜ  (cuánto × qué)")
    flecha((11, 6.55), (11, 7.9), VERDE)
    ax.text(11, 4.85, "ESCRIBIR: ¿qué anoto?", ha="center",
            fontsize=9, color=VERDE, fontweight="bold")

    # ── Compuerta 3: leer ──
    ax.plot([16, 16], [8.4, 6.75], color="#666", linewidth=1.6, zorder=2)
    caja(15.1, 5.9, 1.8, 0.85, "#EFEFEF", "tanh")
    ax.plot([16, 16], [5.9, 5.05], color="#666", linewidth=1.6)
    ax.add_patch(plt.Circle((16, 4.6), 0.45, facecolor="white",
                            edgecolor=NARANJA, linewidth=2.6, zorder=5))
    ax.text(16, 4.6, "⊗", ha="center", va="center", fontsize=15,
            color=NARANJA, fontweight="bold", zorder=6)
    caja(17.6, 4.15, 2.4, 0.9, "#F9E3C8", "oₜ = σ(...)")
    flecha((17.55, 4.6), (16.5, 4.6), NARANJA)
    flecha((16, 4.1), (16, 2.9), NARANJA)
    ax.text(16, 2.35, "hₜ — lo que ve el resto de la red",
            ha="center", fontsize=9.5, fontweight="bold", color=NARANJA)
    ax.text(18.8, 5.45, "LEER — ¿qué comparto?", ha="center", fontsize=9,
            color=NARANJA, fontweight="bold")

    # ── La entrada alimenta a las tres compuertas ──
    caja(1.4, 0.7, 7.6, 1.0, "#F3F3F3",
         "[xₜ, hₜ₋₁] = el token actual + el estado anterior", fs=9)
    for origen, destino in [((3.0, 1.75), (6, 5.35)),
                            ((5.2, 1.75), (11, 5.35)),
                            ((8.6, 1.75), (18.8, 4.1))]:
        flecha(origen, destino, "#999", ls="--")

    ax.text(12, 11.15,
            "La celda LSTM: una cinta transportadora de memoria con tres compuertas",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(20.9, 1.0,
            "La cinta avanza con SUMAS,\nno multiplicaciones repetidas:\n"
            "la autopista del gradiente\n(la misma idea que ResNet)",
            ha="center", fontsize=8.5, color=GRIS, style="italic")
    fig.savefig(DESTINO / "lstm_compuertas.png")
    plt.close(fig)


def fig_multihead() -> None:
    """Multi-head attention: la entrada se proyecta a h heads paralelas
    — cada una con su patrón de atención DISTINTO —, se concatenan y
    W^O las mezcla. Los mini-heatmaps muestran la especialización."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.grid(False)

    colores_head = [CELESTE, VERDE, NARANJA, MORADO]
    # Cuatro patrones de atención distintos (causales), 4 tokens
    diag = np.eye(4)
    prev = np.zeros((4, 4))
    prev[0, 0] = 1
    for i in range(1, 4):
        prev[i, i - 1] = 1
    primero = np.zeros((4, 4))
    primero[:, 0] = 1
    amplio = np.tril(np.ones((4, 4)))
    amplio = amplio / amplio.sum(axis=1, keepdims=True)
    patrones = [
        (diag, "se mira\na sí mismo"),
        (prev, "mira el token\nanterior"),
        (primero, "mira el\nprimer token"),
        (amplio, "reparte en todo\nel pasado"),
    ]

    def barra(x, color, texto):
        ax.add_patch(FancyBboxPatch((x, 2.4), 1.5, 6.2,
                                    boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor=AZUL,
                                    linewidth=1.3))
        ax.text(x + 0.75, 5.5, texto, ha="center", va="center", fontsize=9,
                rotation=90)

    barra(0.8, "#DCE9F6", "x — (B, T, d_model)")

    # ── Las h heads en paralelo ──
    ys = [8.9, 6.65, 4.4, 2.15]           # centro vertical de cada head
    for k, (yc, color, (patron, nota)) in enumerate(
            zip(ys, colores_head, patrones)):
        ax.add_patch(FancyBboxPatch((5.6, yc - 1.0), 5.8, 2.0,
                                    boxstyle="round,pad=0.06",
                                    facecolor="#FAFAFA", edgecolor=color,
                                    linewidth=1.8))
        ax.text(6.9, yc + 0.35, f"head {k + 1}", fontsize=9.5,
                fontweight="bold", color=color, ha="center")
        ax.text(6.9, yc - 0.45, nota, fontsize=7.5, color=GRIS,
                ha="center", style="italic")
        # mini-heatmap 4×4 del patrón de atención
        lado = 0.42
        x0, y0 = 9.0, yc - 0.88
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((x0 + j * lado, y0 + (3 - i) * lado),
                                       lado, lado, facecolor=color,
                                       alpha=0.08 + 0.8 * patron[i, j],
                                       edgecolor="white", linewidth=0.5))
        # entrada → head y head → concat
        ax.add_patch(FancyArrowPatch((2.4, 5.5 + (yc - 5.5) * 0.35),
                                     (5.5, yc), arrowstyle="->",
                                     mutation_scale=11, color="#999",
                                     linewidth=1.2))
        ax.add_patch(FancyArrowPatch((11.5, yc),
                                     (13.8, 8.15 - k * 1.55 - 0.75),
                                     arrowstyle="->", mutation_scale=11,
                                     color=color, linewidth=1.4))

    ax.text(3.9, 0.9, "cada head con sus PROPIAS proyecciones\n"
            "Wᵢᵠ, Wᵢᴷ, Wᵢᵛ (aprendidas)", ha="center", fontsize=8,
            color=GRIS, style="italic")

    # ── Concat: las salidas se apilan ──
    for k, color in enumerate(colores_head):
        ax.add_patch(Rectangle((13.9, 8.15 - k * 1.55 - 1.5), 1.3, 1.5,
                               facecolor=color, alpha=0.55,
                               edgecolor="white", linewidth=1.2))
    ax.text(14.55, 8.7, "Concat", ha="center", fontsize=9.5,
            fontweight="bold")

    # ── W^O mezcla ──
    ax.add_patch(FancyArrowPatch((15.4, 5.25), (17.1, 5.25),
                                 arrowstyle="->", mutation_scale=13,
                                 color=NARANJA, linewidth=2))
    ax.add_patch(FancyBboxPatch((17.3, 4.25), 2.4, 2.0,
                                boxstyle="round,pad=0.06",
                                facecolor="#F9E3C8", edgecolor=NARANJA,
                                linewidth=1.5))
    ax.text(18.5, 5.25, "W^O\nmezcla", ha="center", va="center",
            fontsize=9.5)
    ax.add_patch(FancyArrowPatch((19.9, 5.25), (21.6, 5.25),
                                 arrowstyle="->", mutation_scale=13,
                                 color=NARANJA, linewidth=2))
    barra(21.8, "#BBDEF5", "salida — (B, T, d_model)")

    ax.text(13, 10.55,
            "Multi-head attention: h miradas en paralelo, concatenadas y "
            "mezcladas por W^O", ha="center", fontsize=13,
            fontweight="bold")
    ax.text(16.0, 0.30, "cada head aprende una relación DISTINTA — sus "
            "salidas juntas describen al token desde varios ángulos",
            ha="center", fontsize=8.5, color=GRIS, style="italic")
    fig.savefig(DESTINO / "multihead.png")
    plt.close(fig)


def fig_tokenizacion_hf() -> None:
    """El contrato de entrada de un modelo HF, en una imagen: la frase
    tokenizada en subwords, los tokens especiales [CLS]/[SEP], el
    padding y las dos filas que ve el modelo: input_ids y attention_mask."""
    from matplotlib.patches import Rectangle

    tokens = ["[CLS]", "deep", "learning", "is", "fascin", "##ating",
              "!", "[SEP]", "[PAD]", "[PAD]"]
    ids = ["101", "2784", "4083", "2003", "27596", "5844", "999", "102",
           "0", "0"]
    mask = ["1", "1", "1", "1", "1", "1", "1", "1", "0", "0"]
    anchos = [1.6, 1.4, 2.1, 0.9, 1.6, 1.9, 0.7, 1.5, 1.5, 1.5]

    def color_token(t):
        if t in ("[CLS]", "[SEP]"):
            return "#BBDEF5"
        if t in ("fascin", "##ating"):
            return "#F9E3C8"
        if t == "[PAD]":
            return "#E4E4E4"
        return "#F5F5F5"

    fig, ax = plt.subplots(figsize=(13, 4.9))
    ax.set_xlim(0, 21.5)
    ax.set_ylim(1.7, 11)
    ax.axis("off")
    ax.grid(False)

    xs = [3.4]
    for w in anchos[:-1]:
        xs.append(xs[-1] + w + 0.14)

    for fila, (y, h, valores, fs) in enumerate([
        (6.3, 1.0, tokens, 9.5),
        (4.9, 0.8, ids, 8.5),
        (3.6, 0.8, mask, 9),
    ]):
        for x, w, v, tok in zip(xs, anchos, valores, tokens):
            if fila == 0:
                color = color_token(tok)
            elif fila == 2:
                color = "#CBE8DC" if v == "1" else "#E4E4E4"
            else:
                color = "#FFFFFF"
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="#999", linewidth=0.8))
            ax.text(x + w / 2, y + h / 2, v, ha="center", va="center",
                    fontsize=fs,
                    family="monospace" if fila else "sans-serif",
                    color="#888" if tok == "[PAD]" and fila == 0 else "#222")
    for y, nombre in [(6.8, "tokens"), (5.3, "input_ids"),
                      (4.0, "attention_mask")]:
        ax.text(3.15, y, nombre, ha="right", va="center", fontsize=9.5,
                fontweight="bold")

    # Anotaciones: qué es cada cosa
    def nota(x0, x1, y_llave, y_texto, texto, color):
        ax.plot([x0, x0, x1, x1], [y_llave - 0.15, y_llave, y_llave,
                                   y_llave - 0.15], color=color,
                linewidth=1.3)
        ax.text((x0 + x1) / 2, y_texto, texto, ha="center", va="bottom",
                fontsize=8.3, color=color, fontweight="bold")

    nota(xs[0], xs[0] + anchos[0], 7.7, 7.85,
         "resumen de\nla frase", AZUL)
    nota(xs[4], xs[5] + anchos[5], 7.7, 7.85,
         "una palabra rara se parte en subwords\n('##' = continúa la palabra)",
         "#B96D00")
    nota(xs[7], xs[7] + anchos[7], 7.7, 7.85, "separador", AZUL)
    nota(xs[8], xs[9] + anchos[9], 7.7, 7.85,
         "relleno hasta el máximo\ndel batch", GRIS)

    ax.text(12.1, 2.25,
            "attention_mask: 1 = token real, 0 = padding (≠ la máscara causal "
            "de la Sesión 3) · IDs ilustrativos: dependen del vocabulario del "
            "checkpoint", ha="center", fontsize=8.5, color=GRIS,
            style="italic")
    ax.text(12.1, 10.6,
            'Lo que realmente entra al modelo: "Deep learning is fascinating!" '
            "tokenizada", ha="center", fontsize=12.5, fontweight="bold")
    fig.savefig(DESTINO / "tokenizacion_hf.png")
    plt.close(fig)


def fig_dynamic_padding() -> None:
    """Padding global vs dynamic padding: el mismo batch de 6 frases
    rellenado al máximo del dataset (16) o al máximo del batch (9).
    El gris es cómputo desperdiciado."""
    from matplotlib.patches import Rectangle

    lens = [4, 7, 5, 9, 3, 6]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    escenarios = [
        ("Padding global — al máximo del DATASET", 16),
        ("Dynamic padding — al máximo del BATCH", 9),
    ]
    for ax, (titulo, ancho) in zip(axes, escenarios):
        ax.set_xlim(-0.4, 16.4)
        ax.set_ylim(-1.9, len(lens) + 0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.grid(False)
        for i, ln in enumerate(lens):
            y = len(lens) - 1 - i
            for j in range(ancho):
                real = j < ln
                ax.add_patch(Rectangle((j, y), 0.92, 0.85,
                                       facecolor=AZUL if real else "#E0E0E0",
                                       alpha=0.6 if real else 1.0,
                                       edgecolor="white", linewidth=0.6))
        total = len(lens) * ancho
        utiles = sum(lens)
        ax.set_title(f"{titulo} ({ancho})", fontsize=11)
        ax.text(ancho / 2 - 0.5, -1.0,
                f"celdas procesadas: {len(lens)}×{ancho} = {total}  "
                f"(útiles: {utiles})",
                ha="center", fontsize=9.5, fontweight="bold")

    fig.suptitle("El mismo batch de 6 frases — el gris es puro relleno "
                 "(cómputo desperdiciado). DataCollatorWithPadding aplica "
                 "el dinámico por ti.", fontweight="bold", y=1.02)
    fig.savefig(DESTINO / "dynamic_padding.png")
    plt.close(fig)


def fig_finetuning_anatomia() -> None:
    """Qué construye AutoModelForSequenceClassification: el encoder
    Transformer PREENTRENADO (se ajusta apenas, LR pequeño) + una cabeza
    de clasificación NUEVA sobre el token [CLS] (nace aleatoria y
    aprende la tarea)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.grid(False)

    # ── Tokens de entrada ──
    tokens = ["[CLS]", "this", "movie", "is", "great", "[SEP]"]
    anchos = [1.7, 1.2, 1.7, 0.9, 1.6, 1.6]
    xs = [4.0]
    for w in anchos[:-1]:
        xs.append(xs[-1] + w + 0.15)
    for x, w, t in zip(xs, anchos, tokens):
        especial = t.startswith("[")
        ax.add_patch(Rectangle((x, 1.1), w, 0.95,
                               facecolor="#BBDEF5" if t == "[CLS]"
                               else ("#DCE9F6" if especial else "#F5F5F5"),
                               edgecolor="#999", linewidth=0.9))
        ax.text(x + w / 2, 1.575, t, ha="center", va="center", fontsize=9)
    x_cls = xs[0] + anchos[0] / 2          # la columna del [CLS]

    ax.add_patch(FancyArrowPatch((9.0, 2.25), (9.0, 3.1), arrowstyle="->",
                                 mutation_scale=13, color=NARANJA,
                                 linewidth=1.8))
    ax.text(9.6, 2.6, "tokenizer + embeddings", fontsize=8.5,
            color=NARANJA, fontweight="bold", ha="left")

    # ── El encoder preentrenado (pila de bloques) ──
    for k in range(2, -1, -1):
        ax.add_patch(FancyBboxPatch((3.7 + k * 0.28, 3.3 + k * 0.24),
                                    10.6, 4.3, boxstyle="round,pad=0.08",
                                    facecolor="#DCE9F6" if k == 0
                                    else "#EAF2FA",
                                    edgecolor=AZUL, linewidth=1.3,
                                    zorder=3 + (2 - k)))
    ax.text(9.0, 5.85, "bloques Transformer ×6", ha="center", fontsize=11,
            fontweight="bold", zorder=7)
    ax.text(9.0, 5.0, "Multi-Head Attention + FFN + residuals + LayerNorm\n"
            "(exactamente los de la Sesión 3)", ha="center", fontsize=8.5,
            color=GRIS, zorder=7)
    ax.text(15.6, 5.5,
            "PREENTRENADO\nya \"sabe inglés\".\nEn fine-tuning se ajusta\n"
            "apenas: LR pequeño (2e-5)", fontsize=9, color=AZUL,
            fontweight="bold", va="center")
    ax.plot([15.3, 15.3], [3.6, 7.4], color=AZUL, linewidth=1.6)

    # ── Del [CLS] a la cabeza nueva ──
    ax.add_patch(FancyArrowPatch((x_cls + 0.6, 7.9), (x_cls + 0.6, 8.9),
                                 arrowstyle="->", mutation_scale=13,
                                 color=NARANJA, linewidth=1.8))
    ax.text(x_cls + 1.2, 8.35, "el vector del [CLS] — 768 números:\n"
            "el resumen de toda la frase", fontsize=8.5, color=GRIS,
            ha="left", style="italic")
    ax.add_patch(FancyBboxPatch((2.0, 9.0), 6.8, 1.15,
                                boxstyle="round,pad=0.08",
                                facecolor="#F9E3C8", edgecolor=NARANJA,
                                linewidth=1.6, zorder=4))
    ax.text(5.4, 9.575, "cabeza nueva: Linear(768 → 2)", ha="center",
            va="center", fontsize=9.5, fontweight="bold", zorder=5)
    ax.text(10.4, 9.55, "NUEVA — nace aleatoria;\naprende TU tarea",
            fontsize=9, color="#B96D00", fontweight="bold", va="center")

    # ── Logits → softmax → probabilidades ──
    ax.add_patch(FancyArrowPatch((5.4, 10.25), (5.4, 11.1), arrowstyle="->",
                                 mutation_scale=13, color=NARANJA,
                                 linewidth=1.8))
    ax.text(6.0, 10.6, "softmax", fontsize=8.5, color=NARANJA,
            fontweight="bold", ha="left")
    for fila, (nombre, p, color) in enumerate([
            ("POSITIVE", 0.92, VERDE), ("NEGATIVE", 0.08, "#E0E0E0")]):
        y = 12.3 - fila * 0.95
        ax.add_patch(Rectangle((5.0, y - 0.3), 4.2 * p, 0.6,
                               facecolor=color, edgecolor="none"))
        ax.text(4.86, y, nombre, ha="right", va="center", fontsize=9)
        ax.text(5.12 + 4.2 * p, y, f"{p:.2f}", ha="left", va="center",
                fontsize=9, fontweight="bold")

    ax.text(11.5, 13.5,
            "La anatomía del fine-tuning: encoder preentrenado + cabeza nueva",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(11.5, 0.35,
            "Esto es exactamente lo que construye "
            "AutoModelForSequenceClassification(num_labels=2)",
            ha="center", fontsize=9, color=GRIS, style="italic")
    fig.savefig(DESTINO / "finetuning_anatomia.png")
    plt.close(fig)


def fig_neurona_vs_perceptron() -> None:
    """Anatomía comparada: neurona biológica (izquierda) vs perceptrón (derecha).

    Cada componente lleva un número ①-⑤ y un color propio que se repiten en
    ambos paneles y en el mapa inferior: así la correspondencia se lee mirando
    el dibujo, sin tener que reconstruirla desde el texto.

    Precisión importante: los pesos corresponden a las sinapsis de ENTRADA
    (los contactos sobre las dendritas), no a los terminales del axón, que son
    la salida hacia la siguiente neurona.
    """
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

    # Un color por componente, repetido en ambos paneles (paleta Okabe-Ito)
    C_ENTRADA, C_PESO, C_SUMA, C_ACT, C_SALIDA = AZUL, MORADO, VERDE, NARANJA, ROJO

    fig, (ax_bio, ax_art) = plt.subplots(1, 2, figsize=(15, 6.9))
    for ax in (ax_bio, ax_art):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.grid(False)

    def badge(ax, x: float, y: float, n: int, color: str) -> None:
        """Marca numerada que liga un componente con su par del otro panel."""
        ax.add_patch(Circle((x, y), 0.27, facecolor=color, edgecolor="white",
                            linewidth=1.6, zorder=7))
        ax.text(x, y, str(n), ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", zorder=8)

    def anotar(ax, anotaciones) -> None:
        for texto, xy_text, xy_punta, color in anotaciones:
            ax.annotate(texto, xy=xy_punta, xytext=xy_text, fontsize=9.5,
                        color="#222222", ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=color,
                                        linewidth=1.4, shrinkA=2, shrinkB=4))

    def nota(ax, texto: str) -> None:
        """Caja inferior; ambos paneles la comparten para quedar simétricos."""
        ax.add_patch(FancyBboxPatch((0.5, 0.2), 9.0, 1.5,
                                    boxstyle="round,pad=0.12,rounding_size=0.18",
                                    facecolor="#F7FAFD", edgecolor="#C9D8E4",
                                    linewidth=1.3, zorder=1))
        ax.text(5.0, 0.95, texto, ha="center", va="center", fontsize=9.5,
                style="italic", color="#444444", zorder=2)

    # ─────────────── Panel izquierdo: neurona biológica ───────────────
    ax_bio.set_title("Neurona biológica", fontsize=15, fontweight="bold", pad=14)

    x_soma, y_soma = 3.35, 5.9   # y_soma = y_centro del perceptrón: paneles alineados

    # Soma (cuerpo celular) y núcleo — integra: es el Σ biológico
    ax_bio.add_patch(Circle((x_soma, y_soma), 1.15, facecolor=C_SUMA, alpha=0.22,
                            edgecolor=C_SUMA, linewidth=2.2, zorder=3))
    ax_bio.add_patch(Circle((x_soma, y_soma), 0.40, facecolor=C_SUMA, alpha=0.75,
                            edgecolor="white", linewidth=1.2, zorder=4))

    # Dendritas, con su contacto sináptico de entrada en la punta
    ang_dendritas = (128, 150, 172, 195, 218)
    for ang in ang_dendritas:
        rad = np.deg2rad(ang)
        x0, y0 = x_soma + 2.55 * np.cos(rad), y_soma + 2.55 * np.sin(rad)
        x1, y1 = x_soma + 1.05 * np.cos(rad), y_soma + 1.05 * np.sin(rad)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2 + 0.16
        ax_bio.plot([x0, xm, x1], [y0, ym, y1], color=C_ENTRADA, linewidth=2.2,
                    solid_capstyle="round", zorder=2)
        # La punta es la sinapsis: su fuerza es lo que el perceptrón llama peso
        ax_bio.add_patch(Circle((x0, y0), 0.15, facecolor=C_PESO,
                                edgecolor="white", linewidth=1.0, zorder=5))

    # Cono axónico: donde se decide el disparo — el φ biológico
    ax_bio.add_patch(Polygon([[x_soma + 1.02, y_soma + 0.42],
                              [x_soma + 1.02, y_soma - 0.42],
                              [x_soma + 1.95, y_soma]],
                             closed=True, facecolor=C_ACT, alpha=0.75,
                             edgecolor=C_ACT, linewidth=1.6, zorder=4))

    # Axón, con sus vainas de mielina, y terminales hacia la siguiente neurona
    ax_bio.plot([x_soma + 1.9, 8.0], [y_soma, y_soma], color=C_SALIDA,
                linewidth=3.4, solid_capstyle="round", zorder=2)
    for xm in (5.9, 6.7, 7.5):
        ax_bio.add_patch(FancyBboxPatch((xm - 0.22, y_soma - 0.18), 0.44, 0.36,
                                        boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor=AMARILLO, edgecolor=C_SALIDA,
                                        linewidth=1.4, zorder=3))
    for dy in (0.62, 0.0, -0.62):
        ax_bio.plot([8.0, 8.7], [y_soma, y_soma + dy], color=C_SALIDA,
                    linewidth=2.4, solid_capstyle="round", zorder=2)
        ax_bio.add_patch(Circle((8.8, y_soma + dy), 0.15, facecolor=C_SALIDA,
                                zorder=3))

    # Puntos de referencia sobre dendritas concretas, para badges y flechas
    r150 = np.deg2rad(150)
    med_dendrita = (x_soma + 1.8 * np.cos(r150), y_soma + 1.8 * np.sin(r150) + 0.14)
    r218 = np.deg2rad(218)
    sinapsis_baja = (x_soma + 2.55 * np.cos(r218), y_soma + 2.55 * np.sin(r218))

    badge(ax_bio, med_dendrita[0] + 0.30, med_dendrita[1] + 0.40, 1, C_ENTRADA)
    badge(ax_bio, sinapsis_baja[0] - 0.44, sinapsis_baja[1] - 0.28, 2, C_PESO)
    badge(ax_bio, x_soma - 0.62, y_soma - 0.92, 3, C_SUMA)
    badge(ax_bio, x_soma + 1.38, y_soma + 0.72, 4, C_ACT)
    badge(ax_bio, 7.1, y_soma + 0.52, 5, C_SALIDA)

    anotar(ax_bio, [
        ("Dendritas\nreciben las señales de entrada",
         (0.3, 9.0), med_dendrita, C_ENTRADA),
        ("Sinapsis de entrada\nsu fuerza pondera cada señal\ny se aprende con la experiencia",
         (0.15, 2.75), sinapsis_baja, C_PESO),
        ("Soma (cuerpo celular)\nintegra todo lo que recibe",
         (4.35, 2.75), (x_soma + 0.35, y_soma - 1.15), C_SUMA),
        ("Cono axónico\ndispara si se supera\nel umbral",
         (4.15, 8.85), (x_soma + 1.5, y_soma + 0.3), C_ACT),
        ("Axón y terminales\nllevan el impulso a\nla siguiente neurona",
         (7.15, 8.85), (7.7, y_soma + 0.18), C_SALIDA),
    ])

    nota(ax_bio, "La neurona dispara todo o nada: transmite el impulso solo cuando\n"
                 "la suma de los estímulos recibidos supera un umbral.")

    # ─────────────── Panel derecho: perceptrón ───────────────
    ax_art.set_title("Perceptrón (neurona artificial)", fontsize=15,
                     fontweight="bold", pad=14)

    entradas = [("$x_1$", 7.75), ("$x_2$", 5.9), ("$x_3$", 4.05)]
    x_in, x_sum, x_act, x_out = 1.15, 4.6, 6.5, 8.55
    y_centro = 5.9

    # Nodos de entrada + aristas ponderadas
    for i, (etq, y) in enumerate(entradas, start=1):
        ax_art.add_patch(Circle((x_in, y), 0.42, facecolor="white",
                                edgecolor=C_ENTRADA, linewidth=2.0, zorder=3))
        ax_art.text(x_in, y, etq, ha="center", va="center", fontsize=12, zorder=4)
        ax_art.add_patch(FancyArrowPatch((x_in + 0.42, y), (x_sum - 0.62, y_centro),
                                         arrowstyle="->", mutation_scale=13,
                                         color=C_ENTRADA, linewidth=1.7,
                                         shrinkA=0, shrinkB=0, zorder=2))
        # Etiqueta del peso sobre la arista: el color la liga con la sinapsis
        xw, yw = x_in + 1.55, y + (y_centro - y) * 0.44 + 0.17
        ax_art.text(xw, yw, f"$w_{i}$", fontsize=12, color=C_PESO,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.92), zorder=4)

    # Bias: entrada constante = 1
    ax_art.add_patch(Circle((x_in, 3.05), 0.42, facecolor="#F4F4F4",
                            edgecolor=GRIS, linewidth=1.8, linestyle="--", zorder=3))
    ax_art.text(x_in, 3.05, "$1$", ha="center", va="center", fontsize=12, zorder=4)
    ax_art.add_patch(FancyArrowPatch((x_in + 0.42, 3.05), (x_sum - 0.5, y_centro - 0.42),
                                     arrowstyle="->", mutation_scale=13, color=GRIS,
                                     linewidth=1.6, linestyle="--",
                                     shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text(x_in + 1.62, 3.62, "$b$", fontsize=12, color=GRIS, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=4)

    # Nodo suma (Σ) — el soma artificial
    ax_art.add_patch(Circle((x_sum, y_centro), 0.62, facecolor=C_SUMA, alpha=0.22,
                            edgecolor=C_SUMA, linewidth=2.2, zorder=3))
    ax_art.text(x_sum, y_centro, r"$\Sigma$", ha="center", va="center",
                fontsize=17, zorder=4)

    # Nodo activación (φ) — el umbral artificial
    ax_art.add_patch(FancyArrowPatch((x_sum + 0.62, y_centro), (x_act - 0.62, y_centro),
                                     arrowstyle="->", mutation_scale=13, color=C_SUMA,
                                     linewidth=1.8, shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text((x_sum + x_act) / 2, y_centro + 0.36, "$z$", fontsize=12,
                color=C_SUMA, ha="center", va="center", zorder=4)
    ax_art.add_patch(Circle((x_act, y_centro), 0.62, facecolor=C_ACT, alpha=0.55,
                            edgecolor=C_ACT, linewidth=2.2, zorder=3))
    ax_art.text(x_act, y_centro, r"$\varphi$", ha="center", va="center",
                fontsize=16, zorder=4)

    # Salida
    ax_art.add_patch(FancyArrowPatch((x_act + 0.62, y_centro), (x_out - 0.1, y_centro),
                                     arrowstyle="->", mutation_scale=13, color=C_SALIDA,
                                     linewidth=2.2, shrinkA=0, shrinkB=0, zorder=2))
    ax_art.text(x_out + 0.32, y_centro, r"$\hat{y}$", fontsize=14, color=C_SALIDA,
                fontweight="bold", ha="center", va="center", zorder=4)

    badge(ax_art, x_in - 0.5, entradas[0][1] + 0.38, 1, C_ENTRADA)
    badge(ax_art, x_in + 1.62, entradas[0][1] - 0.02, 2, C_PESO)
    badge(ax_art, x_sum - 0.72, y_centro + 0.62, 3, C_SUMA)
    badge(ax_art, x_act - 0.72, y_centro + 0.62, 4, C_ACT)
    badge(ax_art, x_out + 0.32, y_centro + 0.72, 5, C_SALIDA)

    anotar(ax_art, [
        ("Entradas $x_j$\n(las features)",
         (0.15, 9.0), (x_in - 0.15, entradas[0][1] + 0.5), C_ENTRADA),
        ("Pesos $w_j$\nimportancia de cada\nentrada (se aprenden)",
         (2.3, 8.9), (2.35, 7.35), C_PESO),
        ("Bias $b$ — entrada constante\nque desplaza el umbral",
         (0.15, 2.08), (x_in - 0.28, 2.8), GRIS),
        ("Suma ponderada",
         (5.35, 3.6), (x_sum + 0.35, y_centro - 0.7), C_SUMA),
        ("Activación $\\varphi$\nintroduce la no linealidad",
         (5.9, 8.95), (x_act, y_centro + 0.68), C_ACT),
        ("Salida $\\hat{y}$\nla predicción",
         (8.35, 8.0), (x_out + 0.3, y_centro + 1.05), C_SALIDA),
    ])

    # Fórmulas: misma caja que la nota del panel bio, para que ambos cierren igual
    ax_art.add_patch(FancyBboxPatch((0.5, 0.2), 9.0, 1.5,
                                    boxstyle="round,pad=0.12,rounding_size=0.18",
                                    facecolor="#F7FAFD", edgecolor="#C9D8E4",
                                    linewidth=1.3, zorder=1))
    ax_art.text(5.0, 1.28,
                r"$z \;=\; \sum_{j=1}^{n} w_j x_j \;+\; b \;=\; "
                r"\mathbf{w}^\top \mathbf{x} + b$",
                ha="center", va="center", fontsize=14, zorder=2)
    ax_art.text(5.0, 0.54, r"$\hat{y} \;=\; \varphi(z)$",
                ha="center", va="center", fontsize=14, zorder=2)

    # ─────────────── Mapa de correspondencia ───────────────
    fig.subplots_adjust(bottom=0.155, top=0.90, wspace=0.05)
    pares = [
        ("①", "dendritas", r"entradas $x_j$", C_ENTRADA),
        ("②", "sinapsis", r"pesos $w_j$", C_PESO),
        ("③", "soma", r"suma $\Sigma$", C_SUMA),
        ("④", "cono axónico", r"activación $\varphi$", C_ACT),
        ("⑤", "axón", r"salida $\hat{y}$", C_SALIDA),
    ]
    x0, ancho = 0.055, 0.895 / len(pares)
    for i, (num, bio, art, color) in enumerate(pares):
        fig.text(x0 + ancho * (i + 0.5), 0.072, f"{num}  {bio}  →  {art}",
                 ha="center", va="center", fontsize=10.5, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                           edgecolor=color, linewidth=1.6))
    fig.text(0.5, 0.016,
             "La analogía es inspiración histórica, no equivalencia: una neurona real "
             "es mucho más compleja que esta suma ponderada.",
             ha="center", va="center", fontsize=9, style="italic", color=GRIS)
    fig.savefig(DESTINO / "neurona_vs_perceptron.png")
    plt.close(fig)


def main() -> None:
    """Genera todas las figuras estáticas."""
    aplicar_estilo()
    DESTINO.mkdir(parents=True, exist_ok=True)
    tareas = [
        fig_tensores, fig_neurona_frontera, fig_xor, fig_capa_densa,
        fig_softmax_pasos, fig_losses, fig_paisaje_perdida,
        fig_matriz_confusion, fig_lora,
        fig_activaciones, fig_softmax_temperatura, fig_curvas_aprendizaje,
        fig_learning_rate, fig_moons_frontera, fig_kernels, fig_padding_stride,
        fig_cnn_anatomia, fig_pooling, fig_layernorm_batchnorm,
        fig_lr_schedules,
        fig_receptive_field, fig_rnn_gradientes, fig_atencion,
        fig_positional_encoding, fig_embeddings_2d, fig_complejidad,
        fig_transformer_anatomia, fig_lstm_compuertas, fig_multihead,
        fig_tokenizacion_hf, fig_dynamic_padding, fig_finetuning_anatomia,
        fig_neurona_vs_perceptron,
    ]
    for tarea in tareas:
        tarea()
        print(f"✔ {tarea.__name__}")


if __name__ == "__main__":
    main()

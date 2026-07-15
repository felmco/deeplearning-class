"""Estilo visual compartido por todas las figuras del curso.

Qué es: un módulo de configuración de matplotlib.
Qué hace: define paleta de colores accesible (apta para daltonismo),
tipografía y parámetros comunes, para que TODAS las figuras del curso
tengan identidad visual consistente y profesional.
"""

import matplotlib as mpl

# Paleta accesible (basada en la paleta de Okabe-Ito, estándar en
# publicaciones científicas por ser distinguible con daltonismo)
AZUL = "#0072B2"
NARANJA = "#E69F00"
VERDE = "#009E73"
ROJO = "#D55E00"
MORADO = "#CC79A7"
CELESTE = "#56B4E9"
AMARILLO = "#F0E442"
GRIS = "#666666"

PALETA = [AZUL, NARANJA, VERDE, ROJO, MORADO, CELESTE]


def aplicar_estilo() -> None:
    """Aplica los parámetros globales de estilo a matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=PALETA),
    })

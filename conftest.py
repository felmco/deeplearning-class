"""Asegura que la raíz del repo esté en sys.path para que `import src` funcione en pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

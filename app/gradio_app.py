"""Demo web del clasificador de sentimiento del proyecto final.

Qué es: una interfaz Gradio mínima para probar el modelo fine-tuned.
Qué hace: carga el mejor checkpoint guardado por el Laboratorio 4
(notebooks/06_hf_finetuning.ipynb), tokeniza el texto del usuario y
muestra las probabilidades por clase.

Ejecutar:  python app/gradio_app.py
Requiere:  haber entrenado antes (el checkpoint vive en artifacts/,
           carpeta ignorada por git — cada equipo genera el suyo).
"""

from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ruta del checkpoint producido por el laboratorio de fine-tuning
MODEL_PATH = "artifacts/distilbert-rotten-tomatoes/best_model"

# ── Carga del modelo y tokenizer (deben venir del MISMO checkpoint) ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # modo evaluación: sin dropout

# Mejor dispositivo disponible (misma lógica que src/utils.py)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)


def predict(text: str) -> dict[str, float]:
    """Texto → distribución de probabilidad sobre las clases.

    Pasos: tokenizar (con truncation), mover al device, forward sin
    gradientes, softmax sobre los logits.
    """
    if not text or not text.strip():
        return {"NEGATIVE": 0.0, "POSITIVE": 0.0}

    encoded = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256)
    encoded = {clave: valor.to(device) for clave, valor in encoded.items()}

    with torch.inference_mode():
        probabilidades = model(**encoded).logits.softmax(dim=-1).squeeze(0).cpu()

    return {
        model.config.id2label[i]: float(p)
        for i, p in enumerate(probabilidades)
    }


# ── Interfaz ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Movie review (en inglés)"),
    outputs=gr.Label(num_top_classes=2, label="Predicción"),
    title="Clasificador de sentimiento — demo educativa",
    description=(
        "Prototipo del curso Deep Learning (Future Tales, LLC). "
        "Entrenado sobre cornell-movie-review-data/rotten_tomatoes. "
        "Validar antes de cualquier "
        "uso real: ver la model card del proyecto."
    ),
)

if __name__ == "__main__":
    demo.launch()

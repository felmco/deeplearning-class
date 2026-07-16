# 🌉 Puente PyTorch ↔ TensorFlow/Keras

Este curso usa **PyTorch** (por su grafo visible y su ecosistema de investigación —
ver la justificación en el [programa](../Programa_Completo_Deep_Learning_Transformers_32h.md)).
El otro gran framework de la industria es **TensorFlow** con su API de alto nivel
**Keras** — y buena parte de la bibliografía en español lo usa. La buena noticia:
**los conceptos son idénticos**; solo cambia la sintaxis. Quien domina uno lee el
otro en una tarde.

> Este documento es de **lectura**: TensorFlow no está en `requirements.txt` ni en la
> CI del curso. Si quieres ejecutar el ejemplo Keras, `pip install tensorflow` en un
> entorno aparte.

## Tabla de equivalencias

| Concepto (el que aprendiste) | PyTorch | TensorFlow / Keras |
|---|---|---|
| Tensor | `torch.Tensor` | `tf.Tensor` |
| Capa densa | `nn.Linear(d_in, d_out)` | `layers.Dense(d_out)` (infiere d_in) |
| Convolución | `nn.Conv2d(C_in, C_out, k)` | `layers.Conv2D(C_out, k)` |
| Activación | `nn.ReLU()` / `F.relu` | `layers.ReLU()` / `activation='relu'` |
| Modelo | subclase de `nn.Module` | `keras.Sequential` o subclase de `keras.Model` |
| Loss (binaria, con logits) | `nn.BCEWithLogitsLoss()` | `losses.BinaryCrossentropy(from_logits=True)` |
| Loss (multiclase, con logits) | `nn.CrossEntropyLoss()` | `losses.SparseCategoricalCrossentropy(from_logits=True)` |
| Optimizador | `torch.optim.AdamW(...)` | `optimizers.AdamW(...)` |
| Datos por batches | `Dataset` + `DataLoader` | `tf.data.Dataset.batch()` |
| Training loop | **lo escribes tú** (Sesión 1) | `model.compile()` + `model.fit()` |
| Modo train/eval | `model.train()` / `model.eval()` | `training=True/False` en la llamada |
| Sin gradientes | `torch.inference_mode()` | fuera de `tf.GradientTape` no hay grafo |
| Backprop manual | `loss.backward()` + `optimizer.step()` | `tape.gradient(...)` + `apply_gradients` |
| Convención de imágenes | **NCHW** `(B, C, H, W)` | **NHWC** `(B, H, W, C)` ← ¡ojo! |

Las dos diferencias de filosofía que explican todo lo demás:

1. **El loop:** PyTorch te hace escribirlo (por eso lo aprendiste primero — no hay
   caja negra); Keras lo empaqueta en `model.fit()`, igual que el `Trainer` de
   Hugging Face en la Sesión 4.
2. **El orden de los ejes de imagen:** PyTorch pone los canales antes (NCHW), Keras
   después (NHWC). Fuente clásica de bugs al portar código.

## El MoonMLP del Lab 1, lado a lado

**PyTorch (el del curso — [`src/models.py`](../src/models.py)):**

```python
import torch
from torch import nn

class MoonMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),          # 1 logit crudo
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = MoonMLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(100):                       # el loop es tuyo
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
```

**Keras (equivalente):**

```python
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(2,)),
    layers.Dropout(0.10),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.10),
    layers.Dense(1),                           # 1 logit crudo (sin sigmoid)
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(X_train, y_train, epochs=100, batch_size=64,   # el loop viene incluido
          validation_data=(X_val, y_val))
```

Nota cómo las decisiones importantes son las MISMAS: logits crudos (la sigmoid vive
dentro de la loss), AdamW con weight decay, dropout entre capas. El framework es el
envoltorio; los conceptos del curso son el contenido.

## ¿Y Hugging Face?

`transformers` (Sesión 4) soporta ambos backends; este curso usa el de PyTorch
(el default del ecosistema actual). Los checkpoints del Hub son los mismos.

---

| [🏠 Inicio](../README.md) | [Recursos](README.md) |
|---|---|

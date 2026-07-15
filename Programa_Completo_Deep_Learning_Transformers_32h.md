
# Programa completo de Deep Learning con capítulo de Transformers

**Duración:** 4 sesiones de 8 horas — 32 horas totales  
**Modalidad sugerida:** presencial o sincrónica, con exposición breve, demostración, práctica guiada y trabajo por equipos  
**Stack principal:** Python, PyTorch, Jupyter en VS Code, Git/GitHub y Hugging Face  
**Fecha de actualización de la guía:** 13 de julio de 2026  
**Idioma:** español, conservando términos técnicos estándar en inglés

> Esta guía sirve simultáneamente como programa académico, guion docente, especificación de slides, manual de laboratorios y brief del proyecto final. No pretende convertir cada bloque de código en una diapositiva: los slides deben presentar la intuición, la fórmula, la visualización y el contrato del ejercicio; el código completo debe vivir en notebooks o scripts del repositorio.

---

## 1. Propósito del curso

El curso introduce Deep Learning desde sus fundamentos matemáticos y computacionales hasta el uso práctico de arquitecturas modernas basadas en Transformers. La secuencia pedagógica evita tratar las librerías como una “caja negra”: primero se construyen tensores, grafos computacionales, funciones de pérdida, backpropagation y ciclos de entrenamiento; luego se estudian MLP, CNN, RNN/LSTM, atención y Transformers; finalmente se reutilizan modelos preentrenados con Hugging Face y se entrega un proyecto reproducible en GitHub.

El criterio rector es:

> **Cada concepto debe poder explicarse en cuatro niveles:** intuición, representación visual, formulación matemática e implementación verificable.

## 2. Audiencia y prerrequisitos

### Audiencia

- Estudiantes de ingeniería, ciencia de datos, informática o disciplinas afines.
- Profesionales con experiencia básica en Machine Learning que necesitan comprender Deep Learning y Transformers.
- Docentes o líderes técnicos que quieran desarrollar prototipos reproducibles, no solo ejecutar demos.

### Prerrequisitos mínimos

- Python: variables, funciones, clases básicas, listas/diccionarios y manejo de paquetes.
- NumPy y pandas a nivel introductorio.
- Álgebra lineal básica: vectores, matrices, producto punto y multiplicación matricial.
- Cálculo básico: derivada, derivada parcial y regla de la cadena.
- Fundamentos de aprendizaje supervisado: features, labels, train/test y métricas.
- Git básico es deseable, pero se incluye un flujo mínimo guiado.

### Diagnóstico previo sugerido

Aplicar un quiz de 15–20 minutos antes de la primera sesión:

1. Interpretar el shape `(32, 3, 224, 224)`.
2. Calcular un producto punto sencillo.
3. Explicar diferencia entre parámetro e hiperparámetro.
4. Identificar data leakage en un ejemplo.
5. Leer una función Python y predecir su salida.
6. Explicar qué hace `git commit` y qué hace `git push`.

## 3. Resultados de aprendizaje

Al terminar, el estudiante podrá:

1. Representar datos y operaciones de Deep Learning mediante tensores y shapes.
2. Explicar y programar forward propagation, funciones de pérdida, gradientes y backpropagation.
3. Implementar y entrenar una MLP y una CNN en PyTorch con un ciclo de entrenamiento explícito.
4. Diagnosticar underfitting, overfitting, inestabilidad y errores de datos o shapes.
5. Explicar la lógica de RNN, LSTM, atención, multi-head attention y bloques Transformer.
6. Implementar scaled dot-product attention y un bloque Transformer simplificado.
7. Usar tokenizers, datasets, modelos preentrenados y `Trainer` de Hugging Face.
8. Comparar un baseline con una red propia y un Transformer fine-tuned usando métricas adecuadas.
9. Documentar datos, experimentos, limitaciones y riesgos mediante README, configuración y model card.
10. Colaborar con VS Code y GitHub mediante branches, commits, issues y pull requests.

## 4. Decisiones curriculares

### Framework principal: PyTorch

Se usa PyTorch por su modelo imperativo, visibilidad del grafo computacional, ecosistema de investigación y compatibilidad directa con Hugging Face. El estudiante debe escribir al menos un training loop manual antes de usar abstracciones de alto nivel.

### Librería de Transformers: Hugging Face

Se utiliza Hugging Face para:

- cargar datasets y checkpoints;
- inspeccionar tokenizers y model cards;
- ejecutar inferencia con `pipeline`;
- realizar fine-tuning con `Trainer`;
- introducir PEFT/LoRA, cuantización y publicación responsable.

### Herramienta visual: Transformer Explainer

Se incorpora como laboratorio guiado, no como sustituto de la formulación. Permite recorrer embeddings, positional embeddings, masked multi-head self-attention, MLP, logits y estrategias de sampling usando GPT-2 small en el navegador.

### Arquitecturas cubiertas

- Perceptrón y MLP.
- CNN y conexiones residuales.
- RNN, LSTM y GRU.
- Scaled dot-product attention.
- Multi-head attention.
- Transformer encoder, decoder causal y encoder–decoder.
- BERT-like y GPT-like como familias conceptuales.
- Fine-tuning de un encoder Transformer para clasificación.

### Lo que no se intenta cubrir en profundidad

Por restricción de 32 horas, se presentan como extensiones:

- detección y segmentación avanzada;
- GAN, VAE y diffusion models;
- entrenamiento distribuido multi-GPU;
- pretraining de LLM desde cero;
- RLHF/DPO/GRPO;
- serving de alta escala y optimizaciones de kernels;
- interpretabilidad causal avanzada.

## 5. Metodología pedagógica

Cada bloque de 50–75 minutos sigue este patrón:

1. **Problema o intuición:** qué limitación motiva la técnica.
2. **Objeto matemático:** ecuación y shapes.
3. **Visualización:** gráfica, diagrama o animación.
4. **Código mínimo:** implementación corta y observable.
5. **Experimento:** variar una condición y formular una hipótesis.
6. **Evidencia:** curva, métrica, matriz o ejemplo de error.
7. **Reflexión:** cuándo usarlo, cuándo falla y qué riesgo introduce.

Distribución recomendada del tiempo total:

- 35% conceptos y fundamentos.
- 15% demostraciones del instructor.
- 35% laboratorios guiados o retos.
- 15% proyecto, revisión y comunicación.

## 6. Evaluación

| Componente | Peso | Evidencia |
|---|---:|---|
| Quizzes y exit tickets | 10% | Comprensión de shapes, fórmulas y decisiones |
| Laboratorio 1: MLP | 15% | Notebook, curvas, frontera y reflexión |
| Laboratorio 2: CNN | 15% | Experimentos comparados y análisis de errores |
| Laboratorio 3: atención/Transformer | 15% | Implementación, pruebas de shapes y heatmap |
| Proyecto final | 40% | Repositorio, modelos, evaluación, demo y model card |
| Participación/revisión de pares | 5% | Pull request y feedback técnico |

**Criterio de aprobación sugerido:** 70/100 y entrega reproducible del proyecto.

## 7. Agenda maestra de las cuatro sesiones

> Cada sesión contempla 480 minutos. La propuesta incluye 60 minutos de pausas, para 420 minutos efectivos de aprendizaje y práctica. Puede adaptarse a la política de la institución.

| Sesión | Eje | Conceptos | Laboratorio principal | Producto del día |
|---|---|---|---|---|
| 1 | Fundamentos | Tensores, MLP, activaciones, loss, gradientes, backprop, training loop | MLP sobre `make_moons` | Notebook con frontera, métricas y curvas |
| 2 | Visión y entrenamiento robusto | CNN, convolución, regularización, optimizadores, ResNet, transfer learning | CNN sobre FashionMNIST | Comparación controlada de dos runs |
| 3 | Secuencias y Transformers | Embeddings, RNN/LSTM, atención, multi-head, positional encoding, Transformer | Atención y bloque Transformer desde cero | Heatmap, pruebas y reflexión |
| 4 | Modelos preentrenados y proyecto | Hugging Face, fine-tuning, PEFT, evaluación, riesgos, entrega | DistilBERT + proyecto integrador | Repositorio, demo y model card |

## 8. Preparación técnica

### 8.1 Hardware

Configuración mínima por estudiante:

- CPU de 4 núcleos o más.
- 16 GB RAM recomendados; 8 GB puede funcionar con batches pequeños.
- 10–20 GB libres.
- GPU NVIDIA opcional. Apple Silicon puede usar MPS cuando las operaciones sean compatibles.
- Conexión a internet para descargar datasets/checkpoints antes de la clase.

Plan alternativo:

- Instructor mantiene una copia cacheada de datasets y modelos.
- Reducir el tamaño de entrenamiento mediante `select` o subconjuntos estratificados.
- Usar un runtime remoto institucional, Google Colab o Kaggle solo como contingencia; el flujo principal permanece en VS Code y GitHub.

### 8.2 Software

- VS Code.
- Extensiones Python y Jupyter.
- Git.
- Python 3.11 o 3.12.
- Cuenta GitHub.
- Cuenta Hugging Face opcional; necesaria solo para repositorios privados o `push_to_hub`.

VS Code permite trabajar con notebooks, seleccionar kernels, inspeccionar variables, depurar y conectarse a servidores Jupyter remotos. Por seguridad, los estudiantes deben abrir únicamente workspaces confiables y revisar código antes de ejecutarlo.

### 8.3 Creación del entorno

```bash
# Crear repositorio y entorno
mkdir deep-learning-course
cd deep-learning-course
git init
python -m venv .venv

# Activar
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Instalar PyTorch desde el selector oficial según CPU/CUDA. Después:

```bash
pip install \
  torchvision \
  numpy pandas matplotlib scikit-learn \
  jupyter ipykernel tqdm pillow \
  transformers datasets evaluate accelerate peft \
  gradio tensorboard \
  ruff pytest

python -m ipykernel install --user \
  --name deep-learning-course \
  --display-name "Deep Learning Course"
```

Para una cohorte, el instructor debe congelar una combinación validada de versiones una semana antes de la clase. Como referencia de documentación en julio de 2026, PyTorch publica documentación 2.12 y Transformers mantiene documentación 5.x. No se recomienda fijar una build de PyTorch idéntica para todos sin considerar CPU, CUDA o MPS.

### 8.4 Verificación del entorno

```python
import sys
import torch
import transformers

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('CUDA:', torch.cuda.is_available())
print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
```

### 8.5 Selección de dispositivo

```python
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')
```

### 8.6 Estructura recomendada del repositorio

```text
deep-learning-course/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── mlp.yaml
│   ├── cnn.yaml
│   └── transformer.yaml
├── notebooks/
│   ├── 01_tensors_autograd.ipynb
│   ├── 02_mlp_training.ipynb
│   ├── 03_cnn_fashionmnist.ipynb
│   ├── 04_sequences_rnn.ipynb
│   ├── 05_attention_from_scratch.ipynb
│   └── 06_hf_finetuning.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   ├── test_shapes.py
│   └── test_smoke.py
├── reports/
│   ├── figures/
│   └── final_report.md
└── app/
    └── gradio_app.py
```

### 8.7 `.gitignore` mínimo

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.env
*.pyc
runs/
checkpoints/
models/
data/raw/
.DS_Store
```

No subir tokens, credenciales, datos personales ni checkpoints grandes directamente al repositorio. Usar variables de entorno, Git LFS o repositorios especializados cuando corresponda.

---

# SESIÓN 1 — Fundamentos de Deep Learning, MLP y backpropagation

## Objetivos de la sesión

- Dominar tensores, shapes, dispositivos y vectorización.
- Explicar una neurona, una MLP y las funciones de activación.
- Relacionar función de pérdida, gradiente, regla de la cadena y backpropagation.
- Escribir un ciclo de entrenamiento y evaluación manual en PyTorch.
- Construir un clasificador MLP reproducible y analizar su frontera de decisión.

## Agenda de 8 horas

| Bloque | Duración | Actividad |
|---|---:|---|
| Apertura, diagnóstico y mapa del curso | 30 min | Quiz, expectativas y contratos de trabajo |
| Tensores, shapes, vectorización y device | 75 min | Concepto + mini-ejercicios + demo |
| Pausa | 15 min | — |
| Neurona, MLP, activaciones y salidas | 75 min | Slides + visualizaciones |
| Loss, gradiente, regla de la cadena y backprop | 90 min | Derivación guiada + autograd |
| Almuerzo/pausa | 30 min | — |
| Training loop, generalización y métricas | 60 min | Demo instrumentada |
| Laboratorio 1 | 90 min | MLP para `make_moons` |
| GitHub checkpoint, quiz y cierre | 15 min | Commit, push y exit ticket |

## Fórmulas esenciales

### Neurona lineal

$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

### Capa densa

$$
\mathbf{Z}^{(l)} = \mathbf{H}^{(l-1)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{H}^{(l)} = \phi\left(\mathbf{Z}^{(l)}\right)
$$

Para batch-first, si `H` tiene shape `(B, d_in)` y `W` shape `(d_in, d_out)`, la salida tiene `(B, d_out)`.

### Activaciones

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

### Softmax

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

En implementación se resta `max(z)` para mejorar estabilidad numérica.

### MSE

$$
\mathcal{L}_{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

### Binary cross-entropy

$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_i \left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

En PyTorch, preferir `BCEWithLogitsLoss`, que combina sigmoid y BCE de forma estable.

### Cross-entropy multiclase

$$
\mathcal{L}_{CE}=-\frac{1}{N}\sum_i \log p(y_i\mid x_i)
$$

`CrossEntropyLoss` recibe **logits**, no probabilidades ni one-hot labels.

### Descenso por gradiente

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta \mathcal{L}(\theta_t)
$$

### Regla de la cadena

Si $y=f(u)$ y $u=g(x)$:

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
$$

## Errores conceptuales que el instructor debe anticipar

- Confundir dimensión del batch con número de features.
- Aplicar softmax antes de `CrossEntropyLoss`.
- Olvidar `optimizer.zero_grad()` y acumular gradientes sin intención.
- Evaluar sin `model.eval()` o sin desactivar gradientes.
- Mover el modelo a GPU, pero no los datos, o viceversa.
- Reportar solo accuracy sin inspeccionar desbalance o errores.
- Usar el test set para elegir hiperparámetros.

## Slides propuestos


| # | Título del slide | Contenido esencial | Visualización, demostración o actividad |
|---:|---|---|---|
| 1 | **Portada y pregunta detonante** | Deep Learning: fundamentos, arquitecturas y Transformers. Pregunta: ¿qué debe aprender una red y qué debemos especificar nosotros? | Imagen conceptual datos → representación → decisión. Encuesta inicial. |
| 2 | **Mapa de las 32 horas** | Cuatro sesiones, laboratorios, checkpoints y proyecto integrador. | Línea de tiempo de las cuatro sesiones. |
| 3 | **Resultados de aprendizaje** | Construir, entrenar, evaluar, diagnosticar y comunicar modelos; explicar atención y Transformers; usar GitHub. | Matriz concepto–código–evidencia. |
| 4 | **IA, ML y Deep Learning** | Relación entre reglas, aprendizaje automático y representación aprendida. | Diagrama de conjuntos anidados y ejemplos. |
| 5 | **¿Por qué “deep”?** | Profundidad como composición de transformaciones; representaciones jerárquicas. | Visual de capas: píxeles → bordes → formas → objetos. |
| 6 | **Cuándo usar Deep Learning** | Datos no estructurados, patrones complejos, suficiente volumen/transfer learning; cuándo no usarlo. | Tabla de decisión DL vs modelos clásicos. |
| 7 | **Flujo de un proyecto DL** | Problema, datos, split, baseline, modelo, entrenamiento, evaluación, errores, entrega. | Pipeline reproducible de extremo a extremo. |
| 8 | **Notación supervisada** | Dataset D={(xᵢ,yᵢ)}, modelo fθ, predicción ŷ, pérdida L, riesgo empírico. | Una muestra viaja por el pipeline. |
| 9 | **Tensores: intuición** | Escalar, vector, matriz y tensor; dimensiones y ejes. | Cubos con shapes (), (d,), (n,d), (b,c,h,w). |
| 10 | **Shapes como contrato** | Batch, features, channels, sequence length; errores comunes por incompatibilidad. | Ejercicio de predecir shapes. |
| 11 | **Vectorización** | Operaciones por lotes, broadcasting y aceleración. | Comparar bucle Python vs operación tensorial. |
| 12 | **Dispositivo y precisión** | CPU, GPU, MPS; float32, float16/bfloat16; mover modelo y datos juntos. | Demo de device detection. |
| 13 | **Neurona lineal** | z=wᵀx+b; pesos como importancia y sesgo como desplazamiento. | Plano de decisión 2D. |
| 14 | **Perceptrón** | Clasificación lineal y función escalón; limitación XOR. | Animación conceptual de frontera lineal. |
| 15 | **De perceptrón a MLP** | Capas densas + no linealidad permiten fronteras complejas. | Red 2–4–2 y regiones de decisión. |
| 16 | **Forward pass** | Composición h¹=φ(W¹x+b¹), ŷ=g(W²h¹+b²). | Diagrama de flujo con shapes. |
| 17 | **Funciones de activación** | Sigmoid, tanh, ReLU, GELU; saturación y rango. | Gráficas superpuestas generadas en Python. |
| 18 | **ReLU en profundidad** | max(0,z), gradiente por tramos, sparsity, “dead ReLU”. | Gráfica y ejemplos numéricos. |
| 19 | **Salida según la tarea** | Regresión: identidad; binaria: logit/sigmoid; multiclase: logits/softmax. | Mapa tarea → capa de salida → loss. |
| 20 | **Softmax** | Convierte logits en distribución; estabilidad numérica; invariancia a traslación. | Cambiar temperatura y observar probabilidades. |
| 21 | **Función de pérdida** | Señal de aprendizaje; diferencia entre métrica y loss. | Analogía brújula vs tablero de resultados. |
| 22 | **MSE** | L=(1/n)Σ(y−ŷ)²; penalización cuadrática y sensibilidad a outliers. | Parábola de error. |
| 23 | **Binary cross-entropy** | Entropía cruzada para Bernoulli; usar logits para estabilidad. | Curvas de penalización para y=0 e y=1. |
| 24 | **Cross-entropy multiclase** | L=−log p(y\|x); relación con máxima verosimilitud. | Probabilidad correcta vs pérdida. |
| 25 | **Optimización** | Objetivo minθ J(θ); superficie de pérdida y trayectorias. | Mapa topográfico con SGD. |
| 26 | **Gradiente** | Dirección de máximo crecimiento; actualización θ←θ−η∇J. | Vector tangente sobre una curva. |
| 27 | **Regla de la cadena** | Derivadas locales se multiplican a través de la composición. | Árbol de dependencias sencillo. |
| 28 | **Grafo computacional** | Nodos, operaciones, valores hacia adelante y gradientes hacia atrás. | Construir y anotar z=(wx+b)². |
| 29 | **Backpropagation** | Cálculo eficiente de gradientes desde la pérdida hacia parámetros. | Forward en azul, backward en sentido inverso. |
| 30 | **Autograd en PyTorch** | requires_grad, backward, grad, zero_grad/no_grad. | Demo de 8 líneas. |
| 31 | **Training loop** | train mode, forward, loss, zero_grad, backward, step; evaluación separada. | Pseudocódigo universal. |
| 32 | **Batch, iteración y epoch** | Trade-off entre ruido, memoria y velocidad. | Dataset dividido en mini-batches. |
| 33 | **Learning rate** | Demasiado alto diverge; demasiado bajo avanza lentamente. | Tres curvas de pérdida. |
| 34 | **Inicialización** | Simetría, escalamiento Xavier/He, activaciones y gradientes. | Histograma por capa. |
| 35 | **Generalización y overfitting** | Brecha train–validation; capacidad, datos y ruido. | Curvas de aprendizaje típicas. |
| 36 | **Regularización** | Weight decay, dropout, early stopping, data augmentation. | Matriz método–efecto–riesgo. |
| 37 | **Splits y fuga de información** | Train/validation/test; estratificación; ajustar preprocessing solo con train. | Caso de leakage para discusión. |
| 38 | **Métricas de clasificación** | Accuracy, precision, recall, F1, matriz de confusión; umbral. | Matriz de confusión interactiva. |
| 39 | **Reproducibilidad** | Semillas, versiones, configuración, checkpoints, determinismo razonable. | Checklist de experimento. |
| 40 | **Git/GitHub para experimentos** | Branch, commit pequeño, README, issues, pull request; no versionar secretos ni checkpoints grandes. | Flujo clone → branch → commit → PR. |
| 41 | **Laboratorio 1** | Tensores, autograd, MLP para two-moons, curvas y fronteras. | Pair programming y checkpoints. |
| 42 | **Cierre y mini-evaluación** | Explicar forward, loss, backward y update sin mirar notas. | Exit ticket de cinco preguntas. |


## Demostración 1 — Tensores y broadcasting

```python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])  # (2, 3)

w = torch.tensor([0.2, -0.1, 0.5])  # (3,)
b = torch.tensor(0.3)                # escalar

logits = x @ w + b                    # broadcasting de b
print(x.shape, w.shape, logits.shape)
print(logits)
```

Preguntas:

1. ¿Qué representa cada eje de `x`?
2. ¿Por qué `x @ w` produce shape `(2,)`?
3. ¿Qué cambiaría si `w` fuera `(3, 2)`?

## Demostración 2 — Autograd visible

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y_true = torch.tensor(10.0)

y_pred = w * x + b
loss = (y_pred - y_true) ** 2
loss.backward()

print('y_pred:', y_pred.item())
print('loss:', loss.item())
print('dL/dw:', w.grad.item())
print('dL/db:', b.grad.item())
```

Derivación manual:

$$
\hat y=wx+b,\quad L=(\hat y-y)^2
$$

$$
\frac{\partial L}{\partial w}=2(\hat y-y)x
$$

$$
\frac{\partial L}{\partial b}=2(\hat y-y)
$$

Los estudiantes deben verificar que los valores manuales coincidan con `autograd`.

## Visualización — Activaciones y derivadas

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 500)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, label='sigmoid')
plt.plot(z, tanh, label='tanh')
plt.plot(z, relu, label='ReLU')
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel('z')
plt.ylabel('activation')
plt.legend()
plt.title('Funciones de activación')
plt.show()
```

Actividad: pedir a los estudiantes que agreguen GELU con `torch.nn.functional.gelu` y comparen suavidad y rango.

## Laboratorio 1 — MLP para clasificación no lineal

### Pregunta experimental

> ¿Cómo cambia la frontera de decisión al aumentar la capacidad de una MLP y qué evidencia indica overfitting?

### Código de referencia

```python
from __future__ import annotations

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


X, y = make_moons(n_samples=1500, noise=0.22, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


class MoonMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    f1: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            total_loss += loss.item() * features.size(0)
            all_labels.extend(labels.long().cpu().numpy().ravel().tolist())
            all_predictions.extend(predictions.cpu().numpy().ravel().tolist())

    return EpochMetrics(
        loss=total_loss / len(loader.dataset),
        accuracy=accuracy_score(all_labels, all_predictions),
        f1=f1_score(all_labels, all_predictions),
    )


model = MoonMLP(hidden_dim=32, dropout=0.10).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
best_val_loss = float('inf')
best_state = None
patience = 20
stale_epochs = 0

for epoch in range(1, 301):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer)
    val_metrics = run_epoch(model, val_loader, criterion)

    history['train_loss'].append(train_metrics.loss)
    history['val_loss'].append(val_metrics.loss)
    history['train_f1'].append(train_metrics.f1)
    history['val_f1'].append(val_metrics.f1)

    if val_metrics.loss < best_val_loss - 1e-4:
        best_val_loss = val_metrics.loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stale_epochs = 0
    else:
        stale_epochs += 1

    if epoch % 25 == 0:
        print(
            f'Epoch {epoch:03d} | '
            f'train loss={train_metrics.loss:.4f}, f1={train_metrics.f1:.3f} | '
            f'val loss={val_metrics.loss:.4f}, f1={val_metrics.f1:.3f}'
        )

    if stale_epochs >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

if best_state is None:
    raise RuntimeError('No best model state was captured.')

model.load_state_dict(best_state)
model.to(device)
test_metrics = run_epoch(model, test_loader, criterion)
print('Test:', test_metrics)


# Curvas
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history['train_loss'], label='train')
ax.plot(history['val_loss'], label='validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE loss')
ax.set_title('Curvas de pérdida')
ax.legend()
plt.show()


# Frontera de decisión
x_min, x_max = X_train[:, 0].min() - 0.7, X_train[:, 0].max() + 0.7
y_min, y_max = X_train[:, 1].min() - 0.7, X_train[:, 1].max() + 0.7
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]

with torch.inference_mode():
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    grid_prob = torch.sigmoid(model(grid_tensor)).cpu().numpy().reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, grid_prob, levels=20, alpha=0.6)
plt.contour(xx, yy, grid_prob, levels=[0.5], linewidths=2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', s=25)
plt.title('Frontera de decisión de la MLP')
plt.xlabel('x1 estandarizada')
plt.ylabel('x2 estandarizada')
plt.show()
```

### Experimentos obligatorios

Cada equipo ejecuta dos variantes cambiando **una sola variable**:

- `hidden_dim`: 4 vs 64;
- dropout: 0 vs 0.4;
- weight decay: 0 vs `1e-3`;
- learning rate: `1e-4` vs `1e-2`;
- una capa oculta vs cuatro capas.

### Evidencia a entregar

- Tabla de configuración.
- Curvas train/validation.
- F1 y matriz de confusión en test.
- Frontera de decisión.
- Conclusión de máximo 150 palabras: hipótesis, evidencia, limitación y decisión.
- Commit con mensaje: `feat: complete mlp experiment`.

## Exit ticket de la sesión 1

1. ¿Por qué una red sin activaciones no lineales equivale a una transformación lineal?
2. ¿Por qué `CrossEntropyLoss` debe recibir logits?
3. ¿Qué ocurre si no se limpian los gradientes?
4. ¿Qué diferencia hay entre `model.train()` y `model.eval()`?
5. ¿Qué evidencia permite distinguir underfitting y overfitting?

---

# SESIÓN 2 — CNN, optimización, regularización y transfer learning

## Objetivos de la sesión

- Explicar convolución, kernels, feature maps, stride, padding y receptive field.
- Calcular shapes y número de parámetros de una CNN.
- Entrenar una CNN y diagnosticar errores con curvas y matriz de confusión.
- Comparar regularización, optimizadores y schedules mediante experimentos controlados.
- Comprender conexiones residuales y aplicar transfer learning.

## Agenda de 8 horas

| Bloque | Duración | Actividad |
|---|---:|---|
| Recap y reto de shapes | 20 min | Quiz en parejas |
| Convolución, kernels, padding y stride | 80 min | Cálculos y visualización |
| Pausa | 15 min | — |
| Arquitectura CNN, DataLoader y shape tracing | 70 min | Demo |
| Entrenamiento, métricas y análisis de errores | 60 min | Curvas y confusion matrix |
| Almuerzo/pausa | 30 min | — |
| Regularización, optimizadores y schedules | 65 min | Experimentos comparados |
| ResNet y transfer learning | 45 min | Concepto + demo |
| Laboratorio 2 | 80 min | FashionMNIST |
| Cierre | 15 min | Commit y exit ticket |

## Fórmulas esenciales

### Convolución 2D simplificada

$$
Y[i,j]=\sum_m\sum_n X[i+m,j+n]K[m,n]+b
$$

En una CNN real se suman también los canales de entrada.

### Shape de salida

$$
H_{out}=\left\lfloor\frac{H_{in}+2P-D(K-1)-1}{S}\right\rfloor+1
$$

La misma fórmula aplica para el ancho.

### Parámetros de Conv2d

$$
\#\theta=C_{out}(C_{in}K_hK_w+1)
$$

El término `+1` corresponde a un bias por canal de salida, si `bias=True`.

### Batch Normalization

Para activaciones de un mini-batch:

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

$$
y=\gamma\hat x+\beta
$$

### Momentum

$$
v_t=\beta v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t
$$

### Conexión residual

$$
y=F(x;\theta)+x
$$

## Slides propuestos


| # | Título del slide | Contenido esencial | Visualización, demostración o actividad |
|---:|---|---|---|
| 1 | **Apertura: del MLP a la visión** | Por qué aplanar una imagen destruye estructura espacial. | Comparar imagen 28×28 y vector de 784. |
| 2 | **Imagen como tensor** | H×W, canales, batch; convención NCHW en PyTorch. | Descomponer una imagen RGB. |
| 3 | **Convolución: intuición** | Ventana local que detecta un patrón y se desplaza. | Kernel recorriendo una matriz. |
| 4 | **Convolución discreta** | Y[i,j]=ΣₘΣₙ X[i+m,j+n]K[m,n]+b. | Cálculo manual de una posición. |
| 5 | **Kernels clásicos** | Bordes, blur y sharpening como antesala de filtros aprendidos. | Aplicar tres kernels con Python. |
| 6 | **Filtros aprendidos** | Los pesos del kernel se optimizan por backpropagation. | Filtros de primera capa después de entrenar. |
| 7 | **Feature maps** | Cada filtro produce un mapa; múltiples canales de salida. | Input C_in → C_out feature maps. |
| 8 | **Stride** | Controla el salto, reduce resolución y costo. | Mismo input con stride 1 y 2. |
| 9 | **Padding** | Preserva bordes/tamaño o controla reducción. | Valid vs same. |
| 10 | **Tamaño de salida** | H_out=⌊(H+2P−D(K−1)−1)/S⌋+1. | Ejercicio de shapes. |
| 11 | **Canales y parámetros** | Parámetros Conv2d=C_out(C_in K_h K_w+1). | Comparar con capa fully connected. |
| 12 | **Pooling** | Max/average pooling, invariancia local y reducción. | Ventana 2×2 sobre feature map. |
| 13 | **Receptive field** | Qué región original influye una activación profunda. | Crecimiento por capas. |
| 14 | **Arquitectura CNN básica** | Conv → activación → pooling → bloques → head. | Diagrama con dimensions. |
| 15 | **FashionMNIST** | 10 clases, 28×28 escala de grises; baseline didáctico. | Mosaico de ejemplos. |
| 16 | **Dataset y DataLoader** | Transformaciones, batching, shuffle, workers. | Inspección de un batch. |
| 17 | **Normalización de entradas** | Centrar/escalar canales; calcular estadísticas solo en train. | Histogramas antes/después. |
| 18 | **Primer modelo CNN** | Dos bloques convolucionales y clasificador. | Shape tracing en cada capa. |
| 19 | **Diagnóstico de shapes** | Flatten seguro, AdaptiveAvgPool y forward hooks. | Actividad de encontrar un mismatch. |
| 20 | **Entrenamiento en GPU** | Transferir batch, non_blocking cuando aplica, mixed precision opcional. | Comparar tiempo CPU/GPU. |
| 21 | **Curvas de aprendizaje** | Loss y métrica por epoch; señales de under/overfitting. | Gráfica train vs val. |
| 22 | **Matriz de confusión** | Patrones de errores entre clases similares. | Heatmap normalizado. |
| 23 | **Inspección de errores** | Top errores por confianza; etiqueta dudosa vs límite del modelo. | Galería de falsos positivos. |
| 24 | **Data augmentation** | Transformaciones plausibles preservan la etiqueta. | Antes/después de crops, flips, rotations. |
| 25 | **Dropout** | Máscara Bernoulli en entrenamiento; expectativa en inferencia. | Red antes y después de dropout. |
| 26 | **Batch Normalization** | Normalización por mini-batch + γ,β; diferencias train/eval. | Distribuciones de activación. |
| 27 | **LayerNorm vs BatchNorm** | Ejes normalizados y uso típico: CNN vs Transformers. | Tensor con ejes resaltados. |
| 28 | **SGD con momentum** | Acumula dirección; puede generalizar bien; requiere LR tuning. | Trayectorias con y sin momentum. |
| 29 | **Adam/AdamW** | Momentos adaptativos; weight decay desacoplado en AdamW. | Tabla de elección práctica. |
| 30 | **Learning-rate schedules** | Step, cosine, one-cycle, warmup; el LR es una función del tiempo. | Curvas de schedules. |
| 31 | **Early stopping y checkpoint** | Guardar mejor validación, paciencia y restauración. | State machine simple. |
| 32 | **Weight decay** | Penalización sobre pesos y control de capacidad. | Efecto del λ en fronteras/pesos. |
| 33 | **Conexiones residuales** | y=F(x)+x facilita optimización y flujo de gradientes. | Bloque residual. |
| 34 | **ResNet** | Stacks de bloques residuales; proyección cuando cambian shapes. | Arquitectura simplificada. |
| 35 | **Transfer learning** | Features preentrenadas, reemplazar head, congelar y descongelar. | Pretraining → downstream task. |
| 36 | **Estrategias de fine-tuning** | Head only, últimas capas, full fine-tuning; LR diferencial. | Matriz datos–cómputo–estrategia. |
| 37 | **Interpretabilidad visual** | Saliency/Grad-CAM como evidencia aproximada, no explicación causal. | Mapa de calor sobre imagen. |
| 38 | **Tracking de experimentos** | Config, métricas, artifacts y comparación; CSV/TensorBoard/W&B opcional. | Tabla de runs. |
| 39 | **Laboratorio 2** | CNN en FashionMNIST + regularización + comparación de dos experimentos. | Equipos diseñan una hipótesis. |
| 40 | **Reto de transferencia** | Fine-tune ResNet18 sobre subconjunto de CIFAR-10 o dataset propio. | Checklist de congelamiento y LR. |
| 41 | **Cierre** | Explicar por qué CNN reduce parámetros y preserva localidad. | Quiz de shapes y diagnóstico. |


## Visualización — Aplicar kernels manualmente

```python
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

fashion = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transforms.ToTensor()
)
image, label = fashion[0]  # (1, 28, 28)

kernels = {
    'horizontal_edges': torch.tensor([
        [-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.],
    ]),
    'vertical_edges': torch.tensor([
        [-1., 0., 1.],
        [-1., 0., 1.],
        [-1., 0., 1.],
    ]),
    'sharpen': torch.tensor([
        [ 0., -1.,  0.],
        [-1.,  5., -1.],
        [ 0., -1.,  0.],
    ]),
}

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for axis, (name, kernel) in zip(axes[1:], kernels.items()):
    weight = kernel.view(1, 1, 3, 3)
    output = F.conv2d(image.unsqueeze(0), weight, padding=1)
    axis.imshow(output.squeeze(), cmap='gray')
    axis.set_title(name)
    axis.axis('off')

plt.tight_layout()
plt.show()
```

Mensaje docente: estos kernels fueron diseñados manualmente; en una CNN, el algoritmo aprende sus valores para minimizar la pérdida.

## Laboratorio 2 — CNN para FashionMNIST

```python
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

full_train_aug = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=train_transform
)
full_train_eval = datasets.FashionMNIST(
    root='data', train=True, download=True, transform=eval_transform
)
test_dataset = datasets.FashionMNIST(
    root='data', train=False, download=True, transform=eval_transform
)

# Índices compartidos para que train y validation usen transforms diferentes.
generator = torch.Generator().manual_seed(42)
train_size = 54_000
val_size = len(full_train_aug) - train_size
train_subset, val_indices_subset = random_split(
    range(len(full_train_aug)), [train_size, val_size], generator=generator
)
train_indices = list(train_subset)
val_indices = list(val_indices_subset)

train_dataset = torch.utils.data.Subset(full_train_aug, train_indices)
val_dataset = torch.utils.data.Subset(full_train_eval, val_indices)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)


class FashionCNN(nn.Module):
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class Metrics:
    loss: float
    accuracy: float


def run_epoch(model, loader, criterion, optimizer=None) -> Metrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return Metrics(
        loss=total_loss / len(loader.dataset),
        accuracy=total_correct / len(loader.dataset),
    )


model = FashionCNN(dropout=0.25).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

best_state = copy.deepcopy(model.state_dict())
best_val = float('inf')
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, 13):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer)
    val_metrics = run_epoch(model, val_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_metrics.loss)
    history['val_loss'].append(val_metrics.loss)
    history['train_acc'].append(train_metrics.accuracy)
    history['val_acc'].append(val_metrics.accuracy)

    if val_metrics.loss < best_val:
        best_val = val_metrics.loss
        best_state = copy.deepcopy(model.state_dict())

    print(
        f'Epoch {epoch:02d} | '
        f'train loss {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.3f} | '
        f'val loss {val_metrics.loss:.4f}, acc {val_metrics.accuracy:.3f}'
    )

model.load_state_dict(best_state)
model.to(device)

# Test y predicciones
y_true, y_pred = [], []
model.eval()
with torch.inference_mode():
    for images, labels in test_loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.numpy().tolist())

class_names = test_dataset.classes
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
print(confusion_matrix(y_true, y_pred))
```

### Gráficas de entrenamiento

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history['train_loss'], label='train loss')
ax.plot(history['val_loss'], label='validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('CNN — curvas de pérdida')
plt.show()
```

### Inspección de errores con mayor confianza

```python
records = []
model.eval()
with torch.inference_mode():
    for images, labels in test_loader:
        logits = model(images.to(device))
        probabilities = logits.softmax(dim=1).cpu()
        confidence, predictions = probabilities.max(dim=1)
        for image, label, prediction, conf in zip(images, labels, predictions, confidence):
            if prediction.item() != label.item():
                records.append((conf.item(), image, label.item(), prediction.item()))

records.sort(key=lambda item: item[0], reverse=True)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for axis, (conf, image, label, prediction) in zip(axes.ravel(), records[:10]):
    axis.imshow(image.squeeze(), cmap='gray')
    axis.set_title(
        f'True: {class_names[label]}\nPred: {class_names[prediction]}\nConf: {conf:.2f}',
        fontsize=8,
    )
    axis.axis('off')
plt.tight_layout()
plt.show()
```

## Experimento controlado

Cada equipo selecciona una hipótesis:

- augmentation mejora generalización;
- BatchNorm permite un learning rate mayor;
- AdamW converge más rápido que SGD+momentum en el presupuesto dado;
- dropout alto puede producir underfitting;
- cosine schedule mejora el mejor checkpoint frente a LR constante.

Regla: modificar una variable principal, usar los mismos splits/seed y reportar media de dos runs si el tiempo lo permite.

## Transfer learning con ResNet18 — demo corta

```python
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)
model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable:,} / {total:,}')
```

Para FashionMNIST habría que convertir 1 canal a 3 y redimensionar según los transforms de los pesos. Para una práctica de transferencia, es preferible CIFAR-10 o un dataset de imágenes RGB propio.

## Exit ticket de la sesión 2

1. Calcular output de una conv con `H=28, K=3, P=1, S=2`.
2. ¿Por qué una CNN usa menos parámetros que una capa densa sobre una imagen?
3. ¿Qué cambia en BatchNorm entre entrenamiento e inferencia?
4. ¿Qué evidencia mostraría que augmentation fue demasiado agresivo?
5. ¿Cuándo congelaría el backbone y cuándo haría full fine-tuning?

---

# SESIÓN 3 — Secuencias, atención y arquitectura Transformer

## Objetivos de la sesión

- Comprender tokenización, vocabulario, embeddings y máscaras.
- Explicar RNN, BPTT, LSTM/GRU y sus limitaciones.
- Derivar scaled dot-product attention con shapes.
- Explicar multi-head attention, positional encoding, residuals y LayerNorm.
- Diferenciar encoder, decoder causal y encoder–decoder.
- Implementar y probar atención y un bloque Transformer simplificado.
- Usar Transformer Explainer para conectar la fórmula con un modelo real.

## Agenda de 8 horas

| Bloque | Duración | Actividad |
|---|---:|---|
| Secuencias, tokenización y embeddings | 60 min | Conceptos + visualización |
| RNN, BPTT y gradientes | 60 min | Ecuaciones y demo |
| Pausa | 15 min | — |
| LSTM/GRU, máscaras y clasificación | 55 min | Comparación |
| Motivación y fórmula de atención | 75 min | Cálculo manual y shapes |
| Almuerzo/pausa | 30 min | — |
| Multi-head, posición y bloque Transformer | 75 min | Arquitectura |
| Transformer Explainer | 40 min | Laboratorio visual guiado |
| Laboratorio 3 | 60 min | Atención desde cero |
| Cierre | 10 min | Exit ticket |

## Fórmulas esenciales

### Embedding lookup

Si $E\in\mathbb{R}^{|V|\times d}$ y el token tiene ID $i$:

$$
e_i=E[i]\in\mathbb{R}^{d}
$$

### RNN

$$
h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

$$
y_t=W_{hy}h_t+b_y
$$

### LSTM

$$
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)
$$

$$
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
$$

$$
\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$$

$$
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)
$$

$$
h_t=o_t\odot\tanh(c_t)
$$

### Scaled dot-product attention

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

`M` contiene `0` para posiciones permitidas y un valor muy negativo para posiciones bloqueadas.

### Multi-head attention

$$
\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(head_1,\dots,head_h)W^O
$$

### Positional encoding sinusoidal

$$
PE(pos,2i)=\sin\left(pos/10000^{2i/d_{model}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

### Transformer pre-norm simplificado

$$
x'=x+\operatorname{MHA}(\operatorname{LN}(x))
$$

$$
y=x'+\operatorname{FFN}(\operatorname{LN}(x'))
$$

## Slides propuestos


| # | Título del slide | Contenido esencial | Visualización, demostración o actividad |
|---:|---|---|---|
| 1 | **Apertura: datos secuenciales** | Texto, audio, series de tiempo y eventos; orden y contexto importan. | Ejemplos con la misma bolsa de palabras y distinto significado. |
| 2 | **Problemas de secuencia** | Many-to-one, one-to-many, many-to-many y autoregresión. | Cuatro diagramas. |
| 3 | **Longitud variable** | Padding, truncation, packing y máscaras. | Batch de secuencias desiguales. |
| 4 | **Tokenización** | Palabra, carácter, subword; trade-off vocabulario vs longitud. | Segmentar una frase de tres maneras. |
| 5 | **Vocabulario e IDs** | Token → integer ID → embedding; tokens especiales. | Tabla token/id. |
| 6 | **Embeddings** | Matriz E∈R^{\|V\|×d}; lookup diferenciable. | Tokens como puntos en espacio vectorial. |
| 7 | **Semántica distribuida** | Similitud coseno y analogías como intuición, no garantía. | Proyección 2D con PCA/t-SNE. |
| 8 | **Limitación de bolsa de palabras** | Ignora orden, negación y dependencias. | “no es bueno” vs “es bueno”. |
| 9 | **RNN: estado recurrente** | h_t resume pasado y se actualiza paso a paso. | Celda desplegada en el tiempo. |
| 10 | **Ecuaciones RNN** | h_t=tanh(W_xh x_t+W_hh h_{t−1}+b_h); ŷ_t=W_hy h_t+b_y. | Anotar shapes. |
| 11 | **Backpropagation Through Time** | El grafo se replica por pasos; parámetros compartidos. | Forward y backward a través del tiempo. |
| 12 | **Gradientes que desaparecen/explotan** | Productos repetidos de Jacobianos; dependencias largas difíciles. | Magnitud del gradiente por timestep. |
| 13 | **Gradient clipping** | Limitar norma para controlar explosión; no resuelve vanishing. | Código clip_grad_norm_. |
| 14 | **LSTM: intuición** | Celda de memoria y compuertas para controlar olvidar, escribir y leer. | Diagrama de tubería con compuertas. |
| 15 | **Ecuaciones LSTM** | f_t, i_t, g_t, o_t, c_t y h_t. | Colorear cada compuerta. |
| 16 | **GRU** | Update y reset gates; menos parámetros que LSTM. | Comparación de celdas. |
| 17 | **Bidireccionalidad** | Contexto izquierdo y derecho; no válido para generación causal. | Dos flechas sobre secuencia. |
| 18 | **Clasificación con RNN/LSTM** | Embedding → recurrente → pooling/último estado → head. | Arquitectura many-to-one. |
| 19 | **Máscaras y padding** | Evitar que padding contamine pérdida/atención. | Matriz de máscara binaria. |
| 20 | **Cuello de botella recurrente** | Procesamiento secuencial y dependencia comprimida en un estado. | Comparación throughput RNN vs atención. |
| 21 | **Motivación de atención** | Acceso directo a todas las posiciones relevantes. | Palabra consulta conectada a contexto. |
| 22 | **Query, Key, Value** | Q pregunta, K describe coincidencia y V transporta información. | Analogía buscador documentada. |
| 23 | **Producto punto** | Compatibilidad q·k; similitud direccional dependiente de magnitud. | Vectores y ángulo. |
| 24 | **Scaled dot-product attention** | Attention(Q,K,V)=softmax(QKᵀ/√d_k)V. | Flujo matricial completo. |
| 25 | **Shapes de atención** | Q,K,V: (B,H,T,d_k); scores: (B,H,T,T). | Shape tracing interactivo. |
| 26 | **Por qué escalar** | Evita logits grandes y softmax demasiado saturado al crecer d_k. | Distribuciones con/sin √d_k. |
| 27 | **Máscara causal** | Prohíbe atender al futuro en modelos autoregresivos. | Triángulo inferior. |
| 28 | **Mapa de atención** | Cada fila suma 1; patrón de ruteo, no prueba causal. | Heatmap token-token. |
| 29 | **Multi-head attention** | Varias proyecciones aprenden relaciones complementarias. | Heads paralelos → concat → proyección. |
| 30 | **Posición** | La autoatención sola es permutacional; se añade información posicional. | Secuencia permutada. |
| 31 | **Positional encoding sinusoidal** | PE(pos,2i)=sin(pos/10000^{2i/d}); PE(pos,2i+1)=cos(...). | Ondas por dimensión. |
| 32 | **Bloque Transformer** | MHA + MLP + residual + LayerNorm + dropout. | Bloque pre-norm simplificado. |
| 33 | **MLP por token** | Transformación feed-forward independiente en cada posición. | Atención mezcla tokens; MLP transforma features. |
| 34 | **Encoder** | Atención bidireccional y representaciones contextuales. | Pila de encoders. |
| 35 | **Decoder causal** | Masked self-attention y generación del siguiente token. | Pila GPT-like. |
| 36 | **Encoder–decoder** | Self-attention, cross-attention y salida autoregresiva. | Arquitectura de traducción. |
| 37 | **BERT vs GPT** | Encoder/MLM vs decoder/causal LM; comprensión vs generación como simplificación. | Tabla de objetivos y máscaras. |
| 38 | **Complejidad** | Self-attention O(T²d), RNN O(Td²) secuencial; matices por longitud/hardware. | Gráfica conceptual costo vs T. |
| 39 | **Transformer Explainer** | GPT-2 small, embeddings, 12 heads, máscara, logits y sampling. | Recorrido guiado en la herramienta. |
| 40 | **Temperatura, top-k y top-p** | Transformar distribución de salida y controlar diversidad. | Experimento con el mismo prompt. |
| 41 | **Implementar atención** | Crear scaled_dot_product_attention y verificar shapes/suma de pesos. | Notebook desde cero. |
| 42 | **Implementar un bloque** | MultiheadAttention + LayerNorm + FFN + residual. | Comparar salida y gradientes. |
| 43 | **Inspeccionar atención** | Capturar pesos, visualizar heatmap y discutir límites. | Actividad de interpretación crítica. |
| 44 | **Cierre** | Explicar cómo un token obtiene contexto y cómo se impide ver el futuro. | Mapa conceptual colectivo. |


## Actividad manual — Una fila de atención

Usar tres tokens con $d_k=2$:

$$
Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}1&2\\3&0\\0&4\end{bmatrix}
$$

Los estudiantes calculan:

1. `scores = Q @ K.T`;
2. `scores / sqrt(2)`;
3. softmax de la primera fila;
4. combinación ponderada de `V`;
5. efecto de una máscara causal.

## Laboratorio visual — Transformer Explainer

URL: `https://poloclub.github.io/transformer-explainer/`

### Guion de 35–40 minutos

1. Ingresar un prompt corto y observar tokenización.
2. Inspeccionar token embeddings y positional embeddings.
3. Confirmar que GPT-2 small usa múltiples bloques y 12 attention heads.
4. Elegir una cabeza y comparar pesos para dos tokens.
5. Observar la máscara causal: ninguna posición puede consultar tokens futuros.
6. Seguir el flujo de atención: `QKᵀ`, escala, máscara, softmax y multiplicación por `V`.
7. Inspeccionar MLP, residuals y LayerNorm.
8. Observar logits y distribución de next token.
9. Cambiar temperatura manteniendo prompt constante.
10. Comparar top-k y top-p y registrar qué cambia.

### Preguntas de discusión

- ¿Una cabeza representa siempre una relación lingüística interpretable?
- ¿Un peso alto de atención demuestra causalidad?
- ¿Qué parte del bloque mezcla información entre tokens?
- ¿Qué parte transforma cada token de manera independiente?
- ¿Por qué una máscara causal es indispensable para entrenar next-token prediction?

## Laboratorio 3A — Scaled dot-product attention desde cero

```python
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shapes esperados:
      query: (..., T_q, d_k)
      key:   (..., T_k, d_k)
      value: (..., T_k, d_v)
      mask:  broadcastable a (..., T_q, T_k), True=permitido
    """
    if query.size(-1) != key.size(-1):
        raise ValueError('query y key deben compartir d_k')
    if key.size(-2) != value.size(-2):
        raise ValueError('key y value deben compartir T_k')

    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        weights = torch.nn.functional.dropout(weights, p=dropout_p, training=training)

    output = weights @ value
    return output, weights


# B=1, H=1, T=4, d_k=d_v=4
torch.manual_seed(42)
q = torch.randn(1, 1, 4, 4, requires_grad=True)
k = torch.randn(1, 1, 4, 4, requires_grad=True)
v = torch.randn(1, 1, 4, 4, requires_grad=True)

causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool)).view(1, 1, 4, 4)
output, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask)

assert output.shape == (1, 1, 4, 4)
assert weights.shape == (1, 1, 4, 4)
assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, 4), atol=1e-6)
assert torch.all(weights.masked_select(~causal_mask) < 1e-6)

loss = output.square().mean()
loss.backward()
assert q.grad is not None and torch.isfinite(q.grad).all()

plt.figure(figsize=(5, 4))
plt.imshow(weights.detach().squeeze().numpy())
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.title('Causal attention weights')
plt.colorbar()
plt.show()
```

### Comparación con PyTorch

```python
reference = torch.nn.functional.scaled_dot_product_attention(
    q.detach(), k.detach(), v.detach(), attn_mask=causal_mask
)
print(torch.max(torch.abs(reference - output.detach())))
```

Nota: el comportamiento exacto de máscaras y kernels optimizados puede variar por versión/dispositivo. La comparación debe validar shapes y proximidad numérica en el entorno de la clase.

## Laboratorio 3B — Multi-head self-attention y bloque Transformer

```python
from __future__ import annotations

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model debe ser divisible por num_heads')

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))  # (B, H, T, d_k)

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        if causal:
            mask = torch.tril(mask)
        mask = mask.view(1, 1, seq_len, seq_len)

        if padding_mask is not None:
            # padding_mask: (B, T), True para token válido.
            key_mask = padding_mask[:, None, None, :]
            mask = mask & key_mask

        attended, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout, training=self.training
        )
        merged = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(merged), weights


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
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
        attn_output, weights = self.attention(
            self.norm1(x), causal=causal, padding_mask=padding_mask
        )
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x, weights


torch.manual_seed(42)
x = torch.randn(2, 7, 64, requires_grad=True)
padding_mask = torch.tensor([
    [True, True, True, True, True, False, False],
    [True, True, True, True, True, True, True],
])

block = TransformerBlock(d_model=64, num_heads=4, d_ff=128)
y, attention_weights = block(x, causal=True, padding_mask=padding_mask)

assert y.shape == x.shape
assert attention_weights.shape == (2, 4, 7, 7)

y.mean().backward()
assert x.grad is not None
print('Output:', y.shape, 'Attention:', attention_weights.shape)
```

## Visualización — Positional encoding

```python
import math
import matplotlib.pyplot as plt
import torch


def sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


pe = sinusoidal_encoding(max_len=100, d_model=32)
plt.figure(figsize=(9, 5))
plt.imshow(pe.numpy(), aspect='auto')
plt.xlabel('Embedding dimension')
plt.ylabel('Position')
plt.title('Sinusoidal positional encoding')
plt.colorbar()
plt.show()
```

## Entregable del laboratorio 3

- Implementación de atención con validaciones de shapes.
- Prueba de que cada fila de pesos suma aproximadamente 1.
- Prueba de máscara causal.
- Backpropagation exitoso.
- Heatmap de una cabeza.
- Comparación de dos prompts en Transformer Explainer.
- Reflexión: qué puede inferirse y qué no de un attention map.

## Exit ticket de la sesión 3

1. ¿Qué representan Q, K y V?
2. ¿Por qué se divide por $\sqrt{d_k}$?
3. ¿Cuál es la diferencia entre padding mask y causal mask?
4. ¿Dónde se mezcla información entre tokens y dónde se transforma cada token?
5. ¿Por qué un decoder causal no puede usar información futura durante entrenamiento?

---

# SESIÓN 4 — Hugging Face, fine-tuning, evaluación, entrega y proyecto final

## Objetivos de la sesión

- Seleccionar un checkpoint mediante evidencia, licencia y restricciones.
- Tokenizar, cargar datasets y crear batches con padding dinámico.
- Fine-tunear un Transformer para clasificación con Hugging Face.
- Evaluar con accuracy, macro-F1, matriz de confusión y análisis de errores.
- Introducir PEFT/LoRA, cuantización y eficiencia sin confundirlas con garantías de calidad.
- Documentar reproducibilidad, riesgos, privacidad y limitaciones.
- Completar y presentar el proyecto integrador.

## Agenda de 8 horas

| Bloque | Duración | Actividad |
|---|---:|---|
| Ecosistema Hugging Face y model cards | 45 min | Selección crítica |
| Tokenización, datasets y collators | 60 min | Demo |
| Pausa | 15 min | — |
| Fine-tuning con Trainer | 75 min | Ejecución guiada |
| Evaluación y análisis de errores | 50 min | Métricas y taxonomía |
| Almuerzo/pausa | 30 min | — |
| PEFT, eficiencia, riesgos y entrega | 55 min | Concepto + checklist |
| Proyecto final en clase | 105 min | Equipos con mentoría |
| Demo day y cierre | 45 min | Presentaciones y feedback |

## Slides propuestos


| # | Título del slide | Contenido esencial | Visualización, demostración o actividad |
|---:|---|---|---|
| 1 | **Apertura: de arquitectura a aplicación** | Reutilizar modelos preentrenados y adaptar con evidencia. | Pipeline de transferencia. |
| 2 | **Pretraining y downstream tasks** | Objetivo general a gran escala → especialización con pocos datos. | Embudo pretrain → fine-tune. |
| 3 | **Modelos fundacionales** | Representaciones reutilizables en texto, visión, audio y multimodalidad. | Mapa de modalidades. |
| 4 | **Ecosistema Hugging Face** | Hub, Transformers, Datasets, Tokenizers, Evaluate, Accelerate, PEFT y Spaces. | Diagrama del ecosistema. |
| 5 | **Model card** | Arquitectura, datos, licencia, usos, límites, métricas y riesgos. | Checklist para seleccionar un modelo. |
| 6 | **Selección de checkpoint** | Idioma, tarea, tamaño, licencia, contexto, costo y evidencia. | Matriz de decisión. |
| 7 | **Pipeline API** | Inferencia rápida para validar tarea y contrato de entrada/salida. | Demo de sentiment-analysis. |
| 8 | **Tokenizer y modelo deben corresponder** | Mismo checkpoint para reglas de tokenización, vocabulario y embeddings. | Error conceptual de mezclar tokenizer/model. |
| 9 | **Subwords** | BPE/WordPiece/Unigram; palabras raras se descomponen. | Tokenización de términos técnicos. |
| 10 | **Tokens especiales** | CLS/BOS/EOS/SEP/PAD según arquitectura. | Secuencia anotada. |
| 11 | **Attention mask** | 1 para tokens válidos, 0 para padding; distinta de máscara causal. | Dos máscaras lado a lado. |
| 12 | **AutoClasses** | AutoTokenizer y AutoModelFor... resuelven configuración por checkpoint. | Árbol tarea → clase. |
| 13 | **Logits y probabilidades** | El modelo devuelve logits; aplicar softmax/sigmoid según tarea. | Inspeccionar ModelOutput. |
| 14 | **Datasets** | load_dataset, splits, map, select, filter, shuffle; streaming opcional. | Vista de DatasetDict. |
| 15 | **Dynamic padding** | DataCollatorWithPadding reduce cómputo innecesario por batch. | Padding global vs dinámico. |
| 16 | **Baseline obligatorio** | Modelo simple y métrica base antes del Transformer. | Escalera majority → TF-IDF → MLP → Transformer. |
| 17 | **Fine-tuning** | Actualizar pesos preentrenados con LR pequeño y datos de tarea. | Zona de parámetros que se ajusta. |
| 18 | **TrainingArguments** | Batch, epochs, LR, evaluación, logging, checkpoints y mixed precision. | Ficha de configuración. |
| 19 | **Trainer** | Orquesta loops de train/eval sin ocultar diseño experimental. | Componentes: model, args, datasets, collator, metrics. |
| 20 | **Métricas** | Accuracy y macro-F1; precision/recall por clase; calibración si importa. | Dashboard mínimo. |
| 21 | **Desbalance** | Class weights, sampling, focal loss o recolección; no depender solo de accuracy. | Ejemplo 95/5. |
| 22 | **Validación y test** | Ajustar con validación; test una vez; evitar benchmark overfitting. | Flujo de decisiones. |
| 23 | **Análisis de errores** | Taxonomía: ambigüedad, negación, dominio, truncation, ruido, sesgo. | Tabla error → hipótesis → acción. |
| 24 | **Inferencia por lotes** | model.eval, no_grad/inference_mode, batching y device. | Throughput vs latency. |
| 25 | **Guardar y cargar** | save_pretrained/from_pretrained, tokenizer y configuración juntos. | Estructura del artifact. |
| 26 | **Publicar al Hub** | Repositorio de modelo, versionado y model card; respetar privacidad/licencia. | Flujo push_to_hub opcional. |
| 27 | **PEFT y LoRA** | Adaptadores de bajo rango reducen parámetros entrenables y almacenamiento. | W + ΔW≈W+BA. |
| 28 | **Cuantización** | Menos bits para memoria/inferencia; medir degradación y compatibilidad. | FP32 → BF16/FP16 → INT8/4. |
| 29 | **Presupuesto de cómputo** | Parámetros, activaciones, optimizer states, batch y sequence length. | Hoja de cálculo de memoria conceptual. |
| 30 | **Eficiencia** | Gradient accumulation/checkpointing, mixed precision, truncation y modelos distilled. | Matriz técnica–beneficio–costo. |
| 31 | **Explicabilidad en NLP** | Atención, saliency y perturbaciones son evidencia parcial. | Heatmap con advertencias. |
| 32 | **Sesgo y representatividad** | Datos, etiquetas y despliegue pueden generar daño diferencial. | Preguntas de auditoría. |
| 33 | **Privacidad y seguridad** | PII, memorization, prompt/data leakage, dependencias y supply chain. | Threat model básico. |
| 34 | **Licencias y uso** | Revisar dataset card/model card; restricciones comerciales y atribución. | Checklist legal no vinculante. |
| 35 | **Reproducibilidad de entrega** | requirements/lock, seed, config, commit hash, README, model card. | Tarjeta de experimento. |
| 36 | **Demo con Gradio** | UI mínima para probar entradas, salida, confianza y disclaimers. | Interfaz web local. |
| 37 | **Estructura del repositorio** | src, notebooks, configs, tests, reports, models ignorados, README. | Árbol de carpetas. |
| 38 | **CI mínima** | Lint/test smoke, ejecución de imports y validación de configuración. | GitHub Actions conceptual. |
| 39 | **Proyecto final** | Clasificación de texto: baseline + red propia + Transformer fine-tuned. | Brief del reto. |
| 40 | **Milestones** | Problema/datos → baseline → modelo → evaluación → errores → demo/model card. | Kanban de proyecto. |
| 41 | **Rúbrica** | Corrección, evidencia, reproducibilidad, análisis crítico y comunicación. | Tabla de pesos. |
| 42 | **Demo day** | 7 minutos por equipo + 3 de preguntas; mostrar evidencia, no solo UI. | Plantilla de pitch. |
| 43 | **Retrospectiva técnica** | Qué mejoró, qué falló, qué harían con más datos/cómputo. | Postmortem de una página. |
| 44 | **Cierre del curso** | Del gradiente a Transformers: ideas invariantes y próximos pasos. | Mapa final conectando todos los conceptos. |


## Demostración — Inferencia con `pipeline`

```python
from transformers import pipeline

classifier = pipeline(
    task='sentiment-analysis',
    model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
)

examples = [
    'The course explains attention clearly and the labs are useful.',
    'The installation process was confusing and nothing worked.',
]

for result in classifier(examples):
    print(result)
```

Discusión: `pipeline` valida rápidamente una capacidad, pero no sustituye revisar model card, idioma, licencia, datos, métricas, sesgos y contrato de entrada.

## Fine-tuning de DistilBERT sobre Rotten Tomatoes

El dataset contiene frases de reseñas de películas etiquetadas como positivas o negativas. Es pequeño y apropiado para un laboratorio. Para una cohorte con recursos limitados, usar un subconjunto de entrenamiento y validar primero que todo el pipeline funciona.

```python
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

model_name = 'distilbert/distilbert-base-uncased'
output_dir = Path('artifacts/distilbert-rotten-tomatoes')
output_dir.mkdir(parents=True, exist_ok=True)

raw = load_dataset('rotten_tomatoes')

# Para demo rápida, activar estas líneas:
# raw['train'] = raw['train'].shuffle(seed=42).select(range(2500))
# raw['validation'] = raw['validation'].select(range(500))
# raw['test'] = raw['test'].select(range(500))

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(batch):
    return tokenizer(batch['text'], truncation=True, max_length=256)


tokenized = raw.map(tokenize_batch, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: 'NEGATIVE', 1: 'POSITIVE'},
    label2id={'NEGATIVE': 0, 'POSITIVE': 1},
)

accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='macro',
    )
    return {'accuracy': accuracy['accuracy'], 'macro_f1': f1['f1']}


use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

training_args = TrainingArguments(
    output_dir=str(output_dir),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    save_total_limit=2,
    bf16=use_bf16,
    fp16=use_fp16,
    report_to='none',
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
validation_metrics = trainer.evaluate(tokenized['validation'])
test_output = trainer.predict(tokenized['test'])

predictions = np.argmax(test_output.predictions, axis=-1)
labels = np.asarray(tokenized['test']['label'])

print(validation_metrics)
print(classification_report(labels, predictions, target_names=['NEGATIVE', 'POSITIVE']))
print(confusion_matrix(labels, predictions))

trainer.save_model(str(output_dir / 'best_model'))
tokenizer.save_pretrained(str(output_dir / 'best_model'))

with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as file:
    json.dump(test_output.metrics, file, indent=2)
```

### Compatibilidad

La API de Transformers evoluciona. Esta guía usa `processing_class=tokenizer` y `eval_strategy`, coherentes con documentación 5.x. El instructor debe ejecutar un smoke test antes de la clase y congelar versiones. No habilitar `push_to_hub=True` por defecto en una clase: primero revisar autenticación, privacidad, licencia y model card.

## Análisis de errores

```python
import pandas as pd

probabilities = torch.tensor(test_output.predictions).softmax(dim=-1).numpy()
confidence = probabilities.max(axis=1)

errors = pd.DataFrame({
    'text': raw['test']['text'],
    'label': labels,
    'prediction': predictions,
    'confidence': confidence,
})
errors = errors[errors['label'] != errors['prediction']]
errors = errors.sort_values('confidence', ascending=False)

print(errors.head(20).to_string(index=False))
```

Crear una columna manual `error_type` con categorías:

- negación;
- ironía/sarcasmo;
- frase ambigua;
- conocimiento de dominio;
- etiqueta posiblemente dudosa;
- truncation;
- expresión rara o fuera de distribución;
- error no clasificado.

La conclusión debe recomendar una acción para la categoría dominante: recolectar datos, revisar etiquetas, cambiar arquitectura, ajustar longitud, calibrar umbral o aceptar el límite.

## PEFT/LoRA — extensión demostrativa

LoRA aproxima la actualización de una matriz grande mediante dos matrices de bajo rango:

$$
W'=W+\Delta W,\qquad \Delta W\approx BA
$$

con rango $r$ mucho menor que las dimensiones de $W$.

Ejemplo conceptual para un modelo compatible:

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_lin', 'v_lin'],  # específico de DistilBERT; verificar arquitectura
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

El nombre de los módulos objetivo depende de la arquitectura. Inspeccionar `model.named_modules()` antes de configurarlos. PEFT reduce parámetros entrenables y artifacts, pero no elimina la necesidad de validar datos, métricas y riesgos.

## Demo local con Gradio

```python
import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = 'artifacts/distilbert-rotten-tomatoes/best_model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model.to(device)


def predict(text: str) -> dict[str, float]:
    if not text or not text.strip():
        return {'NEGATIVE': 0.0, 'POSITIVE': 0.0}
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probabilities = model(**encoded).logits.softmax(dim=-1).squeeze(0).cpu()
    return {
        model.config.id2label[index]: float(probability)
        for index, probability in enumerate(probabilities)
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label='Movie review'),
    outputs=gr.Label(num_top_classes=2, label='Prediction'),
    title='Sentiment classifier — educational demo',
    description='Prototype for learning. Validate before any real-world use.',
)

demo.launch()
```

---

# PROYECTO FINAL EN CLASE

## Nombre

**Clasificador de texto reproducible: baseline, red neuronal y Transformer**

## Contexto

Una organización necesita clasificar mensajes o reseñas para priorizar análisis y respuesta. El equipo debe demostrar que una solución basada en Deep Learning aporta valor frente a un baseline, y debe documentar límites, errores y condiciones de uso.

## Dataset principal recomendado

`rotten_tomatoes` de Hugging Face para clasificación binaria de sentimiento.

Ventajas didácticas:

- tamaño manejable;
- splits disponibles;
- dos clases relativamente balanceadas;
- compatible con el ejemplo oficial de Transformers;
- permite comparar modelos bajo un mismo contrato.

### Alternativas

- `ag_news`: clasificación multiclase de noticias, si la cohorte tiene más cómputo.
- `mteb/amazon_reviews_multi` o un subconjunto español, después de verificar disponibilidad, licencia, schema y costo de descarga.
- Dataset institucional anonimizado, solo si existe autorización, documentación de origen y control de privacidad.

## Pregunta central

> ¿Qué combinación de representación, arquitectura y estrategia de entrenamiento ofrece el mejor equilibrio entre desempeño, costo, reproducibilidad e interpretabilidad para el problema?

## Modelos mínimos

1. **Baseline no neuronal:** TF-IDF + Logistic Regression.
2. **Modelo neuronal propio:** Embedding + mean pooling + MLP, o una pequeña LSTM.
3. **Transformer:** DistilBERT fine-tuned para sequence classification.

## Hipótesis sugeridas

- El Transformer superará al baseline en macro-F1, especialmente en frases contextuales.
- El baseline será mucho más barato y puede ser competitivo en ejemplos lexicalmente simples.
- El Transformer cometerá errores de alta confianza en ironía, ambigüedad o frases fuera de dominio.
- Dynamic padding reducirá tokens procesados frente a padding global.

## Organización de equipos

Equipos de 3–4 estudiantes:

- **Data/experiment lead:** splits, EDA, baseline y reproducibilidad.
- **Model lead:** arquitectura neuronal y Transformer.
- **Evaluation lead:** métricas, errores, visualizaciones y riesgos.
- **Delivery lead:** GitHub, README, app, pitch y model card.

En equipos de tres, combinar evaluación y entrega.

## Milestones

### M0 — Repositorio listo

- README inicial.
- Entorno documentado.
- Branch por integrante.
- Issue por milestone.
- Smoke test de imports.

### M1 — Problema y datos

- Definición de entrada/salida.
- Descripción de dataset, licencia y splits.
- Distribución de labels y longitudes.
- Riesgos de leakage y representatividad.

### M2 — Baseline

- Majority-class baseline.
- TF-IDF + Logistic Regression.
- Accuracy, macro-F1 y matriz de confusión.
- Tiempo aproximado de entrenamiento/inferencia.

### M3 — Modelo neuronal propio

- Tokenización/vocabulario o estrategia elegida.
- Embedding + pooling/MLP o LSTM.
- Curvas train/validation.
- Ablación sencilla: hidden size, dropout o LR.

### M4 — Transformer

- Justificación del checkpoint.
- Tokenizer y padding dinámico.
- Fine-tuning reproducible.
- Mejor checkpoint seleccionado con validation.

### M5 — Evaluación y error analysis

- Tabla comparativa de tres modelos.
- Confusion matrices.
- Diez errores de alta confianza.
- Taxonomía de errores y acción recomendada.
- Discusión de costo y latencia.

### M6 — Entrega

- App Gradio local o notebook de inferencia.
- README completo.
- Model card.
- Slides de demo.
- Pull request final revisado por otro equipo.

## Baseline TF-IDF + Logistic Regression

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

raw = load_dataset('rotten_tomatoes')

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=30_000,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(raw['train']['text'])
X_val = vectorizer.transform(raw['validation']['text'])
X_test = vectorizer.transform(raw['test']['text'])

y_train = raw['train']['label']
y_val = raw['validation']['label']
y_test = raw['test']['label']

baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)

val_pred = baseline.predict(X_val)
test_pred = baseline.predict(X_test)

print('Validation macro-F1:', f1_score(y_val, val_pred, average='macro'))
print('Test accuracy:', accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))
```

## Modelo neuronal propio: Embedding + mean pooling + MLP

Este modelo permite demostrar padding masks y pooling sin depender de una arquitectura preentrenada.

```python
from __future__ import annotations

from collections import Counter
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def basic_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+|[.,!?;]", text.lower())


counter = Counter()
for text in raw['train']['text']:
    counter.update(basic_tokenize(text))

itos = ['<pad>', '<unk>'] + [token for token, count in counter.items() if count >= 2]
stoi = {token: index for index, token in enumerate(itos)}
PAD_ID = stoi['<pad>']
UNK_ID = stoi['<unk>']


class TextDataset(Dataset):
    def __init__(self, split, max_length: int = 128) -> None:
        self.texts = split['text']
        self.labels = split['label']
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        tokens = basic_tokenize(self.texts[index])[: self.max_length]
        ids = [stoi.get(token, UNK_ID) for token in tokens]
        return torch.tensor(ids, dtype=torch.long), int(self.labels[index])


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    max_len = max(int(length.max()) if length.ndim else len(sequence) for length, sequence in zip(lengths, sequences))
    padded = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)

    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
        mask[row, : len(sequence)] = True

    return padded, mask, torch.tensor(labels, dtype=torch.long)


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        masked = embeddings * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)
```

Los equipos completan training loop, validación, checkpoint y evaluación reutilizando los patrones de sesiones 1 y 2.

## Tabla comparativa obligatoria

| Modelo | Parámetros entrenables | Tiempo de entrenamiento | Accuracy test | Macro-F1 test | Latencia aproximada | Fortalezas | Limitaciones |
|---|---:|---:|---:|---:|---:|---|---|
| Majority | — | — | | | | | |
| TF-IDF + LR | | | | | | | |
| Embedding + MLP/LSTM | | | | | | | |
| DistilBERT fine-tuned | | | | | | | |

No comparar tiempos obtenidos en hardware diferente sin declararlo.

## Rúbrica del proyecto — 100 puntos

| Dimensión | Puntos | Criterios |
|---|---:|---|
| Definición del problema y datos | 10 | Objetivo, labels, splits, licencia, EDA, leakage |
| Baseline y diseño experimental | 12 | Baseline válido, hipótesis y variables controladas |
| Modelo neuronal propio | 13 | Implementación correcta, training loop, curvas y ablation |
| Transformer y fine-tuning | 15 | Selección justificada, tokenización, configuración y checkpoint |
| Evaluación | 15 | Métricas adecuadas, comparación justa y test protegido |
| Análisis de errores | 10 | Taxonomía, ejemplos, causas probables y acciones |
| Reproducibilidad y GitHub | 10 | README, entorno, config, commits, PR y ejecución clara |
| Riesgos y model card | 7 | Sesgo, privacidad, licencia, usos y límites |
| Demo y comunicación | 8 | Narrativa, visuales, evidencia, respuestas técnicas |

### Penalizaciones sugeridas

- −10: usar test para seleccionar hiperparámetros.
- −10: no incluir baseline.
- −10: repositorio no ejecutable o sin instrucciones.
- −5: reportar solo accuracy en un caso desbalanceado.
- −5: subir secretos, datos sensibles o artifacts sin autorización.
- Hasta −15: resultados no reproducibles o inventados.

## Model card mínima

```markdown
# Model Card — <nombre>

## Model details
- Arquitectura/checkpoint:
- Tarea:
- Idioma:
- Versión/commit:

## Intended use
- Uso previsto:
- Usuarios previstos:
- Fuera de alcance:

## Training data
- Dataset y versión:
- Licencia:
- Splits:
- Preprocesamiento:

## Training procedure
- Seed:
- Hiperparámetros:
- Hardware:
- Criterio de selección:

## Evaluation
- Métricas:
- Resultados:
- Baselines:
- Subgrupos o slices:

## Limitations
- Errores conocidos:
- Fuera de distribución:
- Idioma/dominio:

## Risks and mitigations
- Sesgo:
- Privacidad:
- Uso indebido:
- Monitoreo:
```

## Presentación final — 7 minutos

1. **Problema y decisión** — 45 s.
2. **Datos y riesgos** — 45 s.
3. **Baseline** — 45 s.
4. **Modelos y experimento** — 90 s.
5. **Resultados comparativos** — 75 s.
6. **Errores y limitaciones** — 60 s.
7. **Demo y recomendación** — 60 s.

Preguntas del jurado: 3 minutos.

## Definición de terminado

El proyecto está terminado cuando otra persona puede:

1. clonar el repositorio;
2. crear el entorno;
3. ejecutar un smoke test;
4. reproducir al menos el baseline y la inferencia del mejor modelo;
5. encontrar las configuraciones y métricas;
6. comprender usos y limitaciones sin hablar con el equipo.

---

# APÉNDICES DOCENTES

## A. Principios para crear los slides

- Formato 16:9.
- Una idea central por slide.
- Máximo una fórmula principal por slide; los pasos de derivación pueden aparecer progresivamente.
- Mostrar shapes siempre que haya una transformación tensorial.
- Preferir diagramas y ejemplos concretos sobre párrafos.
- Usar código de 8–15 líneas por slide; enviar el código completo al notebook.
- Incluir una pregunta de predicción antes de ejecutar demos.
- Diferenciar visualmente datos, parámetros, activaciones, gradientes y métricas.
- Reservar slides de “errores comunes” y “qué evidencia necesito”.
- Cada arquitectura debe tener: motivación, diagrama, ecuaciones, shape flow, código, entrenamiento, errores y límites.

### Patrón visual recomendado para conceptos

```text
┌─────────────────────┬─────────────────────────────┐
│ Intuición / problema│ Diagrama de arquitectura    │
├─────────────────────┼─────────────────────────────┤
│ Fórmula             │ Shapes y ejemplo numérico  │
└─────────────────────┴─────────────────────────────┘
```

### Patrón visual recomendado para experimentos

```text
Hipótesis → Cambio controlado → Métrica → Evidencia visual → Decisión
```

## B. Banco de visualizaciones a generar

1. Fronteras lineales vs MLP sobre XOR/moons.
2. Sigmoid, tanh, ReLU y GELU.
3. Superficie de pérdida y ruta de SGD.
4. Grafo computacional con forward/backward.
5. Curvas train/validation para underfitting, good fit y overfitting.
6. Kernel recorriendo imagen y creando feature map.
7. Efecto de padding/stride.
8. Receptive field por profundidad.
9. Filtros de primera capa de una CNN.
10. Confusion matrix y galería de errores.
11. Celda RNN desplegada.
12. LSTM con compuertas.
13. Magnitud de gradientes por timestep.
14. Embeddings proyectados a 2D.
15. Matrices Q, K, V y attention scores.
16. Máscara causal triangular.
17. Múltiples attention heads.
18. Positional encoding sinusoidal.
19. Encoder vs decoder vs encoder–decoder.
20. Tokenización subword.
21. Padding global vs dinámico.
22. Comparación de métricas/costo entre modelos.
23. Taxonomía de errores.
24. Matriz de riesgos y mitigaciones.

## C. Mini-quizzes por sesión

### Sesión 1

- Una capa `(64 → 32)` recibe batch `(128,64)`. ¿Shape de salida y parámetros?
- ¿Qué activación usaría en una capa intermedia y por qué?
- ¿Qué recibe `CrossEntropyLoss`?
- ¿Qué hace `loss.backward()`?
- ¿Qué split se usa para selección de hiperparámetros?

### Sesión 2

- Output de Conv2d para `H=32,K=5,P=2,S=1`.
- Parámetros de `Conv2d(3,16,3)`.
- ¿Por qué augmentation debe preservar etiqueta?
- ¿Diferencia entre dropout train/eval?
- ¿Qué significa congelar el backbone?

### Sesión 3

- Shapes de `QKᵀ` para `(B,H,T,d_k)`.
- ¿Qué suma 1 en attention weights?
- ¿Por qué agregar posición?
- ¿Diferencia entre self-attention y cross-attention?
- ¿Por qué BERT puede ver ambos lados y GPT no durante generación causal?

### Sesión 4

- ¿Qué debe coincidir entre tokenizer y modelo?
- ¿Por qué dynamic padding ahorra cómputo?
- ¿Qué dataset se usa para ajustar y cuál para reportar resultado final?
- ¿Qué reduce LoRA y qué no garantiza?
- ¿Qué debe incluir una model card?

## D. Checklist de debugging

### Datos

- ¿El número de labels coincide con ejemplos?
- ¿Hay nulos o labels fuera de rango?
- ¿Las transformaciones preservan la etiqueta?
- ¿Se ajustó el scaler/vocabulario solo con train?
- ¿El batch tiene los shapes y dtypes esperados?

### Modelo

- ¿La última dimensión coincide con número de clases?
- ¿Los logits tienen shape correcto?
- ¿La loss corresponde a la tarea?
- ¿Todos los tensores están en el mismo device?
- ¿Los parámetros tienen `requires_grad=True` cuando corresponde?

### Entrenamiento

- ¿Se ejecuta `zero_grad`?
- ¿Los gradientes son finitos y no todos cero?
- ¿La loss disminuye en un microdataset de 20 muestras?
- ¿El modelo puede sobreajustar deliberadamente un batch pequeño?
- ¿`train()` y `eval()` se usan correctamente?

### Evaluación

- ¿Se desactivaron gradientes?
- ¿Se restauró el mejor checkpoint?
- ¿La métrica usa labels/predictions en el orden correcto?
- ¿Se reporta macro-F1 cuando las clases lo requieren?
- ¿Se inspeccionan errores y no solo promedios?

## E. Prueba de “overfit one batch”

Antes de una corrida larga, intentar que el modelo memorice un batch muy pequeño. Si no puede hacerlo, probablemente hay un error de datos, loss, arquitectura, optimizador o gradientes.

```python
small_inputs, small_labels = next(iter(train_loader))
small_inputs = small_inputs.to(device)
small_labels = small_labels.to(device)

model.train()
for step in range(300):
    optimizer.zero_grad(set_to_none=True)
    logits = model(small_inputs)
    loss = criterion(logits, small_labels)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(step, loss.item())
```

Adaptar la firma del modelo según la arquitectura. La pérdida debería bajar de manera marcada si el pipeline es correcto y la capacidad es suficiente.

## F. Checklist de reproducibilidad

- [ ] Seed registrada.
- [ ] Split determinístico o índices guardados.
- [ ] Versiones de Python/librerías registradas.
- [ ] Hardware declarado.
- [ ] Configuración fuera del notebook o claramente centralizada.
- [ ] Mejor checkpoint identificado.
- [ ] Métricas guardadas en JSON/CSV.
- [ ] Commit hash incluido en reporte.
- [ ] README con comandos exactos.
- [ ] Model card con limitaciones.
- [ ] Tokens y datos sensibles excluidos.

## G. Comandos Git mínimos

```bash
git clone <repository-url>
cd <repository>
git checkout -b feat/mlp-experiment

git status
git add notebooks/02_mlp_training.ipynb reports/figures/
git commit -m "feat: add reproducible mlp experiment"
git push -u origin feat/mlp-experiment
```

Convenciones sugeridas:

- `feat:` nueva capacidad.
- `fix:` corrección.
- `docs:` documentación.
- `test:` pruebas.
- `refactor:` reorganización sin cambiar comportamiento.
- `chore:` mantenimiento/configuración.

## H. Pull request template

```markdown
## Objetivo

## Cambios

## Cómo reproducir

## Evidencia
- Métricas:
- Figuras:

## Riesgos o limitaciones

## Checklist
- [ ] Ejecuté el notebook/script desde cero
- [ ] No incluí secretos ni datos sensibles
- [ ] Actualicé README/config
- [ ] Agregué o actualicé pruebas
```

## I. Pruebas mínimas

```python
# tests/test_shapes.py
import torch

from src.models import MoonMLP


def test_mlp_output_shape():
    model = MoonMLP(hidden_dim=16)
    x = torch.randn(8, 2)
    logits = model(x)
    assert logits.shape == (8, 1)
    assert torch.isfinite(logits).all()
```

```python
# tests/test_smoke.py

def test_imports():
    import torch
    import transformers

    assert torch.__version__
    assert transformers.__version__
```

## J. GitHub Actions mínimo

```yaml
name: smoke-test

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install --upgrade pip
      - run: pip install torch transformers pytest
      - run: pytest -q
```

No descargar datasets/modelos pesados en CI básica. Las pruebas deben ser rápidas y determinísticas.

## K. Prompts para discusión en clase

- “El modelo tiene 98% de accuracy, ¿qué información falta antes de aprobarlo?”
- “La validation loss sube, pero accuracy sigue estable. ¿Qué hipótesis investigarías?”
- “Un attention head mira una palabra específica. ¿Es una explicación?”
- “El Transformer supera 1.2 puntos de F1, pero cuesta 30 veces más. ¿Cuál elegirías?”
- “El modelo falla en un subgrupo. ¿Qué acciones técnicas y de gobernanza corresponden?”
- “¿Qué cambiaría si el dataset se actualiza cada mes?”

## L. Extensiones opcionales después del curso

1. Vision Transformer y comparación con CNN.
2. Fine-tuning de modelos multilingües para español.
3. LoRA sobre un modelo generativo pequeño.
4. Distillation y quantization-aware evaluation.
5. Retrieval-Augmented Generation.
6. Evaluación de LLMs, hallucination y factuality.
7. Diffusion models.
8. MLOps: tracking, registry, monitoring y drift.
9. Adversarial robustness y seguridad.
10. Distributed training con Accelerate/FSDP/DeepSpeed.

---

# FUENTES Y RECURSOS

## Documentación oficial

- PyTorch documentation: https://docs.pytorch.org/docs/stable/index.html
- PyTorch tutorials: https://docs.pytorch.org/tutorials/
- Autograd tutorial: https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
- PyTorch optimizers: https://docs.pytorch.org/docs/stable/optim.html
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- Transformers quickstart: https://huggingface.co/docs/transformers/en/quicktour
- Transformers fine-tuning: https://huggingface.co/docs/transformers/en/training
- Hugging Face Datasets: https://huggingface.co/docs/datasets/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/
- Hugging Face PEFT: https://huggingface.co/docs/peft/
- VS Code Jupyter notebooks: https://code.visualstudio.com/docs/datascience/jupyter-notebooks
- Python in VS Code: https://code.visualstudio.com/docs/languages/python
- Transformer Explainer: https://poloclub.github.io/transformer-explainer/
- Transformer Explainer source: https://github.com/poloclub/transformer-explainer

## Papers fundamentales

- Rumelhart, Hinton & Williams. *Learning representations by back-propagating errors* (1986).
- Hochreiter & Schmidhuber. *Long Short-Term Memory* (1997).
- Kingma & Ba. *Adam: A Method for Stochastic Optimization* (2014): https://arxiv.org/abs/1412.6980
- Ioffe & Szegedy. *Batch Normalization* (2015): https://arxiv.org/abs/1502.03167
- He et al. *Deep Residual Learning for Image Recognition* (2015): https://arxiv.org/abs/1512.03385
- Vaswani et al. *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
- Devlin et al. *BERT* (2018): https://arxiv.org/abs/1810.04805
- Sanh et al. *DistilBERT* (2019): https://arxiv.org/abs/1910.01108
- Cho et al. *Transformer Explainer* (2024): https://arxiv.org/abs/2408.04619

## Recursos interactivos opcionales

- TensorFlow Playground: https://playground.tensorflow.org/
- CNN Explainer: https://poloclub.github.io/cnn-explainer/
- Distill — visual articles: https://distill.pub/
- PyTorch TensorBoard tutorial: https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

## Nota de mantenimiento

Las APIs, checkpoints, licencias y datasets cambian. Antes de cada edición del curso:

1. ejecutar todos los notebooks en un entorno limpio;
2. congelar versiones validadas;
3. confirmar que los datasets/checkpoints siguen accesibles;
4. revisar model cards y licencias;
5. descargar artifacts necesarios para contingencia;
6. actualizar capturas de Transformer Explainer y Hugging Face;
7. verificar que los ejemplos no publiquen información sensible.

---

# RESUMEN EJECUTIVO PARA EL DOCENTE

- **Sesión 1:** que el estudiante comprenda el mecanismo de aprendizaje y pueda escribir el loop.
- **Sesión 2:** que aprenda a explotar estructura espacial y a experimentar rigurosamente.
- **Sesión 3:** que conecte secuencias, atención, shapes y arquitectura Transformer sin caja negra.
- **Sesión 4:** que adapte un modelo preentrenado, lo compare con baselines y entregue evidencia reproducible.
- **Proyecto:** no premiar únicamente la métrica más alta; evaluar calidad del experimento, análisis de errores, costo, reproducibilidad y riesgos.

La meta final no es que los estudiantes memoricen arquitecturas, sino que puedan responder con evidencia:

> **¿Qué representa el modelo, cómo aprende, por qué funciona, dónde falla y cómo se reproduce responsablemente?**

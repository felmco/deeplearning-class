# 🔗 Recursos del curso

Simuladores, documentación, papers, librerías y plataformas — todo lo que rodea al material
de las sesiones. Curado y verificado para esta edición del curso.

## 🕹️ Simuladores y visualizaciones interactivas

### Del propio curso (GitHub Pages)

| Simulador | Concepto |
|---|---|
| [MLP Playground](https://felmco.github.io/deeplearning-class/interactivos/mlp-playground.html) | red neuronal entrenándose en el navegador |
| [Descenso de gradiente](https://felmco.github.io/deeplearning-class/interactivos/descenso-gradiente.html) | learning rate, momentum, superficies de pérdida |
| [Activaciones](https://felmco.github.io/deeplearning-class/interactivos/activaciones.html) | sigmoid/tanh/ReLU/GELU y sus derivadas |
| [Convolución 2D](https://felmco.github.io/deeplearning-class/interactivos/convolucion.html) | kernels, stride, padding, feature maps |
| [Attention](https://felmco.github.io/deeplearning-class/interactivos/atencion.html) | QKᵀ, escala, máscara causal, softmax |
| [Softmax y temperatura](https://felmco.github.io/deeplearning-class/interactivos/softmax-temperatura.html) | sampling: temperatura, top-k, top-p |
| [Positional encoding](https://felmco.github.io/deeplearning-class/interactivos/positional-encoding.html) | las ondas que codifican posición |

### Externos (excelentes, úsalos)

| Recurso | Qué ofrece |
|---|---|
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | **GPT-2 small corriendo en tu navegador** — el laboratorio visual de la Sesión 3 ([código fuente](https://github.com/poloclub/transformer-explainer), [paper](https://arxiv.org/abs/2408.04619)) |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | una CNN completa, capa por capa, interactiva |
| [TensorFlow Playground](https://playground.tensorflow.org/) | el playground clásico de redes densas |
| [Distill.pub](https://distill.pub/) | artículos visuales de referencia (momentum, interpretabilidad…) |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | la explicación ilustrada más citada de la arquitectura |
| [Neural Network Playground de Karpathy — micrograd](https://github.com/karpathy/micrograd) | autograd en ~100 líneas, para leer completo |
| [Serie Neural Networks de 3Blue1Brown](https://www.3blue1brown.com/topics/neural-networks) | la mejor intuición visual de redes, gradiente y backprop (subtítulos en español); los capítulos 5-7 cubren GPT y attention |
| [Understanding LSTM Networks (Chris Olah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | las compuertas de la LSTM con diagramas, paso a paso (Sesión 3) |

## 📚 Documentación oficial

| Herramienta | Enlace |
|---|---|
| PyTorch | <https://docs.pytorch.org/docs/stable/index.html> |
| PyTorch tutorials | <https://docs.pytorch.org/tutorials/> |
| Autograd tutorial | <https://docs.pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html> |
| Optimizadores | <https://docs.pytorch.org/docs/stable/optim.html> |
| HF Transformers | <https://huggingface.co/docs/transformers/> |
| HF quickstart | <https://huggingface.co/docs/transformers/en/quicktour> |
| HF fine-tuning | <https://huggingface.co/docs/transformers/en/training> |
| HF Datasets | <https://huggingface.co/docs/datasets/> |
| HF Evaluate | <https://huggingface.co/docs/evaluate/> |
| HF PEFT | <https://huggingface.co/docs/peft/> |
| Gradio | <https://www.gradio.app/docs> |
| VS Code + Jupyter | <https://code.visualstudio.com/docs/datascience/jupyter-notebooks> |
| Remotion | <https://www.remotion.dev/docs> |

## 🧰 Librerías y plataformas

| Categoría | Herramientas |
|---|---|
| Framework principal | **PyTorch** (+ torchvision) |
| Modelos preentrenados | **Hugging Face Hub** (modelos, datasets, Spaces) |
| Experiment tracking | TensorBoard (incluido) · [Weights & Biases](https://wandb.ai/) · [MLflow](https://mlflow.org/) (opcionales) |
| Demos | **Gradio** · [Streamlit](https://streamlit.io/) |
| Cómputo de contingencia | [Google Colab](https://colab.research.google.com/) · [Kaggle Notebooks](https://www.kaggle.com/code) — el flujo principal del curso permanece en VS Code + GitHub |
| Calidad de código | **ruff** (lint) · **pytest** (tests) |
| Animaciones | **Remotion** (video programático) · matplotlib (GIFs) |

## 📄 Papers fundamentales

Lista comentada en [papers.md](papers.md). Los seis imprescindibles:

1. Rumelhart, Hinton & Williams (1986) — backpropagation.
2. Hochreiter & Schmidhuber (1997) — LSTM.
3. He et al. (2015) — ResNet / conexiones residuales.
4. Vaswani et al. (2017) — *Attention Is All You Need*.
5. Devlin et al. (2018) — BERT.
6. Sanh et al. (2019) — DistilBERT.

## 🌿 Flujo de trabajo

- [Git y GitHub del curso](git-flujo.md) — branches, commits, PRs y convenciones.
- [Checklist de debugging](debugging.md) — datos, modelo, entrenamiento y evaluación.
- [Puente PyTorch ↔ TensorFlow/Keras](pytorch-keras.md) — tabla de equivalencias y el
  MLP del Lab 1 lado a lado en ambos frameworks.

## 🔭 Después del curso (extensiones)

Vision Transformers · fine-tuning multilingüe para español · LoRA sobre un modelo generativo ·
distillation y quantization · Retrieval-Augmented Generation (RAG) · evaluación de LLMs ·
diffusion models · MLOps (tracking, registry, monitoring, drift) · adversarial robustness ·
entrenamiento distribuido (Accelerate/FSDP/DeepSpeed).

## 🛠️ Nota de mantenimiento

Las APIs, checkpoints, licencias y datasets cambian. Antes de cada edición: ejecutar todos los
notebooks en un entorno limpio, congelar versiones validadas, confirmar que datasets y
checkpoints siguen accesibles, revisar model cards y licencias, y verificar que los ejemplos
no publiquen información sensible.

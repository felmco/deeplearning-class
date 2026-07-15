# 📄 Papers fundamentales, comentados

Cada paper con su idea central en una frase y por qué importa para este curso.

## Los cimientos

### Learning representations by back-propagating errors
**Rumelhart, Hinton & Williams (1986)** · *Nature*
La regla de la cadena aplicada sistemáticamente permite entrenar redes multicapa. Todo lo
demás — desde tu MLP del Lab 02 hasta GPT — corre sobre esta idea. *(Sesión 1)*

### Long Short-Term Memory
**Hochreiter & Schmidhuber (1997)** · *Neural Computation*
Compuertas + una celda de memoria con camino aditivo resuelven (en gran parte) el vanishing
gradient de las RNN. Prefigura la idea de "autopista del gradiente". *(Sesión 3)*

## Optimización y entrenamiento

### Adam: A Method for Stochastic Optimization
**Kingma & Ba (2014)** · <https://arxiv.org/abs/1412.6980>
Lleva un promedio móvil del gradiente (hacia dónde ir) y de su magnitud (cuánto confiar),
por parámetro. Con la corrección de AdamW
(weight decay desacoplado), es el optimizador por defecto del curso. *(Sesión 2)*

### Batch Normalization
**Ioffe & Szegedy (2015)** · <https://arxiv.org/abs/1502.03167>
Normalizar activaciones por mini-batch estabiliza y acelera el entrenamiento. Ojo con la
diferencia train/eval — fuente clásica de bugs. *(Sesión 2)*

### Deep Residual Learning for Image Recognition
**He et al. (2015)** · <https://arxiv.org/abs/1512.03385>
$y = F(x) + x$: aprender el residuo en lugar de la transformación completa permite apilar
cientos de capas. La misma conexión residual vive en cada bloque Transformer. *(Sesiones 2 y 3)*

## La era de la atención

### Attention Is All You Need
**Vaswani et al. (2017)** · <https://arxiv.org/abs/1706.03762>
Elimina la recurrencia: solo attention + FFN + residuals + LayerNorm + positional encoding.
El paper que define la arquitectura de la década. Léelo después de la Sesión 3 — entenderás
cada ecuación. *(Sesión 3)*

### BERT: Pre-training of Deep Bidirectional Transformers
**Devlin et al. (2018)** · <https://arxiv.org/abs/1810.04805>
Pretraining bidireccional (masked language modeling) + fine-tuning barato por tarea.
Establece el paradigma que usas en el Lab 06. *(Sesión 4)*

### DistilBERT, a distilled version of BERT
**Sanh et al. (2019)** · <https://arxiv.org/abs/1910.01108>
Knowledge distillation: 40% más pequeño, 60% más rápido, ~97% del desempeño. El modelo del
laboratorio y el proyecto final. *(Sesión 4)*

## Herramientas y responsabilidad

### LoRA: Low-Rank Adaptation of Large Language Models
**Hu et al. (2021)** · <https://arxiv.org/abs/2106.09685>
$\Delta W \approx BA$ con rango bajo: fine-tuning entrenando <1% de los parámetros. La base
de PEFT. *(Sesión 4)*

### Model Cards for Model Reporting
**Mitchell et al. (2019)** · <https://arxiv.org/abs/1810.03993>
El estándar de documentación responsable de modelos: usos previstos, métricas por subgrupo,
límites. La plantilla del proyecto final deriva de aquí. *(Sesión 4 y proyecto)*

### Transformer Explainer: Interactive Learning of Text-Generative Models
**Cho et al. (2024)** · <https://arxiv.org/abs/2408.04619>
La herramienta del laboratorio visual de la Sesión 3: GPT-2 small ejecutándose en el navegador
con cada matriz inspeccionable.

---

**Cómo leer un paper de ML** (sugerencia del curso): primero abstract + figuras + tablas;
luego la introducción y la conclusión; las ecuaciones al final, con lápiz, anotando shapes.
Dos pasadas valen más que una lectura "completa" pasiva.

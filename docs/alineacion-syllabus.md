# 📋 Alineación con el syllabus oficial (R-FAC-002)

**Asignatura:** Electiva I — Deep Learning · **Programa:** Maestría en Ciencia de Datos,
Universidad del Sinú (seccional Cartagena) · **Intensidad:** 36 h presenciales
(12 teóricas + 24 de laboratorio) + 108 independientes = 144 h, en 6 semanas ·
**Syllabus fuente:** `docs/R-FAC-003 SYLLABUS ELECTIVA I (Deep Learning).pdf`

Este documento mapea cada **Resultado de Aprendizaje (RA)** y **criterio de
evaluación** del syllabus al material del repositorio y a la evidencia que produce el
estudiante.

## RA1 — Identificar los fundamentos formales de las herramientas y arquitecturas de ML y, en particular, del DL

| Criterio del syllabus | Dónde vive en el repo | Evidencia del estudiante |
|---|---|---|
| Identifica los modelos derivados de los datos y el diseño de algoritmos de ML y DL | [Sesión 0](../sesiones/00-fundamentos-ml.md) (mapa del ML, galería de modelos clásicos) + [Sesión 1](../sesiones/01-fundamentos.md) | Lab 00: tabla comparativa de 5 modelos + baseline |
| Analiza los fundamentos de aprendizaje automático | Sesión 0 §1–§3 y §6 (flujo de un proyecto, splits, baseline) + Sesión 1 §7 (generalización) | Exit tickets 0 y 1 |
| Evalúa los métodos de aprendizaje supervisado, no supervisado y reforzado | Sesión 0 §2–§5 (los tres paradigmas, k-means/PCA, Q-learning) | Lab 00 completo (los tres paradigmas con código) |

## RA2 — Implementar algoritmos en lenguajes de alto nivel para resolver problemas reales

| Criterio del syllabus | Dónde vive en el repo | Evidencia del estudiante |
|---|---|---|
| Analiza los conceptos de redes neuronales artificiales y redes profundas | [Sesión 1](../sesiones/01-fundamentos.md) (neurona, MLP, activaciones, backprop) | Lab 02 (MLP sobre make_moons) |
| Aplica los métodos de optimización de modelos | Sesión 1 §5–§6 (gradiente, autograd, training loop) + [Sesión 2 §6](../sesiones/02-cnn-vision.md) (SGD/AdamW, schedules) | Labs 02–03: experimentos controlados |
| Analiza la función de pérdida, los hiperparámetros y las estrategias de aprendizaje | Sesión 1 §4 (losses como curvas de castigo) + Sesiones 1–2 (LR, regularización, early stopping) | Experimentos de una variable + curvas |
| Implementa modelos en TensorFlow | El curso implementa en **PyTorch** (equivalente pedagógico — justificación en el [programa](../Programa_Completo_Deep_Learning_Transformers_32h.md)); el [puente PyTorch↔Keras](../recursos/pytorch-keras.md) da la traducción completa y el MLP del Lab 1 en ambos frameworks | Lectura del puente + (opcional) reproducir el MLP en Keras |

## RA3 — Implementar soluciones DL considerando aspectos éticos, legales, económicos y sociales

| Criterio del syllabus | Dónde vive en el repo | Evidencia del estudiante |
|---|---|---|
| Implementa modelos de analítica de imágenes con redes neuronales convolucionales | [Sesión 2](../sesiones/02-cnn-vision.md) completa | Lab 03 (CNN sobre FashionMNIST) |
| Implementa modelos de analítica de series temporales con redes neuronales recurrentes | [Sesión 3 §3](../sesiones/03-secuencias-transformers.md) ("No solo texto: analítica de series temporales") | Notebook 04 §7 (forecasting con LSTM y split temporal) |
| Identifica y aplica las técnicas de paralelización y computación en procesadores | [Sesión 2 §5](../sesiones/02-cnn-vision.md) ("El cómputo: CPU, GPU y paralelización") + Sesión 4 §6 (mixed precision, eficiencia) | Uso de device/batch/workers en los labs |
| Aspectos éticos, legales, económicos y sociales | [Sesión 4 §7](../sesiones/04-hugging-face-proyecto.md) (riesgos, ética, implicaciones económico-sociales) + [model card](../proyecto/model-card-template.md) | [Ensayo](../proyecto/ensayo.md) + model card del proyecto |

*Además del alcance del syllabus, el curso cubre atención/Transformers (Sesión 3) y
fine-tuning con Hugging Face (Sesión 4), alineados con la competencia general de la
asignatura ("implementar soluciones basadas en Deep Learning desde una perspectiva
amplia").*

## Mapeo del sistema de evaluación

| Mecanismo del syllabus | Peso | Equivalente en el curso |
|---|---:|---|
| Taller teórico | 10% | Quizzes y exit tickets de las Sesiones 0–4 (con respuestas auto-verificables) |
| Ensayo | 15% | [Ensayo: automatizar una decisión](../proyecto/ensayo.md) (rúbrica propia de 100 pts) |
| Avance 1 del proyecto de aula + exposición | 25% + 10% | Milestones [M0–M3](../proyecto/README.md) (repo, datos, baseline, modelo propio) + presentación de avance |
| Avance 2 del proyecto de aula + exposición | 15% + 15% | Milestones M4–M6 (Transformer, evaluación, entrega) + demo day |

> ⚠️ Los porcentajes del syllabus fuente suman **90%** (10+15+25+10+15+15): confirmar
> con la coordinación del programa la asignación del 10% restante antes de aplicar la
> ponderación en una cohorte.

## Mapeo de las 6 semanas del syllabus

| Semana | Eje del syllabus | Material del repo |
|---|---|---|
| 1 | Fundamentos de ML; supervisado, no supervisado y reforzado | **Sesión 0** + Lab 00 |
| 2 | Redes neuronales, optimización, loss, hiperparámetros | **Sesión 1** + Labs 01–02 |
| 3 | Implementación y entrenamiento robusto | **Sesión 2** (1ª parte) + Lab 03 |
| 4 | Analítica de imágenes con CNN; paralelización y cómputo | **Sesión 2** (2ª parte: transfer learning, cómputo) |
| 5 | Series temporales y secuencias con RNN; atención | **Sesión 3** + Labs 04–05 |
| 6 | Proyecto integrador, ética y entrega | **Sesión 4** + proyecto final + ensayo + demo day |

## Notas

- **Bibliografía del syllabus** (Torres; Pajares & Herrera; Pineda — centradas en
  Keras/TensorFlow): compatible vía el [puente PyTorch↔Keras](../recursos/pytorch-keras.md);
  los conceptos (tensores, backprop, CNN, RNN) son idénticos en ambos frameworks.
- El repositorio es **el aula**: cada semana enlaza páginas navegables, simuladores,
  notebooks y código de referencia (ver el [README](../README.md)).

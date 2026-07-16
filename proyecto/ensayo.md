# ✍️ Ensayo — Automatizar una decisión: implicaciones éticas, económicas y sociales

> Evidencia escrita del curso (peso sugerido: **15%** de la evaluación, ajustable a la
> institución donde se dicte). Se entrega antes del inicio del proyecto final y se
> retoma en la model card.

## La consigna

Elige **una decisión concreta que hoy toma una persona** y que podría automatizarse
con un modelo como los de este curso (priorizar reclamos, filtrar hojas de vida,
aprobar créditos pequeños, moderar comentarios, priorizar triage…). Argumenta, con la
evidencia técnica que el curso te dio, **si la automatizarías, cómo y con qué
salvaguardas** — o por qué no.

## Estructura sugerida (1500–2500 palabras)

1. **La decisión y su contexto** — quién la toma hoy, con qué información, a qué costo
   y con qué tasa de error humana (estimada).
2. **La solución técnica** — qué modelo usarías (baseline, red propia o Transformer
   fine-tuned), qué datos requiere y qué métricas la validarían. Sé específico: este
   curso te dio el vocabulario (macro-F1, matriz de confusión, errores de alta
   confianza, model card).
3. **Análisis ético y legal** — sesgo y subgrupos afectados, privacidad de los datos,
   licencias, explicabilidad exigible, vía de apelación humana.
4. **Análisis económico y social** — qué tareas/empleos transforma, quién captura el
   ahorro, quién asume el costo de los errores, qué pasa a escala (100 vs 100 000
   decisiones/día).
5. **Veredicto y salvaguardas** — tu recomendación: automatizar todo, asistir a la
   persona (human-in-the-loop), o no automatizar. Con condiciones verificables.

## Rúbrica — 100 puntos (pondera al 15% del curso)

| Dimensión | Puntos | Qué se evalúa |
|---|---:|---|
| Precisión técnica | 30 | el modelo, los datos y las métricas propuestas son coherentes con lo visto en el curso |
| Análisis ético/legal | 25 | identifica sesgos, afectados y obligaciones concretas — no generalidades |
| Análisis económico/social | 25 | costos, beneficios y su distribución; efectos a escala |
| Argumentación y evidencia | 20 | postura clara, contraargumentos considerados, fuentes citadas |

**Penalizaciones:** afirmaciones técnicas incorrectas (−5 c/u), no considerar a los
afectados por los errores del modelo (−10), ausencia de veredicto (−10).

## Referencias de partida

- [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993) — el estándar de documentación responsable.
- La [Sesión 4 §7](../sesiones/04-hugging-face-proyecto.md) — riesgos, ética y documentación.
- La taxonomía de errores del curso (Sesión 4 §5): tu análisis de "quién paga los errores" puede apoyarse en ella.

---

| [🏠 Inicio](../README.md) | [Proyecto final](README.md) | [Model card](model-card-template.md) |
|---|---|---|

# Propuesta de proyecto propio — <nombre del equipo>

> Entregar al docente **antes del Checkpoint 1**. Sin aprobación explícita, el equipo
> sigue la ruta guiada. Una vez aprobada, commitear este archivo como `propuesta.md`
> en la raíz del repositorio del equipo (es un entregable del Checkpoint 1).

## 1. Problema

Qué se clasifica, para quién y por qué importa (3–5 líneas).

## 2. Dataset

- **Fuente:** URL del Hugging Face Hub, u origen institucional.
- **Licencia:** ¿permite uso académico?
- **Tamaño e idioma:** filas por split (o total), idioma(s).
- **Si es institucional:** autorización de uso por escrito y control de privacidad
  (anonimización). Sin autorización no se usa.

## 3. Labels y tarea

Clases y su distribución esperada. ¿Está balanceado? Métrica principal
(**macro-F1 por defecto**; justificar si se propone otra).

## 4. Splits

¿Train/validation/test vienen dados o los crea el equipo? Si los crea el equipo:
¿cómo se hace la partición y cómo se evita el leakage (p. ej. el mismo autor o
documento repartido entre splits)?

## 5. Riesgos

Sesgo, privacidad, representatividad, tamaño insuficiente, costo de cómputo.

## 6. Plan de los tres modelos (contrato mínimo del curso)

- **Baseline no neuronal:** …
- **Modelo neuronal propio** (`src/models.py` + `src/train.py`): …
- **Transformer fine-tuned** (checkpoint y por qué): …

---

**Aprobación del docente:** ☐ Aprobada · ☐ Aprobada con cambios: ______ · **Fecha:** ______

---

| [🏠 Inicio](../README.md) | [Brief del proyecto](README.md) |
|---|---|

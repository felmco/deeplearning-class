# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este repositorio

Curso completo de Deep Learning (32 h, en español) que **es el aula**: las páginas markdown de `sesiones/` son el material del estudiante, `docs/` se sirve por GitHub Pages (https://felmco.github.io/deeplearning-class/), y los notebooks/código son los laboratorios. No hay slides. Todo el contenido va en **español conservando los términos técnicos en inglés** (batch, loss, attention…).

## Comandos

```bash
# Lint (lo que corre CI)
python3 -m ruff check src/ tests/

# Tests (CI los corre con torch instalado; local sin torch fallará en la colección)
pytest tests/ -q -k "not transformers"     # smoke tests del CI
pytest tests/test_shapes.py -q             # un solo archivo

# Regenerar TODAS las figuras estáticas (escriben en docs/assets/figuras/)
python3 -m src.figuras.generar_estaticas
# Regenerar los GIFs
python3 -m src.figuras.generar_gifs

# Renderizar los videos Remotion (desde remotion/)
npm run render:todo    # o render:forward-backward / render:atencion
```

CI (`.github/workflows/smoke-test.yml`): ruff + pytest sin datasets ni modelos pesados. El `conftest.py` de la raíz existe para que `import src` funcione en pytest — no borrarlo.

## Arquitectura del contenido

La cadena pedagógica es: `sesiones/*.md` (teoría con figuras/simuladores/videos embebidos) → `notebooks/*.ipynb` (labs que referencian las sesiones) → `src/` (implementación de referencia comentada línea a línea) → `proyecto/` (proyecto final con rúbrica). `Programa_Completo_...md` es el documento del docente.

**Regla editorial del curso:** cada concepto se presenta en 4 niveles — intuición → visual → fórmula → código. Cada sigla se expande en su primera aparición; cada símbolo matemático se glosa cuando aparece (el público NO tiene base de cálculo/álgebra/estadística); toda fórmula lleva una lectura en palabras cerca. Al editar contenido, mantener ese contrato.

### Figuras (reproducibles)

Todas las figuras de `docs/assets/figuras/` se generan con `src/figuras/generar_estaticas.py` (una función `fig_*` por figura, registrada en la lista `tareas` de `main()`) y `generar_gifs.py`. Paleta y estilo compartidos en `src/figuras/estilo.py` (Okabe-Ito, accesible para daltonismo). Para una figura nueva: escribir `fig_x()`, registrarla, renderizar, **inspeccionar visualmente el PNG** (los solapes de texto son el defecto más común) e insertar en la sesión con ruta relativa `../docs/assets/figuras/x.png`.

Regenerar todas las figuras toca los PNG existentes solo en metadata: antes de commitear, revertir los PNG cuyo contenido no cambió (`git checkout -- <los no tocados>`) para mantener diffs limpios.

### GitHub Pages y simuladores

`docs/` es la raíz de Pages (branch `main`, carpeta `/docs`). `docs/interactivos/*.html` son 7 simuladores autocontenidos (HTML+JS inline, sin dependencias); `docs/videos/` sirve los mp4 renderizados de `remotion/`. Los enlaces externos en HTML propio llevan `target="_blank" rel="noopener"`.

## Trampas de renderizado de GitHub (todas ya cazadas aquí — no reintroducirlas)

1. **Macros prohibidos en math**: GitHub rechaza `\operatorname` → usar `\mathrm{}`.
2. **Markdown se come escapes ANTES de pasar el math a MathJax**: `\#` llega como `#` (error) y `\{ \}` llegan como llaves peladas. No usar `#` en math (usar `N_\theta` para "número de"); usar `\lbrace \rbrace` en vez de `\{ \}`.
3. **Math inline pegado a paréntesis no siempre renderiza**: `($\max(z)$)` sale literal. Preferir Unicode directo (σ, ∇, eᶻ), código inline o reformular.
4. **GitHub sanitiza HTML en markdown**: `<video>` y el atributo `target` se eliminan. Videos se embeben como thumbnail clickable: `[![alt](poster.png)](url.mp4)`.
5. Los notebooks renderizan su markdown con las mismas restricciones (los macros van escapados `\\operatorname` en el JSON).

## Editar notebooks

Editar el JSON con Python (json.loads/dumps, `ensure_ascii=False, indent=1`), nunca con reemplazos de texto ciegos: un replace amplio puede caer dentro de una celda de código y romperla. Validar `json.loads` tras editar. Regla de contenido: los glosarios van en celdas markdown o comentarios — jamás tocar código ejecutable para insertar texto.

## Convenciones

- Commits en español, estilo `feat:`/`fix:`/`mejora:`/`docs:`/`chore:`.
- El test set es sagrado en TODO el material: cualquier ejemplo o plantilla que compare variantes lo hace contra validation; test se usa una sola vez. No introducir código de ejemplo que viole esto.
- `remotion/` usa React 18 + Remotion 4; los labels de las animaciones tienen posición/offset ajustables por arista (ver `ForwardBackward.tsx`) porque los solapes de texto fueron un bug real.

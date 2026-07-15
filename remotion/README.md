# 🎬 Animaciones Remotion del curso

Composiciones de video programáticas (React + [Remotion](https://www.remotion.dev/))
que animan dos conceptos centrales del curso:

| Composición | Concepto | Sesión |
|---|---|---|
| `ForwardBackward` | El grafo computacional: valores fluyendo hacia adelante y gradientes hacia atrás sobre $L=(wx+b-y)^2$ | 1 |
| `FlujoAtencion` | El pipeline de attention: $QK^\top$ → escala → máscara causal → softmax → $\times V$ | 3 |

## Uso

```bash
cd remotion
npm install

# Estudio interactivo (previsualizar y editar en el navegador)
npm run studio

# Renderizar los MP4
npm run render:todo
```

Los videos se generan en `remotion/out/` (carpeta ignorada por git). Si quieres
embeberlos en las sesiones, muévelos a `docs/assets/videos/` y enlázalos desde
el Markdown.

## Por qué Remotion

Cada fotograma es una función de React del número de frame: la animación es
**código versionable**, no un archivo binario opaco. Cambiar los números del
ejemplo (por ejemplo, `w=3, b=1`) regenera el video completo — el mismo
principio de reproducibilidad que rige todo el curso.

© 2026 Future Tales, LLC · MIT

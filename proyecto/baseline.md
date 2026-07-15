# Baseline del proyecto: TF-IDF + Logistic Regression

**Regla del curso: sin baseline no hay evidencia de valor.** Este es el código de partida
del milestone M2 — completo, ejecutable y deliberadamente simple.

```python
"""Baseline no neuronal del proyecto final.

Qué es: TF-IDF (bolsa de n-gramas ponderada) + regresión logística.
Qué hace: establece el piso de desempeño que cualquier modelo neuronal
debe superar CON CLARIDAD para justificar su costo adicional.
"""

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

raw = load_dataset('rotten_tomatoes')

# ── Representación: TF-IDF de unigramas y bigramas ──────────────────
# sublinear_tf suaviza frecuencias; min_df=2 descarta rarezas;
# el vectorizer se ajusta SOLO con train (regla anti-leakage).
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

# ── Baseline 0: clase mayoritaria (el piso absoluto) ────────────────
from collections import Counter
mayoritaria = Counter(y_train).most_common(1)[0][0]
acc_majority = accuracy_score(y_test, [mayoritaria] * len(y_test))
print(f'Majority-class accuracy: {acc_majority:.3f}')

# ── Baseline 1: regresión logística ─────────────────────────────────
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)

val_pred = baseline.predict(X_val)
test_pred = baseline.predict(X_test)

print('Validation macro-F1:', f1_score(y_val, val_pred, average='macro'))
print('Test accuracy      :', accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=3))
```

## Qué registrar para la tabla comparativa

- Macro-F1 y accuracy en test.
- Tiempo de entrenamiento (segundos: sí, segundos).
- Tiempo de inferencia por 1000 ejemplos.
- Los 10 errores de mayor confianza (`predict_proba`) — te sorprenderá
  cuántos comparte con el Transformer.

## Preguntas para el reporte

1. ¿Qué bigramas reciben los pesos más altos? (`vectorizer.get_feature_names_out()` +
   `baseline.coef_`) ¿Tienen sentido lingüístico?
2. ¿En qué tipo de frases falla el baseline que el Transformer resuelve? ¿Y al revés?
3. Si el Transformer gana por ~2 puntos de F1 pero cuesta 100× más en cómputo,
   ¿qué recomendarías a la organización? Justifica.

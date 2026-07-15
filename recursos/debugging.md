# 🩺 Checklist de debugging de modelos

El 90% de los bugs de Deep Learning cae en cuatro categorías. Recorre las listas EN ORDEN:
los bugs de datos son los más comunes y los más baratos de encontrar.

## 1. Datos

- [ ] ¿El número de labels coincide con el número de ejemplos?
- [ ] ¿Hay nulos o labels fuera de rango?
- [ ] ¿Las transformaciones (augmentation) preservan la etiqueta?
- [ ] ¿Se ajustó el scaler/vocabulario **solo con train**?
- [ ] ¿El batch tiene los shapes y dtypes esperados? (imprímelos: no lo asumas)

## 2. Modelo

- [ ] ¿La última dimensión coincide con el número de clases?
- [ ] ¿Los logits tienen el shape correcto?
- [ ] ¿La loss corresponde a la tarea? (logits + `CrossEntropyLoss`, no softmax antes)
- [ ] ¿Todos los tensores están en el mismo device?
- [ ] ¿Los parámetros tienen `requires_grad=True` cuando corresponde?

## 3. Entrenamiento

- [ ] ¿Se ejecuta `optimizer.zero_grad()` en cada iteración?
- [ ] ¿Los gradientes son finitos y no todos cero? (`p.grad.norm()`)
- [ ] ¿La loss disminuye en un microdataset de 20 muestras?
- [ ] ¿El modelo puede **sobreajustar deliberadamente un batch pequeño**? (ver abajo)
- [ ] ¿`model.train()` y `model.eval()` se usan correctamente?

## 4. Evaluación

- [ ] ¿Se desactivaron los gradientes (`torch.inference_mode()`)?
- [ ] ¿Se restauró el mejor checkpoint (no el último)?
- [ ] ¿La métrica recibe labels/predictions en el orden correcto?
- [ ] ¿Se reporta macro-F1 cuando las clases lo requieren?
- [ ] ¿Se inspeccionan errores individuales y no solo promedios?

## La prueba de "overfit one batch"

Antes de cualquier corrida larga: intenta que el modelo **memorice** un batch diminuto.
Si no puede, hay un bug en datos, loss, arquitectura, optimizador o gradientes — y acabas
de ahorrarte horas de GPU.

```python
# Tomar UN batch pequeño y entrenar solo sobre él
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

# Esperado: la loss se DESPLOMA hacia ~0.
# Si se queda plana: revisa el checklist de arriba, en orden.
```

## Síntoma → hipótesis

| Síntoma | Primeras hipótesis |
|---|---|
| Loss constante desde el inicio | LR=0, gradientes no llegan, labels constantes |
| Loss = NaN | LR demasiado alto, división por cero, log(0) |
| Train baja, val nunca baja | leakage inverso, splits mal hechos, bug de eval |
| Accuracy perfecta sospechosa | leakage: el label se filtró a las features |
| Métricas erráticas entre eval | falta `model.eval()`, BatchNorm en modo train |
| GPU out of memory | batch muy grande, grafo retenido (falta `inference_mode`) |

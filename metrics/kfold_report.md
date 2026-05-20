# Diagnóstico del modelo de voz — VoiceCNN

## Configuración inicial

- **Dispositivo utilizado**: MPS (Metal Performance Shaders - Apple Silicon)
- **Modelo cargado**: Sí
- **Precisión CV promedio**: 100.00%
- **Parámetros del modelo**: 2,207,142
- **Conjunto de prueba fijo (test)**: 6420 muestras
- **Train + Validación (K-Fold)**: 25692 muestras
- **Dataset total**: 32112 muestras
- **Clases**: DETENER, ADELANTE, IZQUIERDA, DERECHA, GIRO_IZQ, GIRO_DER
- **Épocas por fold**: 40
- **Split**: 80% Train+Val / 20% Test

---

## Evaluación en conjunto de prueba fijo (20% — nunca visto)

### Resultados generales
- **Precisión (accuracy)**: 100.00%

### Matriz de confusión

| Verdadero \ Predicho | DETENER | ADELANTE | IZQUIERD | DERECHA | GIRO_IZQ | GIRO_DER |
|---|---|---|---|---|---|---|
| **DETENER** | 1070 | 0 | 0 | 0 | 0 | 0 |
| **ADELANTE** | 0 | 1070 | 0 | 0 | 0 | 0 |
| **IZQUIERDA** | 0 | 0 | 1070 | 0 | 0 | 0 |
| **DERECHA** | 0 | 0 | 0 | 1070 | 0 | 0 |
| **GIRO_IZQ** | 0 | 0 | 0 | 0 | 1070 | 0 |
| **GIRO_DER** | 0 | 0 | 0 | 0 | 0 | 1070 |

### Métricas por clase

| Clase | Aciertos / Total | Recall | Precision | F1 |
|-------|-----------------|--------|-----------|-----|
| DETENER | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| ADELANTE | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| IZQUIERDA | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| DERECHA | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| GIRO_IZQ | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| GIRO_DER | 1070 / 1070 | **100.0%** ✅ | 100.0% | 100.0% |
| **MACRO** | — | **100.0%** | **100.0%** | **100.0%** |

> **Nota**: Recall = capacidad de detectar la clase. Precision = cuando predice esa clase, ¿cuántas veces acierta?

### Ejemplo de predicciones (primeras 4 muestras del test)

| Muestra | Verdadero | Predicho | Logits (crudos) |
|---------|-----------|----------|----------------|
| 0 | DETENER | DETENER | [+17.28, -8.69, -8.72, -7.99, -12.57, -11.45] |
| 1 | DETENER | DETENER | [+21.96, -10.88, -10.72, -9.73, -15.53, -15.11] |
| 2 | DETENER | DETENER | [+30.87, -16.98, -13.94, -12.52, -19.96, -19.83] |
| 3 | DETENER | DETENER | [+23.38, -14.89, -4.36, -13.18, -14.67, -15.25] |

> Los logits son las activaciones crudas para cada clase (orden: DETENER, ADELANTE, IZQUIERDA, DERECHA, GIRO_IZQ, GIRO_DER). Valores más altos indican mayor confianza.

---

## 5-Fold Cross-Validation (sobre el 80% de datos)

### Resultados por fold

| Fold | Train | Val | Val Acc |
|------|-------|-----|---------|
| 1 | 20553 | 5139 | **100.0%** |
| 2 | 20553 | 5139 | **100.0%** |
| 3 | 20554 | 5138 | **100.0%** |
| 4 | 20554 | 5138 | **100.0%** |
| 5 | 20554 | 5138 | **100.0%** |
| **Promedio** | — | — | **100.0% ± 0.00%** |

### Métricas agregadas CV (todos los folds combinados)

| Clase | Aciertos / Total | Recall | Precision | F1 |
|-------|-----------------|--------|-----------|-----|
| DETENER | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| ADELANTE | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| IZQUIERDA | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| DERECHA | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| GIRO_IZQ | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| GIRO_DER | 4282 / 4282 | **100.0%** ✅ | 100.0% | 100.0% |
| **MACRO** | — | **100.0%** | **100.0%** | **100.0%** |

### Matriz de confusión CV (acumulada)

| Verdadero \ Predicho | DETENER | ADELANTE | IZQUIERD | DERECHA | GIRO_IZQ | GIRO_DER |
|---|---|---|---|---|---|---|
| **DETENER** | 4282 | 0 | 0 | 0 | 0 | 0 |
| **ADELANTE** | 0 | 4282 | 0 | 0 | 0 | 0 |
| **IZQUIERDA** | 0 | 0 | 4282 | 0 | 0 | 0 |
| **DERECHA** | 0 | 0 | 0 | 4282 | 0 | 0 |
| **GIRO_IZQ** | 0 | 0 | 0 | 0 | 4282 | 0 |
| **GIRO_DER** | 0 | 0 | 0 | 0 | 0 | 4282 |

---

## Resumen de problemas detectados

_Sin problemas detectados — todas las clases ≥ 80% recall._

---

## Conclusión

| Métrica | Valor |
|---------|-------|
| CV Accuracy (media 5 folds) | **100.00%** |
| CV Accuracy (desv. estándar) | **±0.00%** |
| Test Accuracy (holdout 20%) | **100.00%** |
| Macro Recall (test) | **100.00%** |
| Macro Precision (test) | **100.00%** |
| Macro F1 (test) | **100.00%** |

_Modelo guardado en `models/voice_model.pth` — listo para usar con `main_voice.py`._

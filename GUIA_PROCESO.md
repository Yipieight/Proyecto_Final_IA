# GUÍA DE DATASET Y ENTRENAMIENTO — ROBOT SEGUIDOR DE LÍNEA CON VISIÓN

> Manual paso a paso para el equipo durante la semana de trabajo.  
> **No ejecutar código sin leer primero la sección correspondiente.**

---

## BLOQUE A — ANTES DE CAPTURAR IMÁGENES

### A.1 — Lista de verificación física de la cámara

| Parámetro | Valor recomendado |
|---|---|
| Altura de la cámara sobre el chasis | **15–20 cm** desde el suelo |
| Ángulo de inclinación hacia el suelo | **30–45°** respecto al eje horizontal |
| Campo visual en el frame | La línea debe ocupar el **tercio inferior central** del encuadre |

**Cómo verificar el encuadre antes de empezar:**
1. Coloca el robot sobre la línea en un tramo recto.
2. Activa el stream (IP Webcam → ver en navegador o Python).
3. En el frame en vivo, dibuja mentalmente una cruz en el centro.
4. La línea debe entrar por el borde inferior y atravesar el centro-inferior.  
   - Si la línea aparece en el tercio superior → bajar la cámara o aumentar inclinación.
   - Si la línea desaparece en curvas cerradas → subir la cámara ligeramente.

**Condiciones de iluminación — qué evitar:**
- **Sombras duras** del propio robot o de objetos cercanos sobre la pista.
- **Contraluz**: nunca coloques la fuente de luz principal frente a la cámara.
- **Reflejos en acrílico**: si la pista tiene cubierta acrílica, usa luz difusa (LED cenital, no lateral). Coloca papel blanco mate sobre el acrílico si el reflejo persiste.
- **Cambios bruscos de luz** durante una sesión de captura (nube tapando sol = parar y esperar).

**Configuración de IP Webcam en el celular:**

| Ajuste | Valor |
|---|---|
| Resolución | **640×480** (evitar 1080p — latencia alta) |
| FPS | **15–20 FPS** (suficiente para captura; más no aporta) |
| Formato de video | **MJPEG** (menor latencia que H264 en red local) |
| Orientación | **Horizontal / Landscape**, bloqueada (no autorotate) |
| Calidad JPEG | **70–80%** (balance entre peso y detalle) |
| Puerto | 8080 (default) — anotar la IP del celular en el hotspot |
| Audio | **Desactivado** (ahorra batería y ancho de banda) |

> Conectar celular y laptop al **mismo hotspot** (preferir 5 GHz si el celular lo soporta).

---

### A.2 — Checklist de la pista antes de capturar

- [ ] Los 7 segmentos están físicamente estables y sin ondulaciones: **3 rectas, 3 curvas, 1 cruce T**.
- [ ] Ancho de la línea guía: **mínimo 3 cm, recomendado 4–5 cm** (cinta de aislar negra o tape negro mate).
- [ ] Contraste línea/fondo: el fondo debe ser **blanco o gris claro**. Medir con el ojo: la línea debe verse claramente desde 1.5 m de distancia; si hay duda, el modelo tampoco la verá bien.
- [ ] No hay objetos extraños dentro del área de la pista (mochilas, cables, manos).
- [ ] Las uniones entre segmentos no crean saltos de nivel mayores a 2 mm (el robot se desestabiliza y los frames salen movidos).

**Si la pista refleja luz (papel brillante vs papel mate):**
- Preferir siempre papel **mate** o cartulina sin brillo.
- Si solo tienes papel brillante: aplica una capa de aerosol mate o cubre con papel tisú blanco.
- Alternativa rápida: capturar solo con luz artificial cenital y cortinas cerradas — elimina reflejos solares variables.

---

## BLOQUE B — ESTRATEGIA DE CAPTURA POR CLASE

### B.1 — Clases de navegación

#### RECTA

- **Posición del robot:** centrado en un tramo recto, a velocidad normal de operación.
- **Qué debe verse en el frame:** la línea entra por el borde inferior central, sube recta hacia el centro del frame. No debe haber curva visible.
- **Variaciones a capturar:**
  - Robot ligeramente desplazado a la izquierda (10–15% del ancho de la pista).
  - Robot ligeramente desplazado a la derecha.
  - Velocidad lenta (servo 30%), normal (50%), rápida (70%).
  - Iluminación de mañana (luz natural lateral), tarde (contraluz suave), artificial (LED).
- **Frames por sesión:** 600–800 frames válidos (descartar borrosos extremos).
- **Señal de buena captura:** la línea es visible, recta, en el tercio inferior. Sin curvas.
- **Señal de mala captura:** la línea desaparece, aparece en diagonal marcada, o el robot estaba detenido.

#### CURVA_IZQ

- **Posición:** robot en la entrada de una curva que gira a la izquierda.
- **Qué debe verse:** la línea entra por el borde inferior y se curva hacia la **izquierda** del frame.
- **Variaciones:** capturar en 3 momentos de la curva: entrada, mitad, salida.
- **Frames por sesión:** 500–700 frames.
- **Error frecuente:** capturar solo el centro de la curva — asegurarse de incluir entradas y salidas.

#### CURVA_DER

- Igual que CURVA_IZQ pero la línea se curva hacia la **derecha** del frame.
- **Importante:** capturar la misma cantidad que CURVA_IZQ para evitar sesgo direccional.
- **Frames por sesión:** 500–700 frames.

#### GIRO_90_IZQ

- **Posición:** robot justo antes y durante un giro de 90° a la izquierda.
- **Qué debe verse:** la línea hace un quiebre pronunciado hacia la izquierda; puede salir parcialmente del frame.
- **Variaciones:** distintas distancias al quiebre (lejos = línea entrante recta, cerca = quiebre dominando el frame).
- **Frames por sesión:** 400–600 frames (es la clase más difícil de capturar con variedad).

#### GIRO_90_DER

- Espejo de GIRO_90_IZQ hacia la derecha.
- **Frames por sesión:** 400–600 frames.

#### CRUCE_T

- **Posición:** robot aproximándose al cruce T y en el cruce mismo.
- **Qué debe verse:** la línea forma una T visible — la línea longitudinal llega al cruce y la línea transversal cruza horizontalmente.
- **Variaciones:** aproximación desde los 3 posibles lados del cruce T.
- **Frames por sesión:** 500–700 frames.
- **Señal de buena captura:** la T es reconocible en el frame. Mala captura: solo se ve la línea longitudinal sin el cruce.

---

### B.2 — Cronograma sugerido de captura

| Sesión | Día | Duración | Clases objetivo |
|---|---|---|---|
| Sesión 1 | Día 1 | 2 h | RECTA, CURVA_IZQ, CURVA_DER |
| Sesión 2 | Día 2 | 2 h | GIRO_90_IZQ, GIRO_90_DER, CRUCE_T |
| Sesión 3 | Día 3+ | Variable | Refuerzo de clases con accuracy bajo |

> Después de cada sesión: revisar el reporte de balance de `dataset.py` antes de la siguiente sesión.

---

## BLOQUE C — VERIFICACIÓN DEL DATASET ANTES DE ENTRENAR

### C.1 — Ejecutar dataset.py y revisar el reporte

```bash
uv run python dataset.py
```

**Qué buscar en el reporte:**

- Si alguna clase tiene **menos del 15% del total** → capturar más imágenes antes de entrenar.
- La augmentación (x4) **ayuda pero no reemplaza datos reales**:
  - Con ≥ 200 imágenes reales por clase: la augmentación es efectiva.
  - Con < 200 imágenes reales: la augmentación genera variantes de pocos ejemplos — el modelo memoriza esas pocas imágenes. **Capturar más datos reales siempre supera augmentar sobre pocos.**

**Nombres de carpetas — sensible a mayúsculas:**

Los nombres deben coincidir exactamente con lo que espera `dataset.py`. Verificar antes de entrenar:

```
data/
└── navegacion/
    ├── RECTA/
    ├── CURVA_IZQ/
    ├── CURVA_DER/
    ├── GIRO_90_IZQ/
    ├── GIRO_90_DER/
    └── CRUCE_T/
```

> **CURVA_IZQ** ≠ `curva_izq` ≠ `Curva_Izq`. Revisar con `ls data/navegacion/` antes de ejecutar.

---

### C.2 — Revisión visual de 20 imágenes aleatorias por clase

Ejecutar una vista rápida:

```bash
python -c "
import os, random
from PIL import Image
import matplotlib.pyplot as plt

clase = 'RECTA'  # cambiar por la clase a revisar
path = f'data/navegacion/{clase}'
imgs = random.sample(os.listdir(path), min(20, len(os.listdir(path))))
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for ax, img_name in zip(axes.flat, imgs):
    img = Image.open(os.path.join(path, img_name))
    ax.imshow(img)
    ax.set_title(img_name[:15])
    ax.axis('off')
plt.tight_layout()
plt.show()
"
```

**Señales de dataset contaminado — descartar estas imágenes:**

| Señal | Qué significa |
|---|---|
| Imagen completamente borrosa (no distingues la línea) | Frame capturado durante sacudida brusca — eliminar |
| Robot detenido cuando debería estar en movimiento | Frame de pausa — no representa comportamiento real |
| Señal de otra clase visible en el fondo | Contaminación de etiqueta — reclasificar o eliminar |
| Frame completamente negro o sobreexpuesto | Problema de cámara — eliminar |

---

## BLOQUE D — CICLO ITERATIVO DE ENTRENAMIENTO

### D.1 — ITERACIÓN 1: Entrenamiento inicial (epochs 1–20)

**Qué esperar en las curvas:**

- **Loss** debe bajar de forma consistente en los primeros 5 epochs.
- **Accuracy** puede oscilar en los primeros 3 epochs — es normal.

**Señales de alerta tempranas:**

| Síntoma | Causa probable | Acción |
|---|---|---|
| Loss sube después de epoch 5 | Learning rate demasiado alto | Reducir a `lr = 0.0001` |
| Loss no baja en absoluto en epochs 1–3 | Bug en labels o arquitectura | Verificar que las carpetas tienen las imágenes correctas; imprimir 5 labels del DataLoader |
| Accuracy se queda en 1/N (ej. 16.6% para 6 clases) | El modelo predice siempre la misma clase | Dataset desbalanceado severo o bug en el one-hot encoding |

**Umbrales de decisión al terminar iteración 1:**

| Val accuracy (navegación) | Decisión |
|---|---|
| > 70% | Continuar a iteración 2 |
| 50–70% | Capturar más datos de las clases con accuracy más bajo antes de continuar |
| < 50% | Parar. Revisar preprocesamiento, normalización y pipeline de datos |

---

### D.2 — ITERACIÓN 2: Refinamiento (epochs 20–60)

**Detectar sobreajuste (overfitting):**

- Síntoma: `train_accuracy > val_accuracy` con diferencia **> 15 puntos porcentuales**.
- Ejemplo: train 94%, val 76% → overfitting.
- **Acciones:**
  1. Aumentar `Dropout` de 0.3 a 0.5 en las capas densas.
  2. Agregar más augmentation (rotación ±10°, brillo ±20%).
  3. Reducir capacidad: bajar filtros de las capas Conv de (32, 64, 128) a (16, 32, 64).

**Detectar subajuste (underfitting):**

- Síntoma: `train_accuracy` estancado **< 80%** después de epoch 30.
- **Acciones:**
  1. Aumentar epochs a 80–100.
  2. Subir `lr` momentáneamente a `0.001` por 5 epochs y luego volver a bajar.
  3. Verificar que el dataset tiene suficiente variedad (revisar augmentation activa).

---

### D.3 — Cómo leer la matriz de confusión

Ejecutar después de cada entrenamiento:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# (ver código completo en train.py)
```

**Interpretación de confusiones frecuentes:**

| Confusión observada | Significado | Corrección |
|---|---|---|
| RECTA → CURVA_DER | El modelo ve la línea ligeramente descentrada hacia la derecha en rectas | Capturar más RECTAs con robot perfectamente centrado |
| CURVA_IZQ → RECTA | Las curvas capturadas son muy suaves — parecen rectas | Capturar curvas más pronunciadas y en el centro de la curva |
| GIRO_90_IZQ → CURVA_IZQ | Los giros de 90° se capturaron desde demasiado lejos | Capturar giros desde más cerca del quiebre |
| CRUCE_T → RECTA | El cruce fue capturado solo de frente — falta la línea transversal | Capturar desde más cerca del cruce y con el robot lateral |

---

### D.4 — ITERACIÓN 3: Demo-ready

**Criterios mínimos para aprobar el modelo (verificar con `evaluate.py`):**

| Modelo | Métrica | Umbral mínimo |
|---|---|---|
| Navegación | Val accuracy global | ≥ 85% |
| Navegación | Recall por clase (todas) | ≥ 75% |

> Si alguna clase tiene recall < 75%: capturar más imágenes de esa clase y reentrenar antes de la demo.

---

### D.5 — Prueba de humo antes de la demo

1. Colocar el robot al inicio de la pista.
2. Ejecutar:
   ```bash
   uv run python main_robot.py --display
   ```
3. Dejar correr **2 minutos completos** sin intervención.
4. **El robot debe completar el circuito** y detenerse exactamente 2 s en CRUCE_T.
5. Medir FPS reales del pipeline:
   ```python
   import time
   start = time.time()
   # correr 100 predicciones
   fps = 100 / (time.time() - start)
   print(f"FPS: {fps:.1f}")
   ```

**Umbrales de FPS:**

| FPS medidos | Estado | Acción |
|---|---|---|
| ≥ 12 FPS | Óptimo | Sin cambios |
| 8–12 FPS | Aceptable | Monitorear en demo |
| < 8 FPS | Inaceptable | Ver optimizaciones abajo |

**Si FPS < 8 — checklist de optimización:**

1. Reducir resolución del stream: de 640×480 a **320×240**.
2. Bajar calidad JPEG en IP Webcam a **60%**.
3. Cambiar frecuencia WiFi del hotspot a **5 GHz** (si el celular lo soporta).
4. En `main_robot.py`: asegurarse de procesar 1 de cada 2 frames (`frame_skip = 2`).
5. Si aún < 8 FPS: simplificar arquitectura — eliminar una capa Conv y reducir la capa Dense de 256 a 128 neuronas.

---

## BLOQUE E — PROBLEMAS FRECUENTES Y SOLUCIONES

### E.1 — "El robot gira cuando debería ir recto"

**Causa probable:** la clase RECTA está subrepresentada en el dataset, o fue capturada con el robot levemente ladeado (la línea no estaba centrada en el frame durante la captura).

**Solución concreta:**
1. Revisar el reporte de `dataset.py` — ¿RECTA tiene < 15% del dataset de navegación?
2. Abrir 20 imágenes aleatorias de RECTA: ¿la línea está centrada o desplazada?
3. Si está desplazada: esas imágenes están mal etiquetadas (deberían ser CURVA leve). Eliminarlas.
4. Capturar una sesión adicional de RECTA (300+ frames) con el robot perfectamente centrado en un tramo recto.
5. Reentrenar desde cero (no fine-tune — los pesos viejos ya aprendieron el sesgo).

---

### E.2 — "Una clase tiene recall < 75% — el robot se equivoca en ese segmento"

**Diagnóstico:**
1. Ejecutar `uv run python evaluate.py` y revisar la matriz de confusión.
2. Identificar con qué clase se confunde la clase problemática.

**Solución:**
- Capturar 200+ imágenes adicionales de esa clase en las condiciones exactas donde falla.
- Si CURVA_IZQ se confunde con RECTA: capturar curvas más pronunciadas, no solo el centro de la curva.
- Si GIRO_90 se confunde con CURVA: capturar los giros desde más cerca del quiebre.
- Si CRUCE_T se confunde con RECTA: asegurarse de que la T transversal es visible en los frames.
- Reentrenar desde el último checkpoint: `uv run python train.py --resume`

---

### E.3 — "El stream del celular llega a 5 FPS en la demo"

**Checklist de optimización en orden de impacto:**

| Paso | Ajuste | Ganancia esperada |
|---|---|---|
| 1 | Reducir resolución: 640×480 → **320×240** | +3–5 FPS |
| 2 | Cambiar hotspot a **5 GHz** | +2–4 FPS (elimina interferencias) |
| 3 | Bajar calidad JPEG en IP Webcam a **60%** | +1–2 FPS |
| 4 | Cerrar otras apps en el celular | +1–2 FPS |
| 5 | Aumentar threads en `cv2.VideoCapture` con `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` | Reduce latencia acumulada |
| 6 | Frame skip: procesar 1 de cada 2 frames en el loop principal | Duplica FPS efectivos |

---

### E.4 — "El ESP32 no recibe comandos durante la demo"

**Diagnóstico paso a paso:**

1. **Verificar conectividad básica:**
   ```bash
   ping <IP_DEL_ESP32>
   ```
   - Si no responde: el ESP32 y la laptop no están en la misma red. Verificar que ambos estén conectados al mismo hotspot.

2. **Verificar IP del hotspot:**
   - En el celular (hotspot): ir a Ajustes → Punto de acceso personal → verificar el rango de IPs asignadas.
   - La IP del ESP32 puede haber cambiado si se reinició. Conectar el ESP32 por USB y abrir Serial Monitor (115200 baud) — imprime su IP al conectarse al WiFi.

3. **Verificar puerto UDP en el código:**
   - En `utils.py`, confirmar que `ESP32_IP` y `ESP32_PORT` coincidan con lo que imprime el ESP32 en Serial Monitor.
   - El ESP32 escucha en el puerto `4210` (UDP). Verificar con: `nc -vuz <IP_ESP32> 4210`

4. **Si todo lo anterior está correcto y sigue sin funcionar:**
   - Reiniciar el ESP32 (botón EN/RST).
   - Esperar 10 segundos y verificar en Serial Monitor que imprime "WiFi conectado" y su IP.
   - Si no se conecta al hotspot: verificar SSID y contraseña en el código del ESP32 (`credentials.h` o donde estén definidas).

---

### E.5 — "El modelo tiene 95% accuracy en val pero falla en la pista"

**Causa: Dataset shift** — las imágenes de entrenamiento no representan las condiciones reales del entorno de demo.

**Síntomas típicos:**
- Las imágenes de entrenamiento se capturaron con buena luz y sin ruido.
- La demo ocurre en un salón con luz fluorescente variable, sombras de personas, o ángulo de cámara ligeramente diferente.

**Qué hacer:**
1. Capturar 50–100 imágenes en el entorno **exacto** donde será la demo (mismo salón, misma hora, misma luz).
2. Revisar visualmente esas imágenes: ¿se ven diferentes a las del dataset de entrenamiento?
3. Agregar esas imágenes al dataset con sus etiquetas correctas.
4. Reentrenar con el dataset ampliado.
5. Si no hay tiempo de reentrenar: ajustar físicamente las condiciones de demo para que se parezcan más al entorno de entrenamiento (misma lámpara, misma posición de cámara).

---

## TABLA DE DECISIONES RÁPIDAS — DEMO DAY

> Usar esta tabla durante la demo si algo falla. Decisión en < 30 segundos.

| # | SI pasa esto... | ENTONCES haz esto |
|---|---|---|
| 1 | El robot gira en rectas | Verificar que RECTA tiene ≥ 15% del dataset; capturar más y reentrenar |
| 2 | Una clase tiene recall < 75% | Ver sección E.2; capturar más imágenes de esa clase en condiciones reales |
| 3 | FPS < 8 en demo | Reducir resolución a 320×240 en IP Webcam; cerrar apps pesadas en la laptop |
| 4 | ESP32 no responde | Reiniciar ESP32, abrir Serial Monitor, verificar IP nueva, actualizar `ESP32_IP` en utils.py |
| 5 | Stream del celular se corta | Revisar batería del celular; acercar celular al hotspot; reiniciar IP Webcam |
| 6 | El modelo confunde CURVA_IZQ con CURVA_DER | Revisar que las carpetas no están intercambiadas; verificar 20 imágenes de cada clase |
| 7 | Val accuracy subió pero la pista sigue fallando | Dataset shift; capturar imágenes en el entorno real de demo y reentrenar |
| 8 | Training loss sube después de epoch 10 | Reducir learning rate a 0.0001 y reanudar: `uv run python train.py --resume` |
| 9 | El robot no se detiene en CRUCE_T | Verificar que state_machine.py tiene `T_CRUCE_STOP = 2.0` y que CMD_STOP=0x00 llega al ESP32 |
| 10 | El robot completa la pista pero es muy lento | Aumentar `VEL_NORMAL` en esp32_firmware.ino y reflashear; no es problema del modelo |

---

*Última actualización: 2026-05-06 — Reto Bonus cancelado. Solo navegación.*

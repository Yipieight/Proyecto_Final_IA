# Estado del Modelo — Proyecto IA Final
**Universidad Rafael Landívar · 2026-1**

---

## 1. Dataset Final

| Clase | Imágenes | Estado |
|---|---|---|
| RECTA | 592 | ✅ |
| CURVA_IZQ | 594 | ✅ |
| CURVA_DER | 341 | ✅ |
| GIRO_90_IZQ | 496 | ✅ |
| GIRO_90_DER | 481 | ✅ |
| CRUCE_T | 292 | ✅ |
| **TOTAL** | **2,796** | |

---

## 2. Resultados de Evaluación (nav_model.pth)

Entrenado con split 80/10/10 · semilla 42 · 60 epochs · Early stop en epoch 7 · MPS (Apple Silicon)

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| RECTA | 1.00 | 0.98 | 0.99 | 61 |
| CURVA_IZQ | 1.00 | 1.00 | 1.00 | 59 |
| CURVA_DER | 0.97 | 1.00 | 0.98 | 32 |
| GIRO_90_IZQ | 0.96 | 1.00 | 0.98 | 47 |
| GIRO_90_DER | 1.00 | 0.98 | 0.99 | 48 |
| CRUCE_T | 1.00 | 0.97 | 0.99 | 34 |

**Accuracy global: 98.93%** · Avg Loss: 0.0296 · Estado: ✅ DEMO-READY

---

## 3. Análisis de Comportamiento en Video (test.mp4 · 11s)

### Zonas estables ✅
| Sección | Tiempo | Confianza |
|---|---|---|
| RECTA | t=0.2–0.6s | 93–99% |
| CURVA_IZQ | t=5.7–7.8s | 100% sostenido |
| GIRO_90_IZQ | t=8.0–10.6s | 99–100% sostenido |

### Zona problemática ⚠️
**t=1.0s a t=5.6s — oscilación entre GIRO_90_DER / GIRO_90_IZQ / RECTA con confianza 29–67%**

**Causas identificadas:**
1. GIRO_90_IZQ y GIRO_90_DER son imágenes espejo — el modelo duda entre ambas cuando el ángulo de cámara no coincide exactamente con el entrenamiento
2. El FrameBuffer (stack de 3 frames) contamina transiciones: durante ~3 frames al cambiar de sección, el buffer mezcla dos clases distintas

**Impacto en robot real:** La máquina de estados suaviza oscilaciones cortas, pero 4.5s de confusión podría generar un comando incorrecto sostenido.

**Solución pendiente:** Grabar más videos de GIRO_90_DER desde el ángulo exacto de la cámara del robot.

---

## 4. Problema de Perspectiva / Timing (pendiente de implementar)

### Descripción del problema
La cámara detecta la CURVA_IZQ **antes** de que el robot físicamente llegue a ella.
El robot todavía está en la recta cuando recibe el comando de giro → gira antes de tiempo.

```
Vista de la cámara:        Posición real del robot:
┌──────────────┐
│   CURVA_IZQ  │  ← el modelo ya la ve      [robot] ──── recta ──── [curva]
│   detectada  │                                 ↑
└──────────────┘                            todavía aquí
```

### Causas
- La cámara mira hacia adelante con cierto ángulo, captando 1–3 metros de distancia
- El modelo clasifica lo que VE en el frame, no donde ESTÁ el robot físicamente
- La distancia entre "detección" y "llegada física" depende de la velocidad del robot y el ángulo de la cámara

### Soluciones (de menor a mayor complejidad)

#### Opción A — Ajuste físico de la cámara (recomendada para empezar)
Inclinar la cámara más hacia abajo para que vea el suelo más cercano en lugar de lo que está lejos.
- ↑ ángulo de inclinación = ↓ distancia de look-ahead
- No requiere reentrenar el modelo si el cambio es pequeño
- Si el cambio es grande → recolectar datos nuevos con la nueva posición

#### Opción B — Delay de comando en la máquina de estados
Agregar un contador en `state_machine.py`: cuando se detecta una clase nueva, esperar N frames estables antes de enviar el comando al ESP32.

```python
# En state_machine.py — concepto
CONFIRM_FRAMES = 8   # ~0.5s a 15fps

if nueva_clase == self.pending_clase:
    self.pending_count += 1
else:
    self.pending_clase = nueva_clase
    self.pending_count = 1

if self.pending_count >= CONFIRM_FRAMES:
    self.clase_activa = self.pending_clase  # recién aquí se actúa
```

- No requiere reentrenar
- El valor de CONFIRM_FRAMES se calibra empíricamente con el robot

#### Opción C — Modificar el ROI (requiere reentrenar)
Cambiar el recorte superior del frame de 35% a 50–55% en `preprocessing.py`.
El modelo vería solo el suelo cercano → less look-ahead.

```python
# En preprocessing.py
roi = frame[int(h * 0.50):, :]   # antes era 0.35
```

- Requiere recolectar datos nuevos con el ROI ajustado
- Solución más robusta a largo plazo

#### Opción D — Calibración por velocidad (avanzado)
Medir experimentalmente cuántos frames tarda el robot en recorrer la distancia de look-ahead a su velocidad normal. Usar ese número como delay fijo en el envío de comandos.

### Recomendación de implementación
1. Primero probar **Opción B** (delay de frames) — sin reentrenar, ajustable en segundos
2. Si el timing sigue mal, combinar con **Opción A** (ajuste físico de cámara)
3. Reservar **Opción C** para una segunda iteración del proyecto

---

## 5. Archivos del Proyecto

| Archivo | Descripción |
|---|---|
| `models/nav_model.pth` | Modelo final entrenado (NavCNN, 6 clases) |
| `models/quick_test.pth` | Mini-modelo para pruebas rápidas |
| `train.py` | Entrenamiento completo (60 epochs, 80/10/10 split) |
| `evaluate.py` | Evaluación con matriz de confusión |
| `quick_test.py` | Prueba rápida en video o cámara en vivo |
| `main_robot.py` | Pipeline completo con ESP32 |
| `state_machine.py` | Máquina de estados (NAVIGATING / STOPPED_T / CHOOSING_T) |
| `preprocessing.py` | ROI + Gaussian blur + Sobel edges |
| `extract_frames.py` | Extracción de frames desde video iPhone |
| `dataset.py` | NavigationDataset + get_loaders |
| `model_nav.py` | Arquitectura NavCNN |
| `esp32_firmware/` | Firmware Arduino para control de motores |

---

## 6. Comandos Útiles

```bash
# Entrenar modelo completo
uv run python train.py

# Evaluar modelo
uv run python evaluate.py

# Probar en video
uv run python quick_test.py --test ~/Downloads/video.mp4 --nav

# Probar con cámara iPhone (Continuity Camera)
uv run python quick_test.py --camera 2 --nav --scale 0.3

# Pipeline completo con robot
uv run python main_robot.py --display --ip 192.168.x.x

# Extraer frames de video
uv run python extract_frames.py ~/Downloads/video.mp4 CLASE --fps 5 --blur 5
```

---

*Última actualización: 2026-05-11*

# Comandos del Proyecto — Robot IA

## Configuración rápida

Antes de cualquier prueba, verifica que `utils.py` tenga la IP correcta del ESP32:

```python
ESP32_IP   = "172.20.10.3"   # IP que aparece en el Serial Monitor al arrancar
ESP32_PORT = 9999
```

---

## 1. Control manual por teclado

```bash
uv run python test_manual.py
```

Con IP personalizada (si cambia la red):

```bash
uv run python test_manual.py --ip 172.20.10.3
```

**Controles:**

| Tecla | Acción |
|-------|--------|
| `W`   | Adelante |
| `A`   | Izquierda |
| `D`   | Derecha |
| `S`   | Stop |
| `Q`   | Salir (envía STOP antes de cerrar) |

> Modo sticky: el robot mantiene el último comando hasta que presiones `S`.

---

## 2. Cámara en vivo (Continuity Camera del iPhone)

Ver qué cámaras están disponibles:

```bash
uv run python quick_test.py --camera list
```

Encender la cámara sin IA (solo vista):

```bash
uv run python quick_test.py --camera 2
```

Encender la cámara **con IA de navegación**:

```bash
uv run python quick_test.py --camera 2 --nav
```

Con ventana más pequeña (recomendado):

```bash
uv run python quick_test.py --camera 2 --nav --scale 0.3
```

> Si el iPhone no aparece: activa Continuity Camera en Ajustes → General → AirPlay y Handoff.

---

## 3. Probar la IA con un video

```bash
uv run python quick_test.py --test ~/Downloads/test2.mp4 --nav
```

Sin modelo (solo ver el video preprocesado):

```bash
uv run python quick_test.py --test ~/Downloads/test2.mp4
```

Con ventana escalada:

```bash
uv run python quick_test.py --test ~/Downloads/test2.mp4 --nav --scale 0.5
```

---

## 4. Entrenar el modelo

Entrenamiento completo (60 épocas):

```bash
uv run python train.py
```

Reanudar entrenamiento previo:

```bash
uv run python train.py --resume
```

Número de épocas personalizado:

```bash
uv run python train.py --epochs 30
```

> El modelo se guarda en `models/nav_model.pth`. Las métricas en `metrics/`.

---

## 5. Dataset — captura y extracción de frames

Ver balance del dataset actual:

```bash
uv run python dataset.py
```

Extraer frames de un video a una clase:

```bash
uv run python extract_frames.py ~/Downloads/video.mp4 RECTA
uv run python extract_frames.py ~/Downloads/video.mp4 CURVA_IZQ
uv run python extract_frames.py ~/Downloads/video.mp4 CURVA_DER
uv run python extract_frames.py ~/Downloads/video.mp4 GIRO_90_IZQ
uv run python extract_frames.py ~/Downloads/video.mp4 GIRO_90_DER
uv run python extract_frames.py ~/Downloads/video.mp4 CRUCE_T
```

Con opciones avanzadas:

```bash
# 8 fps, saltar los primeros 2 segundos, ver preview
uv run python extract_frames.py ~/Downloads/video.mp4 RECTA --fps 8 --skip 2 --preview

# Límite de 300 frames y umbral de blur más estricto
uv run python extract_frames.py ~/Downloads/video.mp4 CURVA_DER --limit 300 --blur 60
```

---

## 6. Robot autónomo (pipeline completo)

```bash
uv run python main_robot.py --display
```

> Requiere `models/nav_model.pth` entrenado y ESP32 conectado.  
> `--display` activa la ventana con HUD de navegación.

---

## 7. Evaluación del modelo

```bash
uv run python evaluate.py
```

---

## Clases de navegación

| Clase | Descripción |
|-------|-------------|
| `RECTA` | Tramo recto |
| `CURVA_IZQ` | Curva suave izquierda |
| `CURVA_DER` | Curva suave derecha |
| `GIRO_90_IZQ` | Giro de 90° izquierda |
| `GIRO_90_DER` | Giro de 90° derecha |
| `CRUCE_T` | Cruce en T |

---

## Rutas importantes

```
data/navegacion/     ← imágenes del dataset por clase
models/nav_model.pth ← modelo entrenado
metrics/             ← curvas de pérdida y accuracy
```

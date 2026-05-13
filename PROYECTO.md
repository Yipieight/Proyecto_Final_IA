# Proyecto de IA — Robot Navegador con CNN

Documentación completa del proyecto para retomar trabajo en una nueva sesión.

---

## 1. Resumen del proyecto

Robot 4WD que navega autónomamente por una pista de franjas blancas/negras usando:
- **iPhone como cámara** (vía Continuity Camera de macOS)
- **Mac** corre la inferencia con un CNN entrenado en PyTorch
- **ESP32** recibe comandos UDP y controla los motores vía L298N

**Clases de navegación (6):**
| Clase | Patrón visual | Acción |
|-------|---------------|--------|
| `RECTA` | Franjas verticales | Avanzar |
| `CURVA_IZQ` | Franjas curvas hacia izq | Curvar izquierda (diferencial) |
| `CURVA_DER` | Franjas curvas hacia der | Curvar derecha (diferencial) |
| `GIRO_90_IZQ` | Esquina 90° hacia izq | Parar 1s + pivotar 1s izquierda |
| `GIRO_90_DER` | Esquina 90° hacia der | Parar 1s + pivotar 1s derecha |
| `CRUCE_T` | Cruce en T horizontal | Parar 2s + giro aleatorio 0.8s |

---

## 2. Hardware

### ESP32 + L298N
- **ESP32**: WiFi STA, recibe UDP byte por byte en puerto 9999
- **L298N**: 2 canales H-bridge, 4 motores TT en paralelo (2 por canal)
- **Baterías**: 6×AA (≈9V)
- **Jumper ENA/ENB**: puesto (5V permanente para habilitar)

### Pines ESP32
```
ENA=25 (PWM lado izq)   IN1=26  IN2=27
ENB=14 (PWM lado der)   IN3=13  IN4=12
```

### Cableado motores
```
OUT1/OUT2 → 2 motores izquierdos en paralelo
OUT3/OUT4 → 2 motores derechos en paralelo (polaridad INVERTIDA respecto al izq)
```

**Nota importante:** la polaridad del lado derecho está invertida en el código del ESP32 (`derDir=1` usa IN3=LOW, IN4=HIGH). Esto es a propósito para que ambos lados avancen físicamente al mismo tiempo.

### Red WiFi
- ESP32 actúa como cliente WiFi (modo STA)
- IP típica: `10.202.169.157` (cambia según la red)
- Puerto UDP: `9999`
- La IP del ESP32 se actualiza en `utils.py` → `ESP32_IP`

---

## 3. Protocolo UDP (1 byte por comando)

| Byte | Comando | Comportamiento (firmware actual) |
|------|---------|---------------------------------|
| `0x00` | STOP | Detener |
| `0x01` | ADELANTE | Ambos lados PWM 10 |
| `0x02` | CURVA_IZQ | Izq PWM 5, Der PWM 20 (diferencial) |
| `0x03` | CURVA_DER | Izq PWM 20, Der PWM 5 (diferencial) |
| `0x04` | GIRO_IZQ | Solo lado derecho PWM 255 |
| `0x05` | GIRO_DER | Solo lado izquierdo PWM 255 |

Los comandos los envía la state machine (`state_machine.py`) según las predicciones del CNN.

---

## 4. Pipeline de software

```
iPhone (Continuity Camera)
  └─► cv2.VideoCapture (cámara índice 1 o 2)
       └─► preprocess_frame() → 64×64 grayscale + ROI top 35% recortado
            └─► FrameBuffer (3 frames apilados)
                 └─► NavCNN.forward() → 6 logits
                      └─► argmax → clase
                           └─► RobotStateMachine.update()
                                └─► UDP byte → ESP32
                                     └─► setMotores() → L298N
```

---

## 5. Estructura del repositorio

```
proyecto_de_IA_final/
├── data/navegacion/          ← imágenes de entrenamiento
│   ├── RECTA/                    984 imgs (3 videos)
│   ├── CURVA_IZQ/               1079 imgs (3 videos)
│   ├── CURVA_DER/               1051 imgs (3 videos)
│   ├── GIRO_90_IZQ/              900 imgs (3 videos)
│   ├── GIRO_90_DER/              953 imgs (3 videos)
│   └── CRUCE_T/                  862 imgs (4 videos)
├── models/
│   └── nav_model.pth         ← modelo entrenado actual (val_acc ~0.55)
├── metrics/                   ← curvas, matriz de confusión, JSONs
├── utils.py                   ← constantes globales, ESP32_IP/PORT
├── preprocessing.py           ← preprocess_frame() con ROI
├── model_nav.py               ← NavCNN (4 conv + 3 FC)
├── dataset.py                 ← Dataset PyTorch + augment()
├── train_kfold.py             ← entrenamiento K-Fold (PRINCIPAL)
├── train.py                   ← entrenamiento simple (deprecated)
├── state_machine.py           ← lógica de estados del robot
├── main_robot.py              ← pipeline completo cámara→IA→ESP32
├── quick_test.py              ← prueba con video o cámara (sin ESP32)
├── diagnose_test.py           ← matriz de confusión + diagnóstico
├── visualize_dataset.py       ← genera PNGs de muestras del dataset
├── test_manual.py             ← control manual por teclado WASD
├── extract_frames.py          ← extrae frames de video → carpeta de clase
└── capture_dataset.py         ← captura dataset desde IP Webcam
```

---

## 6. Comandos principales

### Control manual del robot (teclado)
```bash
uv run python test_manual.py
```
W=adelante, A=izq, D=der, S=stop, Q=salir. Modo sticky (mantiene último comando).

### Probar IA con video
```bash
uv run python quick_test.py --test ~/Downloads/video.mp4 --nav --scale 0.5
```

### Probar IA con cámara (sin enviar al robot)
```bash
uv run python quick_test.py --camera 1 --nav --scale 0.5
```

### Listar cámaras disponibles
```bash
uv run python quick_test.py --camera list
```

### Pipeline completo (cámara → IA → ESP32)
```bash
uv run python main_robot.py --camera 1 --display --scale 0.5
```

Con look-ahead delay (ms):
```bash
uv run python main_robot.py --camera 1 --display --scale 0.5 --delay 800
```

### Ver matriz de confusión sin reentrenar
```bash
uv run python diagnose_test.py
```

### Visualizar muestras del dataset
```bash
uv run python visualize_dataset.py
```
Genera `metrics/dataset_samples_raw.png` y `dataset_samples_processed.png`.

### Reporte de balance del dataset
```bash
uv run python dataset.py
```

---

## 7. Entrenamiento

### Comando principal
```bash
uv run python train_kfold.py --folds 2 --epochs 5
```

### Cómo elegir épocas
1. Corre con muchas épocas (15-30) UNA vez
2. Mira en qué época `val_acc` llegó a su pico
3. Reentrena con ese número exacto (early stopping manual)

**Tu modelo actual peakeó en época 3-5.** Después de eso solo memoriza (sobreajuste).

### Cómo leer las métricas
```
Epoch 5 | train=0.95  val=0.55  Δ=+0.40 (sobreajuste)
```

| Δ (train - val) | Significado |
|-----------------|-------------|
| < 0.10 | Sano, generaliza bien |
| 0.10 - 0.20 | Sobreajuste leve |
| 0.20 - 0.30 | Sobreajuste fuerte |
| > 0.30 | Memorización pura |

**Reglas de cuándo parar:**
- Train ↑ val ↑ → continuar
- Train ↑ val estancado → parar pronto
- Train ↑ val ↓ → PARAR YA

### Re-extraer dataset desde cero
```bash
# Paso 1: limpiar
BASE="/Users/josegarcia/Documents/GitHub/proyecto_de_IA_final/data/navegacion"
for cls in RECTA CURVA_IZQ CURVA_DER GIRO_90_IZQ GIRO_90_DER CRUCE_T; do
  rm -f "$BASE/$cls/"*.jpg
done

# Paso 2: extraer cada video con --skip 3 (cortar 3s al inicio)
# y --limit calculado para cortar también 3s al final (limit = (duración - 6) * 5)
# Ejemplo:
uv run python extract_frames.py "/Users/josegarcia/Movies/recta_1.mp4" RECTA --skip 3 --limit 329 --blur 15
# ... etc para todos los videos
```

---

## 8. State Machine (lógica de decisiones del robot)

Archivo: `state_machine.py`

```
NAVIGATING
  ├─ CURVA_IZQ/CURVA_DER → CMD_LEFT/RIGHT (continuo, sin parar)
  ├─ RECTA → CMD_FORWARD
  ├─ GIRO_90_* → STOPPED_GIRO (1s STOP) → TURNING_GIRO (1s pivote) → NAVIGATING
  └─ CRUCE_T → STOPPED_T (2s STOP) → CHOOSING_T (0.8s giro random) → NAVIGATING
```

**Constantes ajustables:**
```python
T_CRUCE_STOP = 2.0   # segundos parado en cruce T
T_CRUCE_TURN = 0.8   # segundos girando en cruce T
T_GIRO_STOP  = 1.0   # segundos parado antes de giro 90°
T_GIRO_TURN  = 1.0   # segundos pivotando para giro 90°
```

---

## 9. Firmware ESP32 (Arduino IDE)

Ubicación: subido manualmente al ESP32 vía Arduino IDE.

Estructura clave:
```cpp
void setMotores(int izqDir, int derDir, int velIzq, int velDer) {
  // Motor izquierdo (OUT1/OUT2)
  // ... configura IN1/IN2 según izqDir
  // Motor derecho (OUT3/OUT4) — POLARIDAD INVERTIDA respecto al izq
  // ... configura IN3/IN4 según derDir
  ledcWrite(0, velIzq);
  ledcWrite(1, velDer);
}
```

**Velocidades actuales:**
- ADELANTE: ambos 10 (lento pero suficiente)
- CURVA: lado lento 5, lado rápido 20 (diferencial 4×)
- GIRO 90°: un lado 0 (libre), otro 255 (máximo)

**Recuerda actualizar SSID/PASS antes de subir.**

---

## 10. Estado actual del modelo (al cierre de la última sesión)

### Métricas K-Fold
```
Fold 1: val_acc = 0.4424
Fold 2: val_acc = 0.4863
Media : 0.4643
Desvío: 0.0219 [OK]
```

### Test fijo (videos held-out)
```
Accuracy: 0.5849

Recall por clase:
  RECTA           255/512   49.8%
  CURVA_IZQ       304/334   91.0%   ← mejor
  CURVA_DER       275/338   81.4%
  GIRO_90_IZQ     111/280   39.6%
  GIRO_90_DER     104/315   33.0%   ← peor
  CRUCE_T         188/336   56.0%
```

### Sobre el dataset
- **5829 imágenes totales** distribuidas balanceadamente
- **3-5 videos por clase** (5 para RECTA con los videos de test)
- Cámara con perspectiva CENITAL (apunta hacia abajo)

### Problemas conocidos
- **RECTA confunde con GIRO_90_DER** frecuentemente cuando el carro está descentrado
- **Sobreajuste moderado** (train llega a 99% en pocas épocas)
- **Pocos videos por clase** → modelo no generaliza bien a condiciones nuevas (iluminación, ángulo)

### Para mejorar (orden de impacto)
1. Grabar **2-3 videos más por clase** en distintas condiciones (luz, posición lateral del carro)
2. Después reentrenar con `--folds 2 --epochs 5`
3. Si train sigue saturando rápido, considerar regularización adicional o reducir tamaño del modelo

---

## 11. Configuración importante en `utils.py`

```python
IMG_WIDTH    = 64
IMG_HEIGHT   = 64
FRAME_STACK  = 3              # frames consecutivos apilados
ROI_TOP_FRAC = 0.35           # descarta 35% superior del frame
ESP32_IP     = "10.202.169.157"  # ACTUALIZAR si cambia la red
ESP32_PORT   = 9999
BATCH_SIZE   = 32
LEARNING_RATE = 0.001         # train_kfold usa override a 1e-4
NUM_EPOCHS_NAV = 60
```

---

## 12. Bugs corregidos durante el desarrollo (referencia histórica)

### Bug crítico de MPS — argmax/comparación inconsistente
- **Síntoma**: `val_acc` reportado era 0.97 pero modelo cargado predecía 100% RECTA
- **Causa**: en MPS, leer dos veces el tensor `preds` (uno para `==y` y otro para `.cpu().tolist()`) daba valores diferentes
- **Fix**: ahora todas las comparaciones se hacen en CPU (`preds_cpu = preds.cpu()` antes de comparar). Líneas relevantes en `train_kfold.py` dentro de `train_fold()`.

### Bug de polaridad motores
- **Síntoma**: presionando D todas las ruedas iban adelante (debían girar)
- **Causa**: los motores del lado derecho están físicamente montados en espejo
- **Fix**: en `setMotores()` del ESP32, lado derecho usa IN3=LOW/IN4=HIGH para "adelante" (inverso al izq)

### Bug del DelayBuffer
- **Síntoma**: con `--delay 800` el robot no se movía nada
- **Causa**: cada frame reemplazaba el comando pendiente con uno nuevo 800ms en el futuro → nunca llegaba a la hora de envío
- **Fix**: solo programar nuevo comando si no hay uno pendiente o si cambió el tipo de comando

### Bug del K-Fold splitting (anterior)
- **Síntoma**: train_acc=99% val_acc=85% pero test=14%
- **Causa**: split por frame individual → leakage temporal (frames consecutivos en train y val)
- **Fix**: `make_folds()` ahora divide por SESIÓN de video, no por frame

### Bug del session_of
- **Síntoma**: test fijo crecía a 86% de los datos en lugar del 20%
- **Causa**: `session_of()` no detectaba bien los videos físicos
- **Fix**: usa reset del frame_num en el filename (`f_num < prev_frame` → nuevo video)

---

## 13. Para una próxima sesión

**Si vienes a continuar con esto, primero:**
1. Verifica IP del ESP32 en `utils.py` (revisa Serial Monitor del ESP32)
2. Confirma que el iPhone está conectado (Continuity Camera activo)
3. Identifica el índice de cámara del iPhone: `uv run python quick_test.py --camera list`
4. Prueba control manual primero: `uv run python test_manual.py`
5. Si todo funciona, lanza pipeline completo: `uv run python main_robot.py --camera 1 --display --scale 0.5`

**Si quieres mejorar el modelo:**
1. Graba 2-3 videos NUEVOS por clase con luz/ángulo distintos
2. Guárdalos en `~/Movies/` con nombre `<clase>_extra_N.mp4`
3. Extrae frames: `uv run python extract_frames.py "video.mp4" CLASE --skip 3 --limit X --blur 15`
4. Reentrena: `uv run python train_kfold.py --folds 2 --epochs 5`
5. Diagnostica: `uv run python diagnose_test.py`

**Si la IA está dando solo RECTA o solo una clase:**
- Es el bug de MPS en la verificación. Verifica que `train_kfold.py` use `preds.cpu()` antes de comparar.
- Si reentrenas y sigue, revisa la matriz de confusión con `diagnose_test.py`.

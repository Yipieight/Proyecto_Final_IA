"""
Pipeline principal de inferencia y control del robot.

Flujo:
  Cámara (Continuity Camera o URL MJPEG) → Thread captura → Queue → Thread inferencia
  → preprocess → FrameBuffer → NavCNN → StateMachine
  → comando UDP (1 byte) → ESP32

Uso:
    uv run python main_robot.py --camera 2            # iPhone via Continuity Camera
    uv run python main_robot.py --camera 2 --display  # con ventana de debug
    uv run python main_robot.py --camera list         # ver índices disponibles
"""

import argparse
import os
import socket
import time
import queue
import threading
import cv2
import numpy as np
import torch

from utils import (
    FRAME_STACK,
    ESP32_IP, ESP32_PORT,
    MODEL_NAV_PATH,
    NAV_IDX_CLASS, CMD_NAME,
    CMD_STOP,
    FrameBuffer,
)
from preprocessing import preprocess_frame
from model_nav import build_nav_model
from state_machine import RobotStateMachine


# ── Contador de FPS ───────────────────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window: int = 30):
        self._ts  = []
        self._win = window

    def tick(self) -> None:
        self._ts.append(time.time())
        if len(self._ts) > self._win:
            self._ts.pop(0)

    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ── Envío UDP al ESP32 ─────────────────────────────────────────────────────────

class UDPSender:
    """
    Envía un datagrama UDP de 1 byte al ESP32 en hilo de fondo.
    Solo el comando más reciente importa — la cola descarta el viejo.
    """

    def __init__(self, ip: str = ESP32_IP, port: int = ESP32_PORT):
        self._addr  = (ip, port)
        self._sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._q     = queue.Queue(maxsize=1)
        self._last  = None
        self._ok    = False   # True si el último envío fue exitoso
        self._t     = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def send(self, cmd: int) -> None:
        if cmd == self._last:
            return   # no enviar duplicados consecutivos
        try:
            self._q.put_nowait(cmd)
        except queue.Full:
            pass     # descarta el pending si la cola está llena

    @property
    def wifi_ok(self) -> bool:
        return self._ok

    def close(self) -> None:
        self._sock.close()

    def _worker(self) -> None:
        while True:
            cmd = self._q.get()
            self._last = cmd
            try:
                self._sock.sendto(bytes([cmd]), self._addr)
                self._ok = True
            except Exception:
                self._ok = False


# ── Buffer de delay para compensar look-ahead de la cámara ───────────────────

class DelayBuffer:
    """
    Encola comandos y los libera después de `delay_ms` milisegundos.
    STOP siempre se envía inmediatamente sin delay (seguridad).
    RECTA/ADELANTE tampoco se demoran — solo los giros se retrasan.
    """

    INSTANT_CMDS = {CMD_STOP}   # comandos que nunca esperan

    def __init__(self, sender: 'UDPSender', delay_ms: float = 0):
        self._sender   = sender
        self._delay    = delay_ms / 1000.0
        self._pending: list[tuple[float, int]] = []   # (send_at, cmd)

    def push(self, cmd: int) -> None:
        if cmd in self.INSTANT_CMDS or self._delay == 0:
            self._pending.clear()
            self._sender.send(cmd)
            return
        # Solo programar si no hay uno pendiente o si cambió el comando
        if not self._pending or self._pending[0][1] != cmd:
            self._pending = [(time.time() + self._delay, cmd)]

    def flush(self) -> None:
        if not self._pending:
            return
        now = time.time()
        due = [(t, c) for t, c in self._pending if t <= now]
        if due:
            _, cmd = due[-1]
            self._sender.send(cmd)
            self._pending = [(t, c) for t, c in self._pending if t > now]

    @property
    def wifi_ok(self) -> bool:
        return self._sender.wifi_ok


# ── Thread de captura de frames ────────────────────────────────────────────────

def _list_cameras() -> None:
    print("\nCámaras disponibles:")
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"  [{i}] disponible")
            cap.release()
    print()


def _capture_thread(src, frame_q: queue.Queue, stop_evt: threading.Event) -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara: {src}")
        stop_evt.set()
        return
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[robot] Cámara abierta: {src}")

    while not stop_evt.is_set():
        ret, bgr = cap.read()
        if not ret:
            print("[WARN] Frame perdido, reintentando...")
            time.sleep(0.3)
            continue
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(bgr)

    cap.release()


# ── Carga del modelo ──────────────────────────────────────────────────────────

def _load_nav_model():
    device = torch.device('cpu')
    if not os.path.exists(MODEL_NAV_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_NAV_PATH}\n"
            "  Ejecuta primero: uv run python train.py"
        )
    model = build_nav_model(device)
    ckpt  = torch.load(MODEL_NAV_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[robot] Modelo cargado: {MODEL_NAV_PATH}  "
          f"(val acc: {ckpt.get('best_val_acc', 0):.1%})")
    return model, device


# ── Inferencia ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer(model, stack: np.ndarray, device) -> int:
    """stack: (FRAME_STACK, H, W) float32 → índice de clase"""
    x = torch.tensor(stack[np.newaxis], dtype=torch.float32).to(device)
    return model(x).argmax(1).item()


# ── HUD ───────────────────────────────────────────────────────────────────────

def _draw_hud(bgr: np.ndarray, nav_lbl: str, cmd_lbl: str,
              fps: float, wifi_ok: bool) -> np.ndarray:
    d = bgr.copy()
    green  = (0, 220, 0)
    yellow = (0, 220, 220)
    red    = (0, 60, 220)
    gray   = (160, 160, 160)

    cv2.putText(d, f"NAV: {nav_lbl}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, green, 2)
    cv2.putText(d, f"CMD: {cmd_lbl}",
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2)
    cv2.putText(d, f"FPS: {fps:.1f}",
                (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                green if fps >= 8 else red, 2)

    wifi_txt = "WiFi: OK" if wifi_ok else "WiFi: ERR"
    cv2.putText(d, wifi_txt,
                (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                green if wifi_ok else red, 2)

    cv2.putText(d, "q = salir",
                (10, d.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, gray, 1)
    return d


# ── Loop principal ─────────────────────────────────────────────────────────────

def run(src, display: bool, scale: float = 1.0, delay_ms: float = 0) -> None:
    print(f"\n[robot] Iniciando pipeline.")
    print(f"  Cámara : {src}")
    print(f"  ESP32  : {ESP32_IP}:{ESP32_PORT} (UDP)")
    if delay_ms > 0:
        print(f"  Delay  : {delay_ms:.0f} ms (look-ahead compensación)")

    model, device = _load_nav_model()

    buf     = FrameBuffer(FRAME_STACK)
    sm      = RobotStateMachine()
    sender  = UDPSender(ip=ESP32_IP, port=ESP32_PORT)
    delayed = DelayBuffer(sender, delay_ms)
    fps_ctr = FPSCounter()

    # Lectura directa de cámara (sin thread) — más responsivo, mismo flujo que quick_test
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara: {src}")
        return
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[robot] Cámara abierta: {src}")

    print("[robot] Loop activo. Ctrl+C para detener.\n")
    fps_warn_shown = False
    last_nav_lbl   = "---"
    last_cmd_lbl   = "---"
    stop_flag      = False

    try:
        while not stop_flag:
            ret, bgr = cap.read()
            if not ret:
                print("\n[WARN] Frame perdido, reintentando...")
                time.sleep(0.1)
                continue

            proc = preprocess_frame(bgr)
            buf.push(proc)

            nav_idx      = _infer(model, buf.get_stack(), device)
            cmd_byte     = sm.update(nav_idx)
            delayed.push(cmd_byte)
            delayed.flush()

            fps_ctr.tick()
            fps = fps_ctr.fps()

            last_nav_lbl = NAV_IDX_CLASS.get(nav_idx, '?')
            last_cmd_lbl = CMD_NAME.get(cmd_byte, str(cmd_byte))
            state        = sm.get_state()

            print(f"\r  FPS:{fps:5.1f} | Nav:{last_nav_lbl:<15}"
                  f"| Estado:{state:<15}| CMD:{last_cmd_lbl:<12}"
                  f"| WiFi:{'OK ' if sender.wifi_ok else 'ERR'}",
                  end='', flush=True)

            if fps < 8 and fps > 0 and not fps_warn_shown:
                print(f"\n[WARN] FPS bajo ({fps:.1f} < 8). "
                      "Reduce resolución en IP Webcam o cierra apps pesadas.")
                fps_warn_shown = True

            if display:
                hud = _draw_hud(bgr, last_nav_lbl, last_cmd_lbl,
                                fps, delayed.wifi_ok)
                if scale != 1.0:
                    h, w = hud.shape[:2]
                    hud = cv2.resize(hud, (int(w * scale), int(h * scale)))
                cv2.imshow("Robot — q para salir", hud)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag = True
                    break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        print(f"\n\n[robot] Enviando STOP al ESP32...")
        sender.send(CMD_STOP)
        time.sleep(0.15)
        sender.close()
        if display:
            cv2.destroyAllWindows()
        print("[robot] Finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline principal del robot")
    parser.add_argument("--camera",  type=str, default="2",
                        help="Índice de cámara (ej: 2) o 'list' para ver disponibles")
    parser.add_argument("--display", action="store_true",
                        help="Mostrar ventana de debug con OpenCV")
    parser.add_argument("--scale",   type=float, default=1.0,
                        help="Escala de la ventana de display (ej: 0.5)")
    parser.add_argument("--delay",   type=float, default=0,
                        help="Delay en ms para compensar look-ahead (ej: 800)")
    args = parser.parse_args()

    if args.camera == "list":
        _list_cameras()
    else:
        src = int(args.camera) if args.camera.isdigit() else args.camera
        run(src, args.display, args.scale, args.delay)

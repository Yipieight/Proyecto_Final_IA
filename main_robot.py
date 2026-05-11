"""
Pipeline principal de inferencia y control del robot.

Flujo:
  IP Webcam (MJPEG) → Thread captura → Queue → Thread inferencia
  → preprocess → FrameBuffer → NavCNN → StateMachine
  → comando UDP (1 byte) → ESP32

Uso:
    uv run python main_robot.py
    uv run python main_robot.py --display          # ventana de debug
    uv run python main_robot.py --ip 192.168.2.1   # IP del celular
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


# ── Thread de captura de frames ────────────────────────────────────────────────

def _capture_thread(url: str, frame_q: queue.Queue, stop_evt: threading.Event) -> None:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir stream: {url}")
        stop_evt.set()
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[robot] Stream abierto: {url}")

    while not stop_evt.is_set():
        ret, bgr = cap.read()
        if not ret:
            print("[WARN] Frame perdido, reintentando...")
            cap.release()
            time.sleep(0.3)
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        # mantener solo el frame más reciente
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

def run(ip: str, port: int, display: bool) -> None:
    url = f"http://{ip}:{port}/video"
    print(f"\n[robot] Iniciando pipeline.")
    print(f"  Stream : {url}")
    print(f"  ESP32  : {ESP32_IP}:{ESP32_PORT} (UDP)")

    model, device = _load_nav_model()

    buf     = FrameBuffer(FRAME_STACK)
    sm      = RobotStateMachine()
    sender  = UDPSender(ip=ESP32_IP, port=ESP32_PORT)
    fps_ctr = FPSCounter()

    stop_evt = threading.Event()
    frame_q  = queue.Queue(maxsize=2)

    cap_thread = threading.Thread(
        target=_capture_thread, args=(url, frame_q, stop_evt), daemon=True)
    cap_thread.start()

    print("[robot] Loop activo. Ctrl+C para detener.\n")
    fps_warn_shown = False
    last_nav_lbl   = "---"
    last_cmd_lbl   = "---"

    try:
        while not stop_evt.is_set():
            try:
                bgr = frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            proc = preprocess_frame(bgr)
            buf.push(proc)

            nav_idx      = _infer(model, buf.get_stack(), device)
            cmd_byte     = sm.update(nav_idx)
            sender.send(cmd_byte)

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
                                fps, sender.wifi_ok)
                cv2.imshow("Robot — q para salir", hud)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        print(f"\n\n[robot] Enviando STOP al ESP32...")
        sender.send(CMD_STOP)
        time.sleep(0.15)
        stop_evt.set()
        sender.close()
        if display:
            cv2.destroyAllWindows()
        print("[robot] Finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline principal del robot")
    parser.add_argument("--ip",      default="192.168.2.1",
                        help="IP del celular con IP Webcam (conectado al hotspot del Mac)")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--display", action="store_true",
                        help="Mostrar ventana de debug con OpenCV")
    args = parser.parse_args()
    run(args.ip, args.port, args.display)

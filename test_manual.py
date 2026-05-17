"""
Control manual del robot desde el teclado (prueba de conexión UDP).

Abre una ventana pequeña. El robot mantiene el último comando hasta que
presiones S explícitamente (modo sticky — no para al soltar la tecla).

Controles:
    W  →  ADELANTE
    A  →  CURVA IZQUIERDA  (diferencial)
    D  →  CURVA DERECHA    (diferencial)
    S  →  STOP
    E  →  GIRO IZQ         (pivote real — izq atrás, der adelante)
    R  →  GIRO DER         (pivote real — izq adelante, der atrás)
    Q  →  salir (envía STOP antes de cerrar)

Uso:
    uv run python test_manual.py
    uv run python test_manual.py --ip 192.168.x.x   # IP del ESP32
"""

import argparse
import socket
import time
import cv2
import numpy as np
from utils import (ESP32_IP, ESP32_PORT,
                   CMD_STOP, CMD_FORWARD, CMD_LEFT, CMD_RIGHT,
                   CMD_GIRO_LEFT, CMD_GIRO_RIGHT)

# ── Mapa de teclas → comandos ─────────────────────────────────────────────────
KEY_MAP = {
    ord('w'): (CMD_FORWARD,    'ADELANTE',  (0, 220, 0)),
    ord('a'): (CMD_LEFT,       'CURVA IZQ', (0, 180, 255)),
    ord('d'): (CMD_RIGHT,      'CURVA DER', (0, 60,  220)),
    ord('s'): (CMD_STOP,       'STOP',      (80, 80, 80)),
    ord('e'): (CMD_GIRO_LEFT,  'GIRO IZQ',  (0, 140, 255)),
    ord('r'): (CMD_GIRO_RIGHT, 'GIRO DER',  (0, 80,  200)),
}

# ── Parámetros de respuesta ──────────────────────────────────────────────────
WAITKEY_MS = 5    # polling de teclado a 200Hz (antes 20ms = 50Hz)
REPEAT_MS  = 30   # resend de comando cada 30ms en modo sticky (antes 80ms)


def send(sock, addr, cmd: int) -> bool:
    try:
        sock.sendto(bytes([cmd]), addr)
        return True
    except Exception as e:
        print(f"[ERROR UDP] {e}")
        return False


def draw_canvas(canvas: np.ndarray, cmd_lbl: str, cmd_color: tuple,
                last_cmd: int, wifi_ok: bool) -> None:
    """Redibuja el canvas con el estado actual. Solo se llama cuando algo cambia."""
    canvas[:] = (20, 20, 20)

    # Título
    cv2.putText(canvas, "CONTROL MANUAL", (90, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

    # Comando activo
    cv2.rectangle(canvas, (20, 55), (380, 115), (40, 40, 40), -1)
    cv2.putText(canvas, cmd_lbl, (30, 103),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, cmd_color, 3)

    # WiFi status
    wlbl = "WiFi: OK" if wifi_ok else "WiFi: ERR"
    wclr = (0, 200, 0) if wifi_ok else (0, 60, 220)
    cv2.putText(canvas, wlbl, (240, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, wclr, 2)

    # Cajas de teclas
    def key_box(x, y, lbl, active=False, giro=False):
        if active:
            clr = (0, 200, 80)
        elif giro:
            clr = (60, 40, 80)   # tono diferente para giros
        else:
            clr = (50, 50, 50)
        cv2.rectangle(canvas, (x, y), (x+52, y+42), clr, -1)
        cv2.rectangle(canvas, (x, y), (x+52, y+42), (120, 120, 120), 1)
        cv2.putText(canvas, lbl, (x+14, y+29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

    # Fila superior: E  W  R
    key_box( 30, 130, 'E', last_cmd == ord('e'), giro=True)   # GIRO IZQ
    key_box(164, 130, 'W', last_cmd == ord('w'))               # ADELANTE
    key_box(298, 130, 'R', last_cmd == ord('r'), giro=True)   # GIRO DER

    # Fila inferior: A  S  D
    key_box(106, 185, 'A', last_cmd == ord('a'))               # CURVA IZQ
    key_box(164, 185, 'S', last_cmd == ord('s'))               # STOP
    key_box(222, 185, 'D', last_cmd == ord('d'))               # CURVA DER

    # Leyenda
    cv2.putText(canvas, "E/R = GIRO PIVOT   Q = salir", (30, 248),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)


def run(ip: str, port: int) -> None:
    addr = (ip, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.0)

    print(f"\n{'='*45}")
    print(f"  CONTROL MANUAL — ESP32 @ {ip}:{port}")
    print(f"{'='*45}")
    print("  W = ADELANTE  |  A = CURVA IZQ  |  D = CURVA DER")
    print("  E = GIRO IZQ  |  R = GIRO DER   |  S = STOP  |  Q = salir")
    print(f"  Polling: {1000//WAITKEY_MS}Hz  |  Resend: {REPEAT_MS}ms")
    print(f"{'='*45}\n")

    canvas = np.zeros((265, 420, 3), dtype=np.uint8)
    cv2.namedWindow("Control Manual — Robot", cv2.WINDOW_AUTOSIZE)

    last_cmd   = -1
    last_send  = 0.0
    cmd_lbl    = "---"
    cmd_color  = (80, 80, 80)
    wifi_ok    = False

    # Estado del último dibujo — para evitar redibujar si nada cambió
    last_drawn = None

    # Dibujo inicial
    draw_canvas(canvas, cmd_lbl, cmd_color, last_cmd, wifi_ok)
    cv2.imshow("Control Manual — Robot", canvas)

    while True:
        # ── Lectura de tecla (polling rápido) ────────────────────────────────
        key = cv2.waitKey(WAITKEY_MS) & 0xFF

        if key == ord('q'):
            break

        now = time.time()
        state_changed = False

        # ── Procesar tecla / sticky mode ─────────────────────────────────────
        if key in KEY_MAP:
            cmd, lbl, color = KEY_MAP[key]
            # Cambio de tecla → enviar INMEDIATAMENTE (sin throttle)
            if key != last_cmd:
                wifi_ok   = send(sock, addr, cmd)
                last_send = now
                last_cmd  = key
                cmd_lbl   = lbl
                cmd_color = color
                state_changed = True
            # Misma tecla → throttle keepalive
            elif (now - last_send) * 1000 > REPEAT_MS:
                wifi_ok   = send(sock, addr, cmd)
                last_send = now
        else:
            # Sin tecla → resend del último comando (sticky)
            if last_cmd != -1 and (now - last_send) * 1000 > REPEAT_MS:
                existing = KEY_MAP.get(last_cmd)
                if existing:
                    wifi_ok   = send(sock, addr, existing[0])
                    last_send = now

        # ── Redibujar SOLO si algo visible cambió ────────────────────────────
        current_draw = (cmd_lbl, last_cmd, wifi_ok)
        if current_draw != last_drawn:
            draw_canvas(canvas, cmd_lbl, cmd_color, last_cmd, wifi_ok)
            cv2.imshow("Control Manual — Robot", canvas)
            last_drawn = current_draw

    # Asegurar STOP al salir
    send(sock, addr, CMD_STOP)
    sock.close()
    cv2.destroyAllWindows()
    print("\n[manual] STOP enviado. Conexión cerrada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control manual del robot por teclado")
    parser.add_argument("--ip",   default=ESP32_IP,   help="IP del ESP32")
    parser.add_argument("--port", default=ESP32_PORT, type=int, help="Puerto UDP")
    args = parser.parse_args()
    run(args.ip, args.port)

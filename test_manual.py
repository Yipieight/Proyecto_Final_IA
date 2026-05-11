"""
Control manual del robot desde el teclado (prueba de conexión UDP).

Abre una ventana pequeña — mantén presionada la tecla para mover el robot.
Al soltar la tecla se envía STOP automáticamente.

Controles:
    W  →  ADELANTE
    A  →  IZQUIERDA
    D  →  DERECHA
    S  →  STOP manual
    Q  →  salir

Uso:
    uv run python test_manual.py
    uv run python test_manual.py --ip 192.168.x.x   # IP del ESP32
"""

import argparse
import socket
import time
import cv2
import numpy as np
from utils import ESP32_IP, ESP32_PORT, CMD_STOP, CMD_FORWARD, CMD_LEFT, CMD_RIGHT

# ── Mapa de teclas → comandos ─────────────────────────────────────────────────
KEY_MAP = {
    ord('w'): (CMD_FORWARD, 'ADELANTE',   (0, 220, 0)),
    ord('a'): (CMD_LEFT,    'IZQ',        (0, 180, 255)),
    ord('d'): (CMD_RIGHT,   'DER',        (0, 60,  220)),
    ord('s'): (CMD_STOP,    'STOP',       (80, 80, 80)),
}


def send(sock, addr, cmd: int) -> bool:
    try:
        sock.sendto(bytes([cmd]), addr)
        return True
    except Exception as e:
        print(f"[ERROR UDP] {e}")
        return False


def run(ip: str, port: int) -> None:
    addr = (ip, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.0)

    print(f"\n{'='*45}")
    print(f"  CONTROL MANUAL — ESP32 @ {ip}:{port}")
    print(f"{'='*45}")
    print("  W = ADELANTE  |  A = IZQ  |  D = DER")
    print("  S = STOP      |  Q = salir")
    print("  Mantén presionada la tecla para mover")
    print(f"{'='*45}\n")

    # Ventana de visualización
    canvas = np.zeros((260, 380, 3), dtype=np.uint8)
    cv2.namedWindow("Control Manual — Robot", cv2.WINDOW_AUTOSIZE)

    last_cmd   = -1
    last_send  = 0.0
    REPEAT_MS  = 80   # reenviar el comando cada 80ms mientras se mantiene la tecla

    cmd_lbl   = "---"
    cmd_color = (80, 80, 80)
    wifi_ok   = False

    while True:
        # ── Dibujar interfaz ──────────────────────────────────────────────────
        canvas[:] = (20, 20, 20)

        # Título
        cv2.putText(canvas, "CONTROL MANUAL", (90, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        # Comando activo
        cv2.rectangle(canvas, (20, 55), (360, 115), (40, 40, 40), -1)
        cv2.putText(canvas, cmd_lbl, (30, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, cmd_color, 3)

        # WiFi status
        wlbl  = "WiFi: OK" if wifi_ok else "WiFi: ERR"
        wclr  = (0, 200, 0) if wifi_ok else (0, 60, 220)
        cv2.putText(canvas, wlbl, (230, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, wclr, 2)

        # Teclas guía — W
        def key_box(x, y, lbl, active=False):
            clr = (0, 180, 80) if active else (60, 60, 60)
            cv2.rectangle(canvas, (x, y), (x+52, y+42), clr, -1)
            cv2.rectangle(canvas, (x, y), (x+52, y+42), (120,120,120), 1)
            cv2.putText(canvas, lbl, (x+14, y+29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)

        key_box(164, 130, 'W', last_cmd == CMD_FORWARD)
        key_box(106, 185, 'A', last_cmd == CMD_LEFT)
        key_box(164, 185, 'S', last_cmd == CMD_STOP)
        key_box(222, 185, 'D', last_cmd == CMD_RIGHT)

        cv2.putText(canvas, "Q = salir",
                    (130, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

        cv2.imshow("Control Manual — Robot", canvas)

        # ── Lectura de tecla ──────────────────────────────────────────────────
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break

        now = time.time()
        if key in KEY_MAP:
            cmd, lbl, color = KEY_MAP[key]
            # Enviar inmediatamente al pulsar, o repetir si se mantiene
            if key != last_cmd or (now - last_send) * 1000 > REPEAT_MS:
                wifi_ok   = send(sock, addr, cmd)
                last_send = now
                last_cmd  = key
                cmd_lbl   = lbl
                cmd_color = color
        else:
            # Ninguna tecla de movimiento → STOP
            if last_cmd not in (ord('s'), -1):
                wifi_ok  = send(sock, addr, CMD_STOP)
                last_cmd = ord('s')
                cmd_lbl  = "STOP"
                cmd_color = (80, 80, 80)

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

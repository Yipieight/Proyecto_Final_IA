"""
Constantes globales del módulo de control por voz.
"""

# ── ESP32 — comunicación UDP ──────────────────────────────────────────────────
# Actualizar ESP32_IP con la IP que aparece en el Serial Monitor al arrancar.
ESP32_IP   = "192.168.1.4"
ESP32_PORT = 9999

# ── Protocolo UDP (1 byte por comando) ────────────────────────────────────────
CMD_STOP       = 0x00
CMD_FORWARD    = 0x01
CMD_LEFT       = 0x02
CMD_RIGHT      = 0x03
CMD_GIRO_LEFT  = 0x04
CMD_GIRO_RIGHT = 0x05

CMD_NAME = {
    CMD_STOP:       "STOP",
    CMD_FORWARD:    "ADELANTE",
    CMD_LEFT:       "IZQUIERDA",
    CMD_RIGHT:      "DERECHA",
    CMD_GIRO_LEFT:  "GIRO_IZQ",
    CMD_GIRO_RIGHT: "GIRO_DER",
}

"""
Máquina de estados del robot — solo navegación.

Estados:
  NAVIGATING    — seguimiento normal de la línea
  STOPPED_GIRO  — detenido 1 segundo antes de pivote de 90°
  TURNING_GIRO  — pivotando 1 segundo para completar el giro de 90°
  STOPPED_T     — detenido 2 segundos en cruce T
  CHOOSING_T    — ejecutando la dirección elegida aleatoriamente

Diagrama:
  NAVIGATING ──(GIRO_90_*)──► STOPPED_GIRO ──(1 s)──► TURNING_GIRO ──(1 s)──► NAVIGATING
  NAVIGATING ──(CRUCE_T)─────► STOPPED_T ───(2 s)──► CHOOSING_T  ──(0.8 s)──► NAVIGATING

CURVA_IZQ/CURVA_DER giran continuamente sin parar (curvas suaves).
GIRO_90_*/CRUCE_T paran primero y luego pivotan (giros bruscos).
"""

import time
import random
from utils import (
    NAV_IDX_CLASS,
    CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP,
    CMD_GIRO_LEFT, CMD_GIRO_RIGHT,
)

# ── Constantes de tiempo ───────────────────────────────────────────────────────
T_CRUCE_STOP = 2.0   # segundos detenido en cruce T   (R9: exactamente 2 s)
T_CRUCE_TURN = 0.8   # segundos ejecutando el giro aleatorio
T_GIRO_STOP  = 1.0   # segundos detenido antes del giro de 90°
T_GIRO_TURN  = 1.0   # segundos pivotando para completar el giro de 90°

# ── Estados ────────────────────────────────────────────────────────────────────
S_NAVIGATING   = "NAVIGATING"
S_STOPPED_T    = "STOPPED_T"
S_CHOOSING_T   = "CHOOSING_T"
S_STOPPED_GIRO = "STOPPED_GIRO"
S_TURNING_GIRO = "TURNING_GIRO"


class RobotStateMachine:
    def __init__(self):
        self.state     = S_NAVIGATING
        self._t_start  = time.time()
        self._t_choice = None   # CMD_LEFT o CMD_RIGHT elegido en cruce T
        self._giro_dir = None   # CMD_LEFT o CMD_RIGHT pendiente para giro 90°

    # ── API pública ────────────────────────────────────────────────────────────

    def update(self, nav_idx: int) -> int:
        """
        Actualiza el estado y devuelve el byte de comando UDP correspondiente.

        Args:
            nav_idx : índice de clase del modelo de navegación (0-5)

        Returns:
            Uno de: CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP  (bytes 0x01-0x03, 0x00)
        """
        nav = NAV_IDX_CLASS.get(nav_idx, 'RECTA')

        # ── Estado: DETENIDO antes de giro de 90° ─────────────────────────────
        if self.state == S_STOPPED_GIRO:
            if self._elapsed() >= T_GIRO_STOP:
                self._transition(S_TURNING_GIRO)
                return self._giro_dir
            return CMD_STOP

        # ── Estado: PIVOTANDO 90° ─────────────────────────────────────────────
        if self.state == S_TURNING_GIRO:
            if self._elapsed() >= T_GIRO_TURN:
                self._transition(S_NAVIGATING)
                # caer en navegación normal
            else:
                return self._giro_dir

        # ── Estado: DETENIDO en cruce T ───────────────────────────────────────
        if self.state == S_STOPPED_T:
            if self._elapsed() >= T_CRUCE_STOP:
                self._t_choice = random.choice([CMD_GIRO_LEFT, CMD_GIRO_RIGHT])
                self._transition(S_CHOOSING_T)
                return self._t_choice
            return CMD_STOP

        # ── Estado: ELIGIENDO dirección en cruce T ────────────────────────────
        if self.state == S_CHOOSING_T:
            if self._elapsed() >= T_CRUCE_TURN:
                self._transition(S_NAVIGATING)
                # caer en navegación normal
            else:
                return self._t_choice

        # ── Navegación normal ─────────────────────────────────────────────────
        return self._navigate(nav)

    def get_state(self) -> str:
        return self.state

    # ── Internos ──────────────────────────────────────────────────────────────

    def _elapsed(self) -> float:
        return time.time() - self._t_start

    def _transition(self, new_state: str) -> None:
        self.state    = new_state
        self._t_start = time.time()

    def _navigate(self, nav_class: str) -> int:
        if nav_class == 'RECTA':
            return CMD_FORWARD
        # Curvas suaves: giro continuo en movimiento
        if nav_class == 'CURVA_IZQ':
            return CMD_LEFT
        if nav_class == 'CURVA_DER':
            return CMD_RIGHT
        # Giros bruscos de 90°: parar primero, luego pivotar (un solo lado activo)
        if nav_class == 'GIRO_90_IZQ':
            self._giro_dir = CMD_GIRO_LEFT
            self._transition(S_STOPPED_GIRO)
            return CMD_STOP
        if nav_class == 'GIRO_90_DER':
            self._giro_dir = CMD_GIRO_RIGHT
            self._transition(S_STOPPED_GIRO)
            return CMD_STOP
        if nav_class == 'CRUCE_T':
            self._transition(S_STOPPED_T)
            return CMD_STOP
        return CMD_FORWARD

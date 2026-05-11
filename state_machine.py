"""
Máquina de estados del robot — solo navegación.

Estados:
  NAVIGATING  — seguimiento normal de la línea
  STOPPED_T   — detenido 2 segundos en cruce T
  CHOOSING_T  — ejecutando la dirección elegida aleatoriamente

Diagrama:
  NAVIGATING ──(CRUCE_T)──► STOPPED_T ──(2 s)──► CHOOSING_T ──(0.8 s)──► NAVIGATING
"""

import time
import random
from utils import (
    NAV_IDX_CLASS,
    CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP,
)

# ── Constantes de tiempo ───────────────────────────────────────────────────────
T_CRUCE_STOP = 2.0   # segundos detenido en cruce T   (R9: exactamente 2 s)
T_CRUCE_TURN = 0.8   # segundos ejecutando el giro aleatorio

# ── Estados ────────────────────────────────────────────────────────────────────
S_NAVIGATING = "NAVIGATING"
S_STOPPED_T  = "STOPPED_T"
S_CHOOSING_T = "CHOOSING_T"


class RobotStateMachine:
    def __init__(self):
        self.state     = S_NAVIGATING
        self._t_start  = time.time()
        self._t_choice = None   # CMD_LEFT o CMD_RIGHT elegido en cruce T

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

        # ── Estado: DETENIDO en cruce T ───────────────────────────────────────
        if self.state == S_STOPPED_T:
            if self._elapsed() >= T_CRUCE_STOP:
                self._t_choice = random.choice([CMD_LEFT, CMD_RIGHT])
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
        if nav_class in ('CURVA_IZQ', 'GIRO_90_IZQ'):
            return CMD_LEFT
        if nav_class in ('CURVA_DER', 'GIRO_90_DER'):
            return CMD_RIGHT
        if nav_class == 'CRUCE_T':
            if self.state not in (S_STOPPED_T, S_CHOOSING_T):
                self._transition(S_STOPPED_T)
            return CMD_STOP
        return CMD_FORWARD

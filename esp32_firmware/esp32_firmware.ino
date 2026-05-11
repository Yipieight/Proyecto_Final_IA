/*
  esp32_firmware.ino — Controlador de motores para el robot autónomo

  Hardware:
    - ESP32 (cualquier placa compatible)
    - Puente H L298N
        Motor A (izquierdo): IN1=GPIO26, IN2=GPIO27, ENA=GPIO14  (PWM)
        Motor B (derecho):   IN3=GPIO25, IN4=GPIO33, ENB=GPIO32  (PWM)

  Comunicación:
    - Se conecta al hotspot WiFi del Mac
    - Listener UDP en puerto 4210
    - Recibe datagramas de 1 byte del script Python (main_robot.py)

  Protocolo (1 byte por datagrama):
    0x00  STOP      — detener motores
    0x01  ADELANTE  — avanzar recto (RECTA)
    0x02  IZQUIERDA — girar izq. (CURVA_IZQ, GIRO_90_IZQ)
    0x03  DERECHA   — girar der. (CURVA_DER, GIRO_90_DER)
    0x04  T_GIRO    — giro post-pausa en CRUCE_T (igual a IZQUIERDA/DERECHA)

  IMPORTANTE: Actualizar SSID y PASSWORD antes de flashear.
  Al iniciar imprime la IP asignada en el Serial Monitor (115200 baud).
  Copiar esa IP en ESP32_IP de utils.py del proyecto Python.
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ── Credenciales WiFi (hotspot del Mac) ───────────────────────────────────────
const char* SSID     = "NombreDelHotspot";     // cambiar
const char* PASSWORD = "ContrasenaDelHotspot"; // cambiar

// ── Puerto UDP ────────────────────────────────────────────────────────────────
#define UDP_PORT  4210

// ── Pines del L298N ───────────────────────────────────────────────────────────
#define IN1  26   // Motor A — sentido
#define IN2  27
#define ENA  14   // Motor A — velocidad PWM

#define IN3  25   // Motor B — sentido
#define IN4  33
#define ENB  32   // Motor B — velocidad PWM

// ── Canales PWM (LEDC del ESP32) ──────────────────────────────────────────────
#define CH_A  0
#define CH_B  1
#define PWM_FREQ  1000   // Hz
#define PWM_RES   8      // bits (0-255)

// ── Niveles de velocidad ──────────────────────────────────────────────────────
#define VEL_NORMAL  180
#define VEL_GIRO    155
#define VEL_STOP    0

// ── Bytes de comando ──────────────────────────────────────────────────────────
#define CMD_STOP     0x00
#define CMD_FORWARD  0x01
#define CMD_LEFT     0x02
#define CMD_RIGHT    0x03
#define CMD_T_CROSS  0x04

// ── Estado global ─────────────────────────────────────────────────────────────
WiFiUdp udp;
uint8_t          ultimoByte      = CMD_STOP;
unsigned long    ultimoMensaje   = 0;
const unsigned long WATCHDOG_MS  = 500;   // frenar si no llega paquete en 500 ms

// ── Control de motores ────────────────────────────────────────────────────────

void motorStop() {
  ledcWrite(CH_A, VEL_STOP);
  ledcWrite(CH_B, VEL_STOP);
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void motorForward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);   // Motor A adelante
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);   // Motor B adelante
  ledcWrite(CH_A, VEL_NORMAL);
  ledcWrite(CH_B, VEL_NORMAL);
}

void motorLeft() {
  // Giro izquierda: motor A atrás, motor B adelante
  digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(CH_A, VEL_GIRO);
  ledcWrite(CH_B, VEL_GIRO);
}

void motorRight() {
  // Giro derecha: motor A adelante, motor B atrás
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH);
  ledcWrite(CH_A, VEL_GIRO);
  ledcWrite(CH_B, VEL_GIRO);
}

void ejecutarComando(uint8_t cmd) {
  switch (cmd) {
    case CMD_STOP:    motorStop();    break;
    case CMD_FORWARD: motorForward(); break;
    case CMD_LEFT:    motorLeft();    break;
    case CMD_RIGHT:   motorRight();   break;
    case CMD_T_CROSS: motorLeft();    break;  // T_GIRO — usa izquierda; la SM
                                              // ya elige aleatoriamente en Python
    default:
      // Byte desconocido — ignorar, no cambiar estado de motores
      Serial.printf("[WARN] Byte desconocido: 0x%02X\n", cmd);
      return;
  }
  ultimoByte    = cmd;
  ultimoMensaje = millis();
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(200);

  // Configurar pines de dirección
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Configurar PWM (LEDC)
  ledcSetup(CH_A, PWM_FREQ, PWM_RES);
  ledcSetup(CH_B, PWM_FREQ, PWM_RES);
  ledcAttachPin(ENA, CH_A);
  ledcAttachPin(ENB, CH_B);

  motorStop();   // estado seguro inicial

  // Conectar al hotspot del Mac
  Serial.printf("\n[ESP32] Conectando a WiFi: %s\n", SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);

  int intentos = 0;
  while (WiFi.status() != WL_CONNECTED && intentos < 30) {
    delay(500);
    Serial.print(".");
    intentos++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[ERROR] No se pudo conectar al WiFi.");
    Serial.println("  Verifica SSID y PASSWORD y reflashea.");
    while (true) { delay(2000); }   // halt
  }

  Serial.println("\n[ESP32] WiFi conectado.");
  Serial.printf("[ESP32] IP del robot : %s\n", WiFi.localIP().toString().c_str());
  Serial.println("[ESP32] >> Copia esa IP en ESP32_IP de utils.py <<");

  // Iniciar listener UDP
  udp.begin(UDP_PORT);
  Serial.printf("[ESP32] Escuchando UDP en puerto %d\n", UDP_PORT);
  Serial.println("[ESP32] Esperando comandos...\n");

  ultimoMensaje = millis();
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
  // Leer datagrama UDP si está disponible
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    uint8_t buf[4];
    int len = udp.read(buf, sizeof(buf));
    if (len > 0) {
      ejecutarComando(buf[0]);
    }
  }

  // Watchdog: si no llega ningún paquete en WATCHDOG_MS, frenar por seguridad
  if (ultimoByte != CMD_STOP &&
      millis() - ultimoMensaje > WATCHDOG_MS) {
    motorStop();
    ultimoByte = CMD_STOP;
    Serial.println("[WARN] Watchdog — sin paquetes recientes, robot detenido.");
  }
}

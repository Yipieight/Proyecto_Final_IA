#include <WiFi.h>
#include <WiFiUDP.h>

// ── Configuración WiFi ──────────────────────────────
// const char* SSID     = "iphone de Josecito";
// const char* PASS     = "12345678";

const char* SSID     = "CLARO1_84E9EC";
const char* PASS     = "303K0WIEXC";

// const char* SSID     = "Ruan";
// const char* PASS     = "Jl_7042mk";

const int UDP_PORT = 9999;

// ── Pines L298N ─────────────────────────────────────
// ENA y ENB tienen jumper caps a 5V — NO se controlan desde GPIO.
// La velocidad se regula con PWM en los pines IN (analogWrite).
#define IN1 26   // Motor izquierdo (OUT1 / OUT2)
#define IN2 27
#define IN3 13   // Motor derecho   (OUT3 / OUT4)
#define IN4 12

WiFiUDP udp;

// ── Control de motores ───────────────────────────────
// dir : 1 = adelante | -1 = atrás | 0 = stop
// vel : 0-255  (PWM aplicado en los pines IN)
//
// FIX #1: derDir=1 usa IN3=PWM, IN4=0   (antes estaba invertido → derecho iba atrás)
// FIX #2: velocidad vía analogWrite en IN, NO en ENA (ENA está fija a 5V por jumper)
// FIX #3: stop usa analogWrite(pin,0) para cancelar el PWM limpiamente
void setMotores(int izqDir, int derDir, int velIzq, int velDer) {

  // Motor izquierdo (OUT1 / OUT2)
  if      (izqDir ==  1) { analogWrite(IN1, velIzq); analogWrite(IN2, 0);      }
  else if (izqDir == -1) { analogWrite(IN1, 0);      analogWrite(IN2, velIzq); }
  else                   { analogWrite(IN1, 0);      analogWrite(IN2, 0);      }

  // Motor derecho (OUT3 / OUT4) — polaridad física corregida
  if      (derDir ==  1) { analogWrite(IN3, velDer); analogWrite(IN4, 0);      }
  else if (derDir == -1) { analogWrite(IN3, 0);      analogWrite(IN4, velDer); }
  else                   { analogWrite(IN3, 0);      analogWrite(IN4, 0);      }
}

void setup() {
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Arrancar con todos los pines en LOW (STOP seguro)
  analogWrite(IN1, 0); analogWrite(IN2, 0);
  analogWrite(IN3, 0); analogWrite(IN4, 0);

  Serial.begin(115200);
  Serial.println("\nConectando a WiFi...");

  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado!");
  Serial.print("IP del ESP32: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
  Serial.println("Esperando comandos UDP en puerto 9999...");
}

void loop() {
  int sz = udp.parsePacket();
  if (sz > 0) {
    uint8_t cmd;
    udp.read(&cmd, 1);

    Serial.print("CMD 0x"); Serial.println(cmd, HEX);

    switch (cmd) {
      case 0x00:  // STOP
        setMotores(0, 0, 0, 0);
        Serial.println("→ STOP");
        break;

      case 0x01:  // ADELANTE
        setMotores(1, 1, 220, 220);
        Serial.println("→ ADELANTE");
        break;

      case 0x02:  // CURVA IZQUIERDA (derecho empuja, izquierdo frena)
        setMotores(1, 1, 0, 255);
        Serial.println("→ CURVA IZQ");
        break;

      case 0x03:  // CURVA DERECHA (izquierdo empuja, derecho frena)
        setMotores(1, 1, 255, 0);
        Serial.println("→ CURVA DER");
        break;

      case 0x04:  // GIRO 90° IZQUIERDA — pivote real (izq atrás, der adelante)
        setMotores(-1, 1, 255, 255);
        Serial.println("→ GIRO IZQ");
        break;

      case 0x05:  // GIRO 90° DERECHA — pivote real (izq adelante, der atrás)
        setMotores(1, -1, 255, 255);
        Serial.println("→ GIRO DER");
        break;
    }
  }
}

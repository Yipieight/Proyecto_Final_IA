#include <WiFi.h>
#include <WiFiUDP.h>

// ── Configuración WiFi ──────────────────────────────
// const char* SSID     = "iphone de Josecito";
// const char* PASS     = "12345678";
// const int   UDP_PORT = 9999;

const char* SSID     = "CLARO1_84E9EC";
const char* PASS     = "303K0WIEXC";
const int   UDP_PORT = 9999;

// const char* SSID     = "Ruan";
// const char* PASS     = "Jl_7042mk";
// const int   UDP_PORT = 9999;

// ── Pines L298N ─────────────────────────────────────
#define ENA 25
#define IN1 26
#define IN2 27
#define ENB 14
#define IN3 13
#define IN4 12

WiFiUDP udp;

// Acepta velocidades INDIVIDUALES para cada lado
void setMotores(int izqDir, int derDir, int velIzq, int velDer) {
  // Motor izquierdo (OUT1 / OUT2)
  if (izqDir == 1)       { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  }
  else if (izqDir == -1) { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); }
  else                   { digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);  }

  // Motor derecho (OUT3 / OUT4) — polaridad física invertida respecto al izq
  if (derDir == 1)       { digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH); }
  else if (derDir == -1) { digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);  }
  else                   { digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);  }

  ledcWrite(0, velIzq);
  ledcWrite(1, velDer);
}

void setup() {
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  ledcSetup(0, 5000, 8); ledcAttachPin(ENA, 0);
  ledcSetup(1, 5000, 8); ledcAttachPin(ENB, 1);

  Serial.begin(115200);
  Serial.println("Conectando a WiFi...");

  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi conectado!");
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

    Serial.print("Comando recibido: 0x0");
    Serial.println(cmd, HEX);

    switch (cmd) {
      case 0x00:  // STOP
        setMotores(0, 0, 0, 0);
        Serial.println("PARAR");
        break;
      case 0x01:  // ADELANTE
        setMotores(1, 1, 10, 10);
        Serial.println("ADELANTE");
        break;
      case 0x02:  // CURVA IZQUIERDA suave (diferencial)
        setMotores(1, 1, 5, 20);
        Serial.println("CURVA IZQ");
        break;
      case 0x03:  // CURVA DERECHA suave (diferencial)
        setMotores(1, 1, 20, 5);
        Serial.println("CURVA DER");
        break;
      case 0x04:  // GIRO 90° IZQUIERDA (solo lado derecho activo)
        setMotores(0, 1, 0, 255);
        Serial.println("GIRO IZQ");
        break;
      case 0x05:  // GIRO 90° DERECHA (solo lado izquierdo activo)
        setMotores(1, 0, 255, 0);
        Serial.println("GIRO DER");
        break;
    }
  }
}

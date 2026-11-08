/*
 * Rua Acesa — Robô de ronda (Maker Lab · Global Solution FIAP 2026.2 · CitizenTech)
 *
 * O robô faz a ronda por um trecho de rua à noite: lê a luminosidade (LDR), classifica a
 * iluminação com um modelo TinyML (árvore de decisão embarcada) e, ao confirmar um ponto
 * escuro/fraco/instável, alerta e envia a ocorrência para a API central do Rua Acesa.
 *
 * Sensores:  LDR (luz) · HC-SR04 (obstáculo, sensor adicional) · DHT (temperatura/umidade)
 * Tudo roda sozinho dentro do loop(): não há botões. Ao ligar, a ronda já começa.
 *
 * Sem bibliotecas externas: o DHT é lido por um driver próprio (DHT_USAR_BIBLIOTECA 0).
 * Se preferir a biblioteca da Adafruit, ponha DHT_USAR_BIBLIOTECA 1 e crie a aba
 * libraries.txt com as linhas "DHT sensor library" e "Adafruit Unified Sensor".
 */
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
#include <math.h>

// ======================= CONFIGURAÇÃO (edite aqui) =======================
// ---- Rede (no Wokwi, a rede aberta "Wokwi-GUEST" dá acesso à internet) ----
#define WIFI_SSID     "Wokwi-GUEST"
#define WIFI_PASSWORD ""
#define WIFI_CHANNEL  6

// ---- API central do Rua Acesa ----
// TROQUE pela URL do endpoint que recebe relatos do robô (a API central do Rua Acesa).
// Para testar ANTES da API estar no ar: crie um endereço em https://webhook.site
// e cole aqui; os POSTs aparecem lá em tempo real (ótimo para o vídeo).
// NÃO coloque chaves reais aqui se for publicar no GitHub (o pipeline roda Gitleaks).
#define API_URL        "https://webhook.site/COLE-SEU-ID-AQUI"
#define DEVICE_ID      "rua-acesa-robo-01"
#define DEVICE_KEY     "dev-key-demo"   // placeholder: NÃO é segredo real

// ---- Fonte da luz (sem botões: troque aqui e rode de novo) ----
//   0 = sensor LDR (no Wokwi, use o slider do módulo)   1 = ADEQUADO   2 = FRACO
//   3 = ESCURO      4 = INSTAVEL (luz piscando)
#define FONTE_LUZ      0
// 1 = percorre as 5 fontes sozinho, DEMO_FASE_MS em cada (ideal para gravar o vídeo)
#define DEMO_AUTO      0
#define DEMO_FASE_MS   45000

// ---- Ronda ----
#define SAMPLE_INTERVAL_MS   300   // intervalo entre leituras do LDR
#define DIST_INTERVAL_MS     100   // intervalo entre medidas do ultrassônico (mín. 60 ms)
#define OBSTACLE_CM          25    // abaixo disso há obstáculo à frente
#define CONFIRM_WINDOWS      3     // janelas consecutivas "ruins" para confirmar ocorrência
#define REPORT_COOLDOWN_MS   15000 // não reenviar a MESMA classe de ocorrência antes disso
#define RED_LED_MS           3000  // quanto tempo o LED vermelho fica aceso após reportar

// ---- DHT (temperatura e umidade do ponto) ----
#define DHT_TIPO             22    // 22 = DHT22 (peça do Wokwi) · 11 = DHT11 (robô real)
#define DHT_USAR_BIBLIOTECA  0     // 0 = leitura própria, sem biblioteca · 1 = biblioteca Adafruit
#define DHT_INTERVAL_MS      3000
#define UMIDADE_ALTA_PCT     85    // acima disso o relato leva contexto "umidade_alta"
#if DHT_USAR_BIBLIOTECA
  #include <DHT.h>
#endif

// ---- Posição simulada (o robô não tem GPS) ----
// Trecho de teste ao longo da Av. Paulista, SP (coordenadas aproximadas, SIMULADAS).
// A cada STEP_EVERY_MS a ronda "avança" para o próximo ponto.
static const float ROUTE[][2] = {
  {-23.56450f, -46.65270f}, {-23.56480f, -46.65210f}, {-23.56510f, -46.65150f},
  {-23.56540f, -46.65090f}, {-23.56570f, -46.65030f}, {-23.56600f, -46.64970f},
  {-23.56630f, -46.64910f}, {-23.56660f, -46.64850f}
};
#define ROUTE_LEN (sizeof(ROUTE) / sizeof(ROUTE[0]))
#define STEP_EVERY_MS 2000

// ================= ATRIBUTOS DA JANELA DO LDR =================
// C++ puro: é compilado igual no ESP32 e no teste de paridade em PC.
// Precisa ser IDÊNTICO ao cálculo de ml/train_classifier.py (função extract_features).

#define WINDOW_SIZE 10
#define LUX_MIN 0.1f
#define LUX_MAX 100000.0f

// Atributos usados pelo classificador (ordem fixa!)
//   [0] lmean = log10(média da janela em lux)
//   [1] lmin  = log10(menor leitura da janela)
//   [2] lmax  = log10(maior leitura da janela)
//   [3] cv    = desvio padrão / média (coeficiente de variação)
#define N_FEATURES 4

static inline float clampLux(float v) {
  if (!(v > LUX_MIN)) return LUX_MIN;  // também trata NaN
  if (v > LUX_MAX) return LUX_MAX;
  return v;
}

static inline void extractFeatures(const float *lux, int n, float *out) {
  float sum = 0, mn = LUX_MAX, mx = LUX_MIN;
  for (int i = 0; i < n; i++) {
    float v = clampLux(lux[i]);
    sum += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  float mean = sum / n;
  float var = 0;
  for (int i = 0; i < n; i++) {
    float d = clampLux(lux[i]) - mean;
    var += d * d;
  }
  float sd = sqrtf(var / n);
  out[0] = log10f(mean);
  out[1] = log10f(mn);
  out[2] = log10f(mx);
  out[3] = sd / mean;
}

// ================= CLASSIFICADOR TinyML (gerado por ml/train_classifier.py) =================
#define CLASS_ADEQUADO 0
#define CLASS_FRACO    1
#define CLASS_ESCURO   2
#define CLASS_INSTAVEL 3

static const char *CLASS_NAMES[4] = {"ADEQUADO", "FRACO", "ESCURO", "INSTAVEL"};

// f[] = {lmean, lmin, lmax, cv} calculado por extractFeatures()
static inline int classifyWindow(const float *f) {
  if (f[0] <= 0.669724f) {
    if (f[0] <= 0.641033f) {
      return 2;  // ESCURO
    } else {
      if (f[2] <= 0.779144f) {
        return 2;  // ESCURO
      } else {
        return 2;  // ESCURO
      }
    }
  } else {
    if (f[3] <= 0.389160f) {
      if (f[0] <= 1.312255f) {
        if (f[0] <= 0.704590f) {
          if (f[2] <= 0.774096f) {
            return 2;  // ESCURO
          } else {
            return 1;  // FRACO
          }
        } else {
          if (f[0] <= 1.297120f) {
            return 1;  // FRACO
          } else {
            return 1;  // FRACO
          }
        }
      } else {
        if (f[1] <= 1.115762f) {
          return 3;  // INSTAVEL
        } else {
          if (f[0] <= 1.341214f) {
            return 0;  // ADEQUADO
          } else {
            return 0;  // ADEQUADO
          }
        }
      }
    } else {
      if (f[3] <= 0.459886f) {
        return 3;  // INSTAVEL
      } else {
        return 3;  // INSTAVEL
      }
    }
  }
}

// ============================ PROGRAMA ============================

// ------------------------------- Pinagem -------------------------------
#define PIN_LDR_AO   34   // ADC1 (ok com Wi-Fi ligado)
#define PIN_TRIG     19
#define PIN_ECHO     18   // HC-SR04 ECHO é 5 V: no robô real use divisor de tensão
#define PIN_DHT      32
#define PIN_LED_G    25
#define PIN_LED_Y    26
#define PIN_LED_R    27
#define PIN_BUZZER   23

// --------------------------- Estado do robô ---------------------------
static const char *SRC_NAMES[] = {"SENSOR(LDR)", "ADEQUADO", "FRACO", "ESCURO", "INSTAVEL"};

#if DHT_USAR_BIBLIOTECA
static DHT dht(PIN_DHT, DHT_TIPO);
#endif
static float tempC = NAN, umidPct = NAN;       // última leitura válida do DHT

static float window[WINDOW_SIZE];
static int winCount = 0;
static int badStreak = 0;                      // janelas consecutivas diferentes de ADEQUADO
static bool obstaculoAtivo = false;
static float distCm = 999.0f;
static int lastSrc = -1;
static int routeIdx = 0;
static int lastReportClass = -1;               // classe da última ocorrência enviada
static uint32_t eventSeq = 0;
static unsigned long lastSampleMs = 0, lastStepMs = 0, lastReportMs = 0;
static unsigned long lastDistMs = 0, lastDhtMs = 0, lastWifiTryMs = 0, redUntilMs = 0;

// Fila de ocorrências (tolerância a falha de rede)
#define QUEUE_SIZE 8
static String queueBuf[QUEUE_SIZE];
static int qHead = 0, qCount = 0;

// ----------------------------- Utilidades -----------------------------
static void setLeds(bool g, bool y, bool r) {
  digitalWrite(PIN_LED_G, g); digitalWrite(PIN_LED_Y, y); digitalWrite(PIN_LED_R, r);
}

static void beep(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_BUZZER, HIGH); delay(onMs);
    digitalWrite(PIN_BUZZER, LOW);  delay(offMs);
  }
}

static String isoTime() {
  struct tm t;
  if (getLocalTime(&t, 50)) {
    char b[25]; strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &t); return String(b);
  }
  return String("1970-01-01T00:00:00Z");  // NTP ainda não sincronizou
}

// ------------------------------ Sensores ------------------------------
// Conversão tensão→lux usando a fórmula da documentação do sensor de luz do Wokwi.
// No LDR real o valor é APROXIMADO: calibre com um luxímetro (app de celular serve p/ demo).
static float readLuxRaw() {
  int a = analogRead(PIN_LDR_AO);
  if (a < 1) a = 1;
  if (a > 4094) a = 4094;
  const float GAMMA = 0.7f, RL10 = 50.0f;
  float voltage = a / 4095.0f * 3.3f;
  float resistance = 2000.0f * voltage / (1.0f - voltage / 3.3f);
  float lux = powf(RL10 * 1e3f * powf(10.0f, GAMMA) / resistance, 1.0f / GAMMA);
  return clampLux(lux);
}

// Fonte da luz em uso (fixa pelo #define, ou ciclando se DEMO_AUTO = 1)
static int fonteAtual() {
#if DEMO_AUTO
  return (int)((millis() / DEMO_FASE_MS) % 5);
#else
  return FONTE_LUZ;
#endif
}

// Perfis injetados (modo de teste). Ruído leve para parecer sensor real.
static float noise(float pct) { return 1.0f + pct * ((random(-1000, 1001)) / 1000.0f); }
static float readLux(int src) {
  switch (src) {
    case 1: return clampLux(120.0f * noise(0.06f));   // ADEQUADO
    case 2: return clampLux(11.0f * noise(0.08f));    // FRACO
    case 3: return clampLux(1.5f * noise(0.20f));     // ESCURO
    case 4: return clampLux(random(0, 2) ? 150.0f * noise(0.05f) : 0.8f * noise(0.2f));  // INSTAVEL
    default: {  // mediana de 3 amostras: filtra ruído impulsivo do ADC
      float a = readLuxRaw(), b = readLuxRaw(), c = readLuxRaw();
      if (a > b) { float t = a; a = b; b = t; }
      if (b > c) { float t = b; b = c; c = t; }
      if (a > b) { float t = a; a = b; b = t; }
      return b;
    }
  }
}

static float readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW); delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, 30000UL);  // timeout 30 ms (~5 m)
  if (us == 0) return 999.0f;                            // sem eco = livre
  return us * 0.0343f / 2.0f;
}

// ---- Leitura do DHT sem biblioteca (protocolo de 1 fio, 40 bits) ----
#if !DHT_USAR_BIBLIOTECA
// Mede quanto tempo a linha fica em um nível; devolve DHT_TIMEOUT se passar do limite.
// Limite largo (1 ms, como a biblioteca da Adafruit): o simulador pode ser mais lento que o chip.
#define DHT_TIMEOUT    0xFFFFFFFFUL
#define DHT_LIMITE_US  1000
static uint32_t dhtPulso(int nivel) {
  uint32_t t0 = micros();
  while (digitalRead(PIN_DHT) == nivel) { if (micros() - t0 > DHT_LIMITE_US) return DHT_TIMEOUT; }
  return micros() - t0;
}
static bool dhtEsperar(int nivel) {            // espera a linha chegar ao nível
  uint32_t t0 = micros();
  while (digitalRead(PIN_DHT) != nivel) { if (micros() - t0 > DHT_LIMITE_US) return false; }
  return true;
}
static const char *dhtEtapa = "";              // onde a última leitura falhou (diagnóstico)
#endif

static bool dhtLerHw(float &h, float &t) {
#if DHT_USAR_BIBLIOTECA
  h = dht.readHumidity(); t = dht.readTemperature();
  return !(isnan(h) || isnan(t));
#else
  uint8_t d[5] = {0, 0, 0, 0, 0};
  uint32_t baixo[40], alto[40];
  // Sinal de início: o ESP32 segura a linha em baixo (≥18 ms no DHT11, ≥1 ms no DHT22)...
  pinMode(PIN_DHT, OUTPUT);
  digitalWrite(PIN_DHT, LOW);
  delay(DHT_TIPO == 11 ? 20 : 2);
  noInterrupts();                              // a leitura leva ~5 ms e depende de tempo preciso
  pinMode(PIN_DHT, INPUT_PULLUP);              // ...e SOLTA a linha (sem forçar nível alto, como a Adafruit)
  dhtEtapa = "resposta";
  bool ok = dhtEsperar(LOW)                    // sensor puxa a linha para baixo (resposta, ~80 µs)
         && dhtPulso(LOW) != DHT_TIMEOUT
         && dhtPulso(HIGH) != DHT_TIMEOUT;     // ~80 µs em alto; depois vêm os 40 bits
  for (int i = 0; ok && i < 40; i++) {         // cada bit: ~50 µs baixo + (26 µs = 0 | 70 µs = 1) alto
    baixo[i] = dhtPulso(LOW);
    alto[i]  = dhtPulso(HIGH);
    if (baixo[i] == DHT_TIMEOUT || alto[i] == DHT_TIMEOUT) { ok = false; dhtEtapa = "bits"; }
  }
  interrupts();
  if (!ok) return false;
  for (int i = 0; i < 40; i++) { d[i / 8] <<= 1; if (alto[i] > baixo[i]) d[i / 8] |= 1; }
  if ((uint8_t)(d[0] + d[1] + d[2] + d[3]) != d[4]) { dhtEtapa = "checksum"; return false; }
  if (DHT_TIPO == 11) {
    h = d[0] + d[1] / 10.0f;
    t = (d[2] & 0x7F) + (d[3] & 0x0F) / 10.0f; if (d[2] & 0x80) t = -t;
  } else {
    h = ((d[0] << 8) | d[1]) / 10.0f;
    t = (((d[2] & 0x7F) << 8) | d[3]) / 10.0f; if (d[2] & 0x80) t = -t;
  }
  return true;
#endif
}

// Mantém a última leitura válida; se a leitura falhar, registra e conserva a anterior
static void lerDht() {
  float h, t;
  for (int tentativa = 0; tentativa < 2; tentativa++) {   // uma nova tentativa antes de desistir
    if (dhtLerHw(h, t) && h >= 0 && h <= 100) { umidPct = h; tempC = t; return; }
    delay(50);
  }
#if DHT_USAR_BIBLIOTECA
  Serial.println("[dht] leitura inválida — mantendo a anterior");
#else
  Serial.printf("[dht] leitura inválida (etapa: %s) — mantendo a anterior\n", dhtEtapa);
#endif
}

// ------------------------------- Rede --------------------------------
static void connectWifi() {
  Serial.print("[wifi] conectando");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD, WIFI_CHANNEL);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) { delay(250); Serial.print("."); }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" ok");
    configTime(0, 0, "pool.ntp.org");
  } else {
    Serial.println(" FALHOU (modo offline: ocorrências ficam na fila)");
  }
}

// Retorna true se a API aceitou (HTTP 2xx)
static bool postJson(const String &body) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  WiFiClientSecure secure;
  bool https = String(API_URL).startsWith("https");
  if (https) {
    secure.setInsecure();  // SÓ para demo/simulador. Em produção: validar o certificado (setCACert).
    http.begin(secure, API_URL);
  } else {
    http.begin(API_URL);
  }
  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Id", DEVICE_ID);
  http.addHeader("X-Device-Key", DEVICE_KEY);
  int code = http.POST(body);
  http.end();
  Serial.printf("[api] POST -> HTTP %d\n", code);
  return code >= 200 && code < 300;
}

static void enqueue(const String &body) {
  if (qCount == QUEUE_SIZE) {  // fila cheia: descarta a mais antiga (e registra)
    Serial.println("[fila] cheia — descartando a ocorrência mais antiga");
    qHead = (qHead + 1) % QUEUE_SIZE; qCount--;
  }
  queueBuf[(qHead + qCount) % QUEUE_SIZE] = body;
  qCount++;
  Serial.printf("[fila] %d pendente(s)\n", qCount);
}

static void flushQueue() {
  while (qCount > 0) {
    if (!postJson(queueBuf[qHead])) return;  // ainda sem rede: tenta depois
    queueBuf[qHead] = "";
    qHead = (qHead + 1) % QUEUE_SIZE; qCount--;
    Serial.printf("[fila] reenviada, restam %d\n", qCount);
  }
}

// ------------------------- Montagem da ocorrência -------------------------
static void fmtNum(char *out, size_t n, float v, const char *fmt) {
  if (isnan(v)) snprintf(out, n, "null"); else snprintf(out, n, fmt, v);   // JSON válido sem leitura
}

static String buildPayload(int cls, const float *f, int src) {
  float lux = powf(10.0f, f[0]);
  float luxMin = powf(10.0f, f[1]);
  float lat = ROUTE[routeIdx][0], lon = ROUTE[routeIdx][1];
  char sT[12], sH[12];
  fmtNum(sT, sizeof sT, tempC, "%.1f");
  fmtNum(sH, sizeof sH, umidPct, "%.0f");
  const char *ctx = (!isnan(umidPct) && umidPct >= UMIDADE_ALTA_PCT) ? "umidade_alta" : "normal";
  char buf[560];
  snprintf(buf, sizeof buf,
    "{\"device_id\":\"%s\",\"event_id\":\"%s-%lu\",\"ts\":\"%s\","
    "\"lat\":%.5f,\"lon\":%.5f,\"position_source\":\"simulated\","
    "\"classe\":\"%s\",\"lux_medio\":%.1f,\"lux_min\":%.1f,\"cv\":%.3f,"
    "\"janelas_confirmadas\":%d,\"temperatura_c\":%s,\"umidade_pct\":%s,\"contexto\":\"%s\","
    "\"fonte_luz\":\"%s\"}",
    DEVICE_ID, DEVICE_ID, (unsigned long)eventSeq, isoTime().c_str(),
    lat, lon, CLASS_NAMES[cls], lux, luxMin, f[3], CONFIRM_WINDOWS, sT, sH, ctx,
    src == 0 ? "sensor" : "teste");
  return String(buf);
}

// ------------------------------- Setup -------------------------------
void setup() {
  Serial.begin(115200);
  randomSeed(esp_random());
  pinMode(PIN_TRIG, OUTPUT); pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_LED_G, OUTPUT); pinMode(PIN_LED_Y, OUTPUT); pinMode(PIN_LED_R, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  analogReadResolution(12);
#if DHT_USAR_BIBLIOTECA
  dht.begin();
#endif
  setLeds(false, false, false);
  Serial.println("\n=== Rua Acesa · Robô de ronda ===");
  connectWifi();
  lerDht();
  Serial.println("[ronda] INICIADA (automática)");
  setLeds(true, false, false);
}

// ------------------------------- Loop --------------------------------
// Toda a ronda acontece aqui: sensores → IA → decisão → alerta → envio.
void loop() {
  unsigned long now = millis();

  // ---- Fonte da luz: se mudou, descarta a janela misturada ----
  int src = fonteAtual();
  if (src != lastSrc) {
    lastSrc = src; winCount = 0; badStreak = 0;
    Serial.printf("[fonte] luz: %s\n", SRC_NAMES[src]);
  }

  // ---- Rede: tenta reconectar a cada 10 s se caiu ----
  if (WiFi.status() != WL_CONNECTED && now - lastWifiTryMs > 10000) {
    lastWifiTryMs = now; WiFi.reconnect();
  }

  // ---- DHT: temperatura e umidade do ponto ----
  if (now - lastDhtMs >= DHT_INTERVAL_MS) { lastDhtMs = now; lerDht(); }

  // ---- Ultrassônico: obstáculo à frente? ----
  if (now - lastDistMs >= DIST_INTERVAL_MS) {
    lastDistMs = now;
    distCm = readDistanceCm();
    bool obst = distCm < OBSTACLE_CM;
    if (obst && !obstaculoAtivo) {
      Serial.printf("[obstaculo] %.0f cm — leitura de luz suspensa (o LDR pode estar sombreado)\n", distCm);
      beep(2, 80, 80);
    } else if (!obst && obstaculoAtivo) {
      Serial.println("[obstaculo] caminho livre — retomando leituras");
    }
    obstaculoAtivo = obst;
    if (obst) { winCount = 0; badStreak = 0; }   // não classifica luz com obstáculo na frente
  }

  // ---- LEDs: vermelho (acabou de reportar) > amarelo (atenção/obstáculo) > verde (ok) ----
  if (now < redUntilMs)                    setLeds(false, false, true);
  else if (obstaculoAtivo || badStreak > 0) setLeds(false, true, false);
  else                                     setLeds(true, false, false);

  // "Avança" um ponto da rota simulada
  if (now - lastStepMs >= STEP_EVERY_MS) {
    lastStepMs = now;
    routeIdx = (routeIdx + 1) % ROUTE_LEN;
  }

  if (obstaculoAtivo) { delay(10); return; }

  // ---- Amostragem do LDR ----
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) { delay(5); return; }
  lastSampleMs = now;
  window[winCount++] = readLux(src);
  if (winCount < WINDOW_SIZE) return;
  winCount = 0;

  // ---- Janela completa: extrai atributos e classifica (TinyML) ----
  float f[N_FEATURES];
  extractFeatures(window, WINDOW_SIZE, f);
  int cls = classifyWindow(f);
  Serial.printf("[ia] lux=%.1f min=%.1f cv=%.2f -> %s | T=%.1fC U=%.0f%% | rota %d (%.5f,%.5f)\n",
                powf(10.0f, f[0]), powf(10.0f, f[1]), f[3], CLASS_NAMES[cls],
                tempC, umidPct, routeIdx, ROUTE[routeIdx][0], ROUTE[routeIdx][1]);

  if (cls == CLASS_ADEQUADO) { badStreak = 0; flushQueue(); return; }

  badStreak++;
  if (badStreak < CONFIRM_WINDOWS) return;  // histerese: evita falso alarme por sombra passageira

  // ---- Ocorrência confirmada ----
  badStreak = 0;
  // Cooldown só vale para a MESMA classe: mudar de ESCURO para INSTAVEL, por exemplo, é um fato novo
  if (lastReportMs != 0 && cls == lastReportClass && now - lastReportMs < REPORT_COOLDOWN_MS) {
    Serial.println("[ocorrencia] já reportada há pouco — ignorando (cooldown)");
    return;
  }
  lastReportMs = now;
  lastReportClass = cls;
  redUntilMs = now + RED_LED_MS;
  setLeds(false, false, true);
  beep(3, 200, 100);
  eventSeq++;
  String body = buildPayload(cls, f, src);
  Serial.println("[ocorrencia] CONFIRMADA: " + body);
  if (!postJson(body)) { Serial.println("[api] falha — guardando na fila"); enqueue(body); }
}

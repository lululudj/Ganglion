/*
 * S11 — Ganglion external neural module firmware for ESP32-S3
 * ----------------------------------------------------------------
 * Role: tensor-layer module endpoint. Receives the backbone's layer-16
 * hidden state (4096 x fp32, 16 KB), applies the module transform
 *
 *     h -> h * 5.0f + 2.0f
 *
 * and streams the result back. Bit-exact with the host-side reference:
 * IEEE-754 single-precision multiply then add, no FMA on Xtensa LX7.
 *
 * Transport (compile-time choice):
 *   WIFI_SSID == ""  USB-Serial/JTAG — zero config, used by bridge.py
 *   WIFI_SSID set    WiFi station + TCP server on MODULE_PORT
 *
 * Frame protocol (identical to S5/S7, little-endian):
 *   [u32 len][payload]
 *   HELLO on session start:  [4][int32 tenant_id = 1]
 *   STOP from host:          [0] -> module replies [0], session ends
 *   data frame:              len == 16384 -> stream back transformed bytes
 *
 * Streaming: the transform is elementwise, so response bytes are produced
 * while request bytes are still arriving (full-duplex pipelining) — this
 * roughly halves the effective round-trip on both USB and WiFi.
 *
 * No debug prints on the data transport: anything extra corrupts the
 * frame stream. All timing is measured host-side.
 */
#include <Arduino.h>
#include <WiFi.h>

// ---- transport config ----
// Default: USB-Serial/JTAG (hardware CDC) — stable link; the host side MUST
// write frames in small paced chunks (see s11/bridge.py writeChunked) because
// the HWCDC RX ringbuffer overflows under large single writes.
// WiFi AP mode: set AP_SSID non-empty (host connects to the module hotspot).
// WiFi STA mode: set WIFI_SSID (module joins an existing network).
static const char*    AP_SSID     = "";     // "" -> USB mode
static const char*    AP_PASS     = "12345678";
static const char*    WIFI_SSID   = "";    // "" -> AP/USB modes
static const char*    WIFI_PASS   = "";
static const uint16_t MODULE_PORT = 3333;

// ---- protocol constants ----
static const int    DIM         = 4096;
static const size_t FRAME_BYTES = (size_t)DIM * 4;   // 16384
static const size_t CHUNK       = 2048;              // multiple of 4

static const uint32_t STREAM_TIMEOUT_MS = 8000;  // true-deadline watchdog
static const uint32_t HELLO_PERIOD_MS   = 1000;  // idle re-HELLO (serial mode)

// ---------- transport primitives ----------
static void writeAll(const uint8_t* p, size_t n, Stream& s) {
  while (n > 0) {
    size_t w = s.write(p, n);
    if (w == 0) return;             // transport gone
    p += w; n -= w;
  }
}

static void writeU32(uint32_t v, Stream& s) {
  uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
  writeAll(b, 4, s);
}

static bool readExact(uint8_t* dst, size_t n, Stream& s) {
  size_t got = 0;
  uint32_t lastByte = millis();
  while (got < n) {
    int avail = s.available();
    if (avail > 0) {
      size_t want = n - got;
      if ((size_t)avail < want) want = (size_t)avail;
      size_t r = s.readBytes((char*)(dst + got), want);
      if (r == 0) return false;
      got += r;
      lastByte = millis();
    } else {
      // Yield CPU while waiting: busy-polling starves the idle task and
      // trips the task watchdog (~5s), which reboots the module mid-session
      // (observed as a stray HELLO interleaved into the response stream).
      if (millis() - lastByte > STREAM_TIMEOUT_MS) return false;
      delay(1);
    }
  }
  return true;
}

static void sendHello(Stream& s) {
  writeU32(4, s);
  uint8_t hello[4] = { 1, 0, 0, 0 };
  writeAll(hello, 4, s);
}

static uint32_t parseU32(const uint8_t* b) {
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// ---------- one data frame: receive whole frame, transform, send whole ----------
// NOTE: unlike a streaming design, the full frame is buffered before the
// response starts. The USB-Serial/JTAG peripheral cannot sustain duplex
// traffic (its ISR drops RX bytes while busy with TX), so request-response
// framing eliminates the RX/TX race at the cost of pipelining.
static uint8_t frameBuf[FRAME_BYTES];

static bool serveDataFrame(Stream& s) {
  if (!readExact(frameBuf, FRAME_BYTES, s)) return false;
  // Two explicitly rounded float32 ops to match the numpy reference
  // (mul-then-add, two roundings). The compiler would otherwise contract
  // this into a single-rounding fused multiply-add and differ by 1 ULP
  // on ~20% of elements.
  for (size_t i = 0; i < DIM; i++) {
    volatile float t = ((float*)frameBuf)[i] * 5.0f;
    ((float*)frameBuf)[i] = t + 2.0f;
  }
  writeU32((uint32_t)FRAME_BYTES, s);
  for (size_t off = 0; off < FRAME_BYTES; off += CHUNK) {
    writeAll(frameBuf + off, CHUNK, s);
  }
  return true;
}

// ---------- frame loop (first header already consumed) ----------
static bool frameLoop(Stream& s, uint32_t firstLen) {
  uint32_t len = firstLen;
  for (;;) {
    if (len == 0) { writeU32(0, s); return true; }   // STOP
    if (len != FRAME_BYTES) return false;            // protocol error
    if (!serveDataFrame(s)) return false;
    uint8_t hdr[4];
    if (!readExact(hdr, 4, s)) return false;
    len = parseU32(hdr);
  }
}

// ---------- USB-Serial/JTAG transport ----------
static void serveSerialForever() {
  Serial.begin(115200);          // baud is irrelevant for USB CDC
  Serial.setTimeout(STREAM_TIMEOUT_MS);
  for (;;) {
    sendHello(Serial);
    // Wait for the first frame header of a session.
    // HELLO repeats only while NO byte has arrived (pre-session idle),
    // so it can never interleave with a live session.
    uint8_t  hdr[4];
    size_t   got       = 0;
    uint32_t lastHello = millis();
    uint32_t lastByte  = millis();
    while (got < 4) {
      int c = Serial.read();
      if (c >= 0) {
        hdr[got++] = (uint8_t)c;
        lastByte = millis();
        continue;
      }
      // partial header stalled -> drop it and resync
      if (got > 0 && millis() - lastByte > STREAM_TIMEOUT_MS) {
        got = 0;
        lastByte = millis();
      }
      // totally idle -> keep announcing module readiness
      if (got == 0 && millis() - lastHello > HELLO_PERIOD_MS) {
        sendHello(Serial);
        lastHello = millis();
      }
      delay(2);
    }
    frameLoop(Serial, parseU32(hdr));   // ends on STOP / timeout / error
  }
}

// ---------- WiFi TCP transport ----------
static void serveWifiForever() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(100);
  WiFiServer server(MODULE_PORT);
  server.begin();
  for (;;) {
    WiFiClient cli = server.accept();
    if (!cli) { delay(2); continue; }
    cli.setTimeout(STREAM_TIMEOUT_MS);
    sendHello(cli);
    uint8_t hdr[4];
    if (readExact(hdr, 4, cli)) {
      frameLoop(cli, parseU32(hdr));
    }
    cli.stop();
  }
}

// ---------- WiFi AP transport (module is its own hotspot) ----------
static void serveApForever() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  IPAddress ip = WiFi.softAPIP();          // default 192.168.4.1
  WiFiServer server(MODULE_PORT);
  server.begin();
  for (;;) {
    WiFiClient cli = server.accept();
    if (!cli) { delay(2); continue; }
    cli.setTimeout(STREAM_TIMEOUT_MS);
    sendHello(cli);
    uint8_t hdr[4];
    if (readExact(hdr, 4, cli)) {
      frameLoop(cli, parseU32(hdr));
    }
    cli.stop();
  }
}

void setup() {
  if (AP_SSID[0] != '\0')      serveApForever();
  else if (WIFI_SSID[0] == '\0') serveSerialForever();
  else                          serveWifiForever();
}

void loop() {}

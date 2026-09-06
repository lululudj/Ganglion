# -*- coding: utf-8 -*-
"""ESP32 直连 12 帧长测：验证第 8 帧失败是否可复现。"""
import struct
import time

import numpy as np
import serial

DIM = 4096
FRAME = DIM * 4
HELLO = struct.pack("<ii", 4, 1)

ser = serial.Serial("COM3", 115200, timeout=0.05)
ser.reset_input_buffer()
buf = b""
t0 = time.time()
while time.time() - t0 < 6:
    buf += ser.read(4096)
    i = buf.find(HELLO)
    if i >= 0:
        buf = buf[i + 8:]
        break
    if len(buf) > 16:
        buf = buf[-16:]
print("HELLO ok, rem", len(buf))

frame = np.random.RandomState(42).randn(1, DIM).astype(np.float32)
payload = struct.pack("<i", FRAME) + frame.tobytes()
ref = (frame * np.float32(5.0) + np.float32(2.0)).astype(np.float32)

ok_n = fail_n = 0
for k in range(12):
    t = time.perf_counter()
    for off in range(0, len(payload), 256):
        ser.write(payload[off:off + 256])
        time.sleep(0.002)
    g = b""
    dl = time.perf_counter() + 10
    while len(g) < 4 + FRAME and time.perf_counter() < dl:
        g += ser.read(65536)
    rtt = (time.perf_counter() - t) * 1000
    if len(g) == 4 + FRAME:
        (ln,) = struct.unpack("<i", g[:4])
        resp = np.frombuffer(g[4:4 + ln], dtype=np.float32).reshape(1, DIM)
        ok = np.array_equal(resp, ref)
        tag = "PASS" if ok else "FAIL"
        if ok:
            ok_n += 1
        else:
            fail_n += 1
        print(f"帧{k+1:2d}: RTT={rtt:6.1f}ms 位级={tag}")
    else:
        fail_n += 1
        print(f"帧{k+1:2d}: RTT={rtt:6.1f}ms INCOMPLETE {len(g)}/16388")

for off in range(0, 4, 256):
    ser.write(struct.pack("<i", 0)[off:off + 256])
ser.close()
print(f"=== 12帧: {ok_n} PASS / {fail_n} FAIL ===")

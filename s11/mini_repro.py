# -*- coding: utf-8 -*-
"""复现 bridge 场景：HELLO 同步后等待 N 秒（模拟云端模型加载），
再发数据帧——验证 ESP32 在周期 HELLO 期间收帧是否正常。"""
import struct
import sys
import time

import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM3"
WAIT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

DIM = 4096
FRAME = DIM * 4
HELLO = struct.pack("<ii", 4, 1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ser = serial.Serial(PORT, 115200, timeout=0.05)
ser.reset_input_buffer()

# 同步 HELLO
buf = b""
t0 = time.time()
while time.time() - t0 < 6:
    buf += ser.read(4096)
    i = buf.find(HELLO)
    if i >= 0:
        buf = buf[i + len(HELLO):]
        break
    if len(buf) > 16:
        buf = buf[-16:]
print(f"[mini] HELLO 同步，剩余 {len(buf)}B", flush=True)

# 关键：等待（期间 ESP32 每秒发 HELLO，无人读取堆积在驱动缓冲）
print(f"[mini] 等待 {WAIT_S}s（模拟云端模型加载）...", flush=True)
time.sleep(WAIT_S)

# 发一帧（分块限速，与 bridge 相同）
frame = np.random.RandomState(42).randn(1, DIM).astype(np.float32)
payload = struct.pack("<i", FRAME) + frame.tobytes()
t_send = time.perf_counter()
for off in range(0, len(payload), 256):
    ser.write(payload[off:off + 256])
    time.sleep(0.002)
print(f"[mini] 已发送 {len(payload)}B，读响应 ...", flush=True)

got = b""
deadline = time.perf_counter() + 12.0
while len(got) < 4 + FRAME and time.perf_counter() < deadline:
    got += ser.read(65536)
t_rtt = time.perf_counter() - t_send

if len(got) < 4 + FRAME:
    print(f"[mini] ✗ 只收到 {len(got)}/{4+FRAME} B（RTT {t_rtt*1000:.0f}ms）", flush=True)
    print(f"[mini] 前 32B: {got[:32].hex()}", flush=True)
else:
    (ln,) = struct.unpack("<i", got[:4])
    resp = np.frombuffer(got[4:4 + ln], dtype=np.float32).reshape(1, DIM)
    ref = (frame * np.float32(5.0) + np.float32(2.0)).astype(np.float32)
    ok = np.array_equal(resp, ref)
    print(f"[mini] ✓ 完整帧 RTT={t_rtt*1000:.0f}ms | 位级={'PASS' if ok else 'FAIL'}", flush=True)

# STOP
for off in range(0, 4, 256):
    ser.write(struct.pack("<i", 0)[off:off + 256])
g = b""
t0 = time.time()
while len(g) < 4 and time.time() - t0 < 10:
    g += ser.read(4)
print(f"[mini] STOP 确认: {g.hex() if g else 'TIMEOUT'}", flush=True)
ser.close()

# -*- coding: utf-8 -*-
"""ESP32 直连诊断：绕过云端/桥接器，COM3 上直接跑一轮完整帧往返。

步骤：
  1. 打开 COM3，同步 HELLO
  2. 发送一帧 16384B（与云端宿主完全相同的帧格式）
  3. 限时读取响应（打印分批到达的字节数与时间线）
  4. 位级校验 h*5+2
"""
import glob
import struct
import sys
import time

import numpy as np
import serial

PORT = None
for p in [f"COM{i}" for i in range(1, 16)]:
    try:
        s = serial.Serial(p, timeout=0.02)
        s.close()
        PORT = p
        break
    except Exception:
        continue
if PORT is None:
    PORT = "COM3"
if len(sys.argv) > 1:
    PORT = sys.argv[1]
DIM = 4096
FRAME = DIM * 4
HELLO = struct.pack("<ii", 4, 1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ser = serial.Serial(PORT, 115200, timeout=0.05)
ser.reset_input_buffer()
print(f"[diag] {PORT} 已打开，等待 HELLO ...", flush=True)

# 1. 同步 HELLO
buf = b""
t0 = time.time()
hello_at = None
while time.time() - t0 < 6:
    buf += ser.read(4096)
    i = buf.find(HELLO)
    if i >= 0:
        hello_at = time.time() - t0
        buf = buf[i + len(HELLO):]
        break
    if len(buf) > 16:
        buf = buf[-16:]
if hello_at is None:
    print("[diag] ✗ 6 秒内未收到 HELLO——固件未运行")
    sys.exit(2)
print(f"[diag] ✓ HELLO 同步（{hello_at:.2f}s），剩余 {len(buf)}B", flush=True)

# 2. 发送一帧（分块限速写：HWCDC/TinyUSB CDC 的设备端 RX 环形缓冲小，
#    主机一次性大块写会溢出/卡死，分小块+间隔让设备中断处理跟上）
frame = np.random.RandomState(42).randn(1, DIM).astype(np.float32)
payload = struct.pack("<i", FRAME) + frame.tobytes()
t_send = time.perf_counter()
CHUNK_W = 256
PACE_MS = 0.002
for off in range(0, len(payload), CHUNK_W):
    ser.write(payload[off:off + CHUNK_W])
    time.sleep(PACE_MS)
print(f"[diag] 已发送 {len(payload)}B（分块限速），等待响应 ...", flush=True)

# 3. 限时读响应（打印到达时间线）
got = b""
deadline = time.perf_counter() + 15.0
marks = []
while len(got) < 4 + FRAME and time.perf_counter() < deadline:
    chunk = ser.read(65536)
    if chunk:
        got += chunk
        marks.append((round(time.perf_counter() - t_send, 4), len(got)))
t_rtt = time.perf_counter() - t_send

if marks:
    print(f"[diag] 到达时间线（前 5 批）: {marks[:5]}", flush=True)
    print(f"[diag] 到达时间线（后 3 批）: {marks[-3:]}", flush=True)

if len(got) < 4 + FRAME:
    print(f"[diag] ✗ 超时：只收到 {len(got)}/{4+FRAME} B", flush=True)
    ser.close()
    sys.exit(3)

# 4. 校验
(ln,) = struct.unpack("<i", got[:4])
resp = np.frombuffer(got[4:4 + ln], dtype=np.float32).reshape(1, DIM)
ref = (frame * np.float32(5.0) + np.float32(2.0)).astype(np.float32)
ok = np.array_equal(resp, ref)
print(f"[diag] 响应头 len={ln} | RTT={t_rtt*1000:.1f}ms | "
      f"位级一致={'PASS' if ok else 'FAIL'}", flush=True)
if not ok:
    diff = np.where(resp != ref)[1]
    print(f"[diag] 不一致元素数 {len(diff)}/{DIM}，首个位置 {diff[:5] if len(diff) else '无'}", flush=True)
    if len(diff):
        i = int(diff[0])
        print(f"[diag] 输入[{i}] = {frame[0,i]!r} (hex {frame[0,i].view(np.uint32):08x})", flush=True)
        print(f"[diag] 期望[{i}] = {ref[0,i]!r} (hex {ref[0,i].view(np.uint32):08x})", flush=True)
        print(f"[diag] 实收[{i}] = {resp[0,i]!r} (hex {resp[0,i].view(np.uint32):08x})", flush=True)
        # 检查是否整体错位 4 字节
        resp_alt = np.frombuffer(got[8:8+ln-4], dtype=np.float32).reshape(1, -1)[:, :DIM]
        print(f"[diag] 若错位+4B: 一致={np.array_equal(resp_alt, ref)}", flush=True)
        resp_alt2 = np.frombuffer(got[4:4+ln], dtype=np.float32).reshape(1, -1)[:, 1:DIM+1]
        print(f"[diag] 若错位-4B: 一致={np.array_equal(resp_alt2, ref)}", flush=True)

# 再来 5 帧测稳定性
rts = []
for k in range(5):
    t = time.perf_counter()
    for off in range(0, len(payload), CHUNK_W):
        ser.write(payload[off:off + CHUNK_W])
        time.sleep(PACE_MS)
    g = b""
    while len(g) < 4 + FRAME:
        g += ser.read(65536)
    rts.append((time.perf_counter() - t) * 1000)
print(f"[diag] 后续 5 帧 RTT: {[round(x,1) for x in rts]} ms", flush=True)

# STOP
for off in range(0, 4, CHUNK_W):
    ser.write(struct.pack("<i", 0)[off:off + CHUNK_W])
g = b""
while len(g) < 4:
    g += ser.read(4)
print(f"[diag] STOP 确认: {g.hex()} (期望 00000000)", flush=True)
ser.close()
print("[diag] 完成", flush=True)

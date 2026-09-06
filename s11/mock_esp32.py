# -*- coding: utf-8 -*-
"""S11 mock 模块：ESP32 固件的行为孪生（TCP :3333）。

用途：
  1. 烧录前验证桥接器/云端宿主的协议链路（无硬件）
  2. 固件行为的 Python 参考实现

可选 WiFi 抖动模拟（验证宿主 8s 看门狗不误杀）：
  --jitter-prob 0.1 --jitter-ms 50:300   # 10% 帧延迟 50-300ms
"""
import argparse
import socket
import struct
import sys
import threading
import time

import numpy as np

DIM = 4096
SCALE, SHIFT = 5.0, 2.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def serve_conn(conn, addr, jprob, jlo, jhi):
    conn.settimeout(30)
    frames = 0
    try:
        conn.sendall(struct.pack("<i", 4) + struct.pack("<i", 1))   # HELLO
        while True:
            hdr = recv_exact(conn, 4)
            (ln,) = struct.unpack("<i", hdr)
            if ln == 0:
                conn.sendall(struct.pack("<i", 0))
                break
            payload = recv_exact(conn, ln)
            if jprob > 0 and np.random.rand() < jprob:
                time.sleep(np.random.uniform(jlo, jhi) / 1000.0)
            arr = np.frombuffer(payload, dtype=np.float32).reshape(-1, DIM)
            out = (arr * SCALE + SHIFT).astype(np.float32)
            resp = out.tobytes()
            conn.sendall(struct.pack("<i", len(resp)) + resp)
            frames += 1
    except (ConnectionError, OSError):
        pass
    finally:
        print(f"[mock] {addr} 会话结束（{frames} 帧）", flush=True)
        try:
            conn.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--jitter-prob", type=float, default=0.0)
    ap.add_argument("--jitter-ms", default="50:300",
                    help="lo:hi 毫秒区间")
    args = ap.parse_args()
    jlo, jhi = (float(x) for x in args.jitter_ms.split(":"))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(4)
    print(f"[mock] ESP32 行为孪生就绪 127.0.0.1:{args.port} | "
          f"jitter={args.jitter_prob}({jlo}-{jhi}ms) | 变换 h*{SCALE}+{SHIFT} "
          f"dim={DIM} fp32", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[mock] {addr} 已连接", flush=True)
        t = threading.Thread(target=serve_conn,
                             args=(conn, addr, args.jitter_prob, jlo, jhi),
                             daemon=True)
        t.start()


if __name__ == "__main__":
    main()

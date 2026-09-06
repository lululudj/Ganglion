# -*- coding: utf-8 -*-
"""S5 跨机实验——本地端：SSH 反向隧道 + 外置神经模块 TCP 服务。

架构（云端为主力）：
  云机 RTX 4090 跑 Qwen3-0.6B 骨干（HOST）
   │ hidden_states (fp32, [1,1024], 4KB/token)
   ▼ SSH 反向隧道（云端 localhost:19000 → 本机）
  本机外置模块进程（MODULE）：numpy 线性变换 h*5+2

本脚本：
  1. 建立 SSH 传输通道，请求云机 19000 端口反向转发
  2. accept 转发信道，直接在信道上跑模块服务循环（帧协议：4字节长度前缀）
  3. 逐次记录往返延迟，收到 STOP 帧后打印统计
"""
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 12920
USER = "root"
PASSWORD = os.environ["GANGLION_SSH_PASS"]
REMOTE_PORT = 19002

DIM = 4096             # Qwen3-8B hidden_size（每帧 16KB fp32）
SCALE, SHIFT = 5.0, 2.0  # 可独立复算的变换规范（宿主侧可校验）

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

rtts = []            # 每帧往返延迟（本机视角：发送→收到，即模块处理+回程）
lock = threading.Lock()
stop_flag = threading.Event()


def recv_exact(chan, n):
    buf = b""
    while len(buf) < n:
        chunk = chan.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("tunnel closed")
        buf += chunk
    return buf


def serve_channel(chan):
    """在单条转发信道上服务宿主（帧协议：len4 + payload）。"""
    chan.settimeout(30)
    try:
        while not stop_flag.is_set():
            hdr = recv_exact(chan, 4)
            (n,) = struct.unpack("<i", hdr)
            if n == 0:                       # STOP 帧
                chan.sendall(struct.pack("<i", 0))
                break
            payload = recv_exact(chan, n)
            t0 = time.perf_counter()
            arr = np.frombuffer(payload, dtype=np.float32).reshape(-1, DIM)
            out = (arr * SCALE + SHIFT).astype(np.float32)  # 模块计算
            proc_ms = (time.perf_counter() - t0) * 1000.0
            resp = out.tobytes()
            chan.sendall(struct.pack("<i", len(resp)) + resp)
            with lock:
                rtts.append((proc_ms, n))
    except (ConnectionError, socket.timeout, OSError):
        pass
    finally:
        try:
            chan.close()
        except Exception:
            pass


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0  # 秒后自动退出
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                timeout=20, banner_timeout=20)
    transport = cli.get_transport()
    transport.set_keepalive(15)

    # 请求反向端口转发：云机连 localhost:19000 → 到达本机
    transport.request_port_forward("127.0.0.1", REMOTE_PORT)
    print(f"[本地模块服务] 反向隧道就绪：云机 localhost:{REMOTE_PORT} → 本机", flush=True)
    print(f"[本地模块服务] 变换规范: h*{SCALE}+{SHIFT}, dim={DIM}, fp32", flush=True)
    print(f"[本地模块服务] 等待云端宿主连接 ... (运行 {duration:.0f}s 后自动退出)", flush=True)

    threads = []
    t_start = time.perf_counter()
    try:
        while not stop_flag.is_set():
            chan = transport.accept(timeout=1.0)
            if chan is not None:
                print("[本地模块服务] 云端宿主已连接", flush=True)
                t = threading.Thread(target=serve_channel, args=(chan,), daemon=True)
                t.start()
                threads.append(t)
            if time.perf_counter() - t_start > duration:
                print(f"\n[本地模块服务] 运行 {duration:.0f}s 到期，自动退出", flush=True)
                break
    except KeyboardInterrupt:
        stop_flag.set()
        print("\n[本地模块服务] 收到中断，停止", flush=True)
    finally:
        time.sleep(0.3)
        with lock:
            total = len(rtts)
            if total:
                procs = sorted(r[0] for r in rtts)
                sizes = {r[1] for r in rtts}
                print(f"\n[本地模块服务] 统计：服务帧数={total}, 帧大小={sizes}")
                print(f"  模块端纯处理 p50={procs[total//2]:.3f}ms "
                      f"min={procs[0]:.3f}ms max={procs[-1]:.3f}ms")
        for t in threads:
            t.join(timeout=2)
        transport.cancel_port_forward("127.0.0.1", REMOTE_PORT)
        cli.close()


if __name__ == "__main__":
    main()

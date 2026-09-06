# -*- coding: utf-8 -*-
"""S7 多租户实验——本地端：SSH 反向隧道 + N 个外置模块租户。

每个 accept 的信道 = 一个机器人租户（独立故障域）：
  1. 数据隧道：云机 localhost:19002 → 本机（N 条信道，每条=1 租户）
  2. 控制隧道：云机 localhost:19003 → 本机（实验 C 前云端发 ARM 命令武装故障）
  3. 租户号 = 当前活跃信道数 + 1（每批租户从 1 重新编号，HELLO 帧告知云端）
  4. 故障注入：被武装租户的信道在服务 N 帧后主动关闭
     （模拟该租户的模块进程死亡——其他租户信道不受影响）

帧协议：4字节长度前缀 + payload
HELLO 帧：连接后本地端立刻发送 struct('i', tenant_id)
控制命令：ARM:<tenant>:<after_frames> → 回复 OK
"""
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
PASSWORD = "***REMOVED-SSH-PASSWORD***"
DATA_PORT = 19002
CTRL_PORT = 19003

DIM = 4096
SCALE, SHIFT = 5.0, 2.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tenant_stats = {}       # tid -> {"frames": n, "closed_at_frame": n or None}
active_chans = set()    # 活跃信道（用于租户号分配）
armed = {"tenant": None, "after": None}   # 武装状态（由控制信道设置）
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


def serve_channel(chan, tid):
    """单租户模块服务循环：帧协议 + 已武装的定向故障注入。"""
    chan.settimeout(30)
    frames = 0
    closed_at = None
    try:
        # HELLO 帧：告知云端本信道的租户号
        chan.sendall(struct.pack("<i", 4) + struct.pack("<i", tid))
        while not stop_flag.is_set():
            hdr = recv_exact(chan, 4)
            (n,) = struct.unpack("<i", hdr)
            if n == 0:                       # STOP 帧
                chan.sendall(struct.pack("<i", 0))
                break
            payload = recv_exact(chan, n)
            arr = np.frombuffer(payload, dtype=np.float32).reshape(-1, DIM)
            out = (arr * SCALE + SHIFT).astype(np.float32)  # 模块计算
            resp = out.tobytes()
            chan.sendall(struct.pack("<i", len(resp)) + resp)
            frames += 1
            # 已武装的定向故障：目标租户达到帧数后主动关闭（模块死亡语义）
            with lock:
                is_victim = (armed["tenant"] == tid and
                             armed["after"] is not None and
                             frames >= armed["after"])
            if is_victim:
                closed_at = frames
                print(f"[租户{tid}] ⚠ 故障注入：已服务 {frames} 帧，主动关闭信道"
                      f"（该租户模块死亡）", flush=True)
                break
    except (ConnectionError, socket.timeout, OSError):
        pass
    finally:
        with lock:
            tenant_stats[tid] = {"frames": frames, "closed_at_frame": closed_at}
            active_chans.discard(tid)
        try:
            chan.close()
        except Exception:
            pass


def serve_control(transport):
    """控制信道：接受 ARM:<tenant>:<after> 命令并回复 OK。"""
    while not stop_flag.is_set():
        chan = transport.accept(timeout=1.0)
        if chan is None:
            continue
        try:
            chan.settimeout(10)
            hdr = recv_exact(chan, 4)
            (n,) = struct.unpack("<i", hdr)
            cmd = recv_exact(chan, n).decode()
            if cmd.startswith("ARM:"):
                _, t, a = cmd.split(":")
                with lock:
                    armed["tenant"] = int(t)
                    armed["after"] = int(a)
                print(f"[控制] 故障已武装：租户{t} 在 {a} 帧后死亡", flush=True)
                reply = b"OK"
            else:
                reply = b"ERR"
            chan.sendall(struct.pack("<i", len(reply)) + reply)
        except Exception as e:
            print(f"[控制] 错误：{e}", flush=True)
        finally:
            try:
                chan.close()
            except Exception:
                pass


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                timeout=20, banner_timeout=20)
    transport = cli.get_transport()
    transport.set_keepalive(15)
    transport.request_port_forward("127.0.0.1", DATA_PORT)
    print(f"[本地模块服务 S7] 数据隧道就绪：云机 localhost:{DATA_PORT} → 本机", flush=True)

    cli2 = paramiko.SSHClient()
    cli2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli2.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                 timeout=20, banner_timeout=20)
    transport2 = cli2.get_transport()
    transport2.set_keepalive(15)
    transport2.request_port_forward("127.0.0.1", CTRL_PORT)
    print(f"[本地模块服务 S7] 控制隧道就绪：云机 localhost:{CTRL_PORT} → 本机", flush=True)

    ctrl_thread = threading.Thread(target=serve_control, args=(transport2,), daemon=True)
    ctrl_thread.start()
    print(f"[本地模块服务 S7] 每信道=1 租户（批内从1编号）| 变换: h*{SCALE}+{SHIFT} | "
          f"dim={DIM} fp32", flush=True)
    print(f"[本地模块服务 S7] 等待云端宿主连接（运行 {duration:.0f}s 后退出）", flush=True)

    threads = []
    t_start = time.perf_counter()
    try:
        while not stop_flag.is_set():
            chan = transport.accept(timeout=1.0)
            if chan is not None:
                with lock:
                    tid = len(active_chans) + 1
                    active_chans.add(tid)
                print(f"[本地模块服务] 租户{tid} 已连接", flush=True)
                t = threading.Thread(target=serve_channel, args=(chan, tid), daemon=True)
                t.start()
                threads.append(t)
            if time.perf_counter() - t_start > duration:
                print(f"\n[本地模块服务] 运行 {duration:.0f}s 到期，自动退出", flush=True)
                break
    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        time.sleep(0.5)
        with lock:
            print("\n[本地模块服务] 租户统计：")
            for tid in sorted(tenant_stats):
                s = tenant_stats[tid]
                fault = f" → 第{s['closed_at_frame']}帧死亡(注入)" if s["closed_at_frame"] else ""
                print(f"  租户{tid}: 服务帧数={s['frames']}{fault}")
        for t in threads:
            t.join(timeout=2)
        transport.cancel_port_forward("127.0.0.1", DATA_PORT)
        transport2.cancel_port_forward("127.0.0.1", CTRL_PORT)
        cli.close()
        cli2.close()


if __name__ == "__main__":
    main()

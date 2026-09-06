# -*- coding: utf-8 -*-
"""S11 桥接器（Windows 侧）：SSH 反向隧道 ↔ 模块端点。

数据链路：
    云机 127.0.0.1:19002（隧道入口）
      ↕ SSH 反向隧道（公网）
    本机
      ↕ --mode serial: USB-Serial/JTAG（COM 口，ESP32-S3 固件）
      ↕ --mode tcp:    TCP（mock_esp32.py 或 WiFi 版固件）

职责：
  1. 建立 SSH 反向隧道（paramiko request_port_forward）
  2. 打开模块侧传输（串口或 TCP），扫描同步到 HELLO 帧
  3. 云端信道接入后：合成 HELLO 下发（租户注册），随后双向泵流
  4. 模块→云端方向过滤 HELLO 帧（串行模式固件空闲时会周期性播发
     HELLO——桥接层吸收，保证云端字节流纯净）
  5. 故障注入：--kill-after N 在第 N 个数据帧后关闭模块连接
     （模拟 MCU/USB 中途死亡）

帧协议（两侧一致，小端）：[u32 len][payload]
  模块侧会话开始发 HELLO：[4][int32 tenant_id]
  云端发 STOP：[0] → 模块回 [0]
"""
import argparse
import os
import socket
import struct
import sys
import threading
import time

import paramiko

SSH_HOST = "connect.bjb1.seetacloud.com"
SSH_PORT = 12920
SSH_USER = "root"
SSH_PASS = os.environ["GANGLION_SSH_PASS"]
REMOTE_TUNNEL_PORT = 19002

HELLO_MAGIC = struct.pack("<ii", 4, 1)   # len=4, tenant=1

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

stop = threading.Event()


class TcpSide:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(8.0)

    def read(self, n):
        data = self.sock.recv(n)
        if not data:
            raise ConnectionError("module tcp closed")
        return data

    def write(self, b):
        self.sock.sendall(b)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class SerialSide:
    def __init__(self, port, baud=115200):
        import serial as pyserial
        self.ser = pyserial.Serial(port, baud, timeout=0.05)
        self.ser.reset_input_buffer()

    def read(self, n):
        return self.ser.read(n) or b""

    def write(self, b):
        # Paced chunked write: the ESP32-S3 USB-Serial/JTAG peripheral's
        # RX path drops bytes when large single writes arrive (its ISR
        # cannot keep the 64B HW FIFO drained). 256B + 2ms pacing lets the
        # device-side interrupt keep up — measured 16388/16388B intact.
        CHUNK_W = 256
        for off in range(0, len(b), CHUNK_W):
            self.ser.write(b[off:off + CHUNK_W])
            time.sleep(0.002)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


def sync_hello(side, timeout_s=8.0):
    """扫描字节流直到对齐到 HELLO 帧，返回剩余字节（保持帧对齐）。"""
    buf = b""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        buf += side.read(256)
        idx = buf.find(HELLO_MAGIC)
        if idx >= 0:
            remainder = buf[idx + len(HELLO_MAGIC):]
            print(f"[bridge] 模块 HELLO 已同步（丢弃 {idx}B 前导噪声）", flush=True)
            return True, remainder
        # 只保留尾部 8 字节（HELLO 可能跨读分片）
        if len(buf) > 8:
            buf = buf[-8:]
        time.sleep(0.02)
    return False, b""


def pump_host_to_module(chan, side):
    """云端→模块：帧感知泵（解析长度前缀以计数），原样转发。"""
    frames = 0
    try:
        while not stop.is_set():
            hdr = b""
            while len(hdr) < 4:
                b_ = chan.recv(4 - len(hdr))
                if not b_:
                    return
                hdr += b_
            (ln,) = struct.unpack("<i", hdr)
            payload = b""
            while len(payload) < ln:
                b_ = chan.recv(ln - len(payload))
                if not b_:
                    return
                payload += b_
            side.write(hdr + payload)
            if ln > 0:
                frames += 1
                if frames % 12 == 0 or frames <= 2:
                    print(f"[bridge] 数据帧 #{frames}（{ln}B）云端→模块", flush=True)
            else:
                print("[bridge] STOP 帧 → 等待模块确认", flush=True)
    except Exception as e:
        print(f"[bridge] 云端→模块泵结束：{e}", flush=True)
    finally:
        stop.set()


def pump_module_to_host(side, chan, prebuf, kill_after=None):
    """模块→云端：帧过滤泵。HELLO 帧被吸收（云端已在接入时收到合成
    HELLO），其余帧原样转发。prebuf = 同步阶段的剩余字节。
    故障注入：数完第 kill_after 个数据响应后关闭连接——保证模块
    精确服务 kill_after 个完整往返后死亡（与 S5/S7 语义一致）。"""
    buf = prebuf
    frames = 0
    try:
        while not stop.is_set():
            # 每轮都读：buf 里持有不足一帧的字节时也必须继续读，
            # 否则会空转（外层 if len(buf)<4 不成立就永不 read 的 bug）
            data = side.read(65536)
            if data:
                buf += data
            if len(buf) < 4:
                time.sleep(0.02)
                continue
            while len(buf) >= 4:
                (ln,) = struct.unpack("<i", buf[:4])
                total = 4 + (ln if ln > 0 else 0)
                if len(buf) < total:
                    break
                frame = buf[:total]
                buf = buf[total:]
                if ln == 4 and frame[4:8] == HELLO_MAGIC[4:]:
                    continue        # 空闲 HELLO：吸收，不下发云端
                chan.sendall(frame)
                if ln > 0:
                    frames += 1
                    if kill_after is not None and frames >= kill_after:
                        print(f"[bridge] ⚠ 故障注入：模块已服务 {frames} 个完整往返，"
                              f"关闭连接（模块死亡）", flush=True)
                        side.close()
                        chan.close()
                        stop.set()
                        return
            # 帧头异常长：视为噪声，防止死等
            if len(buf) >= 4:
                (ln,) = struct.unpack("<i", buf[:4])
                if ln < 0 or ln > 1 << 20:
                    buf = buf[4:]
    except Exception as e:
        print(f"[bridge] 模块→云端泵结束：{e}", flush=True)
    finally:
        stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["serial", "tcp"], default="serial")
    ap.add_argument("--port", default="COM3",
                    help="serial: COM 口；tcp: 端口号")
    ap.add_argument("--host", default="127.0.0.1", help="tcp 模式模块地址")
    ap.add_argument("--kill-after", type=int, default=None,
                    help="第 N 个数据帧后关闭模块连接（故障注入）")
    ap.add_argument("--duration", type=float, default=300.0)
    args = ap.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS,
                timeout=20, banner_timeout=20)
    transport = ssh.get_transport()
    transport.set_keepalive(15)
    transport.request_port_forward("127.0.0.1", REMOTE_TUNNEL_PORT)
    print(f"[bridge] 隧道就绪：云机 127.0.0.1:{REMOTE_TUNNEL_PORT} → 本机", flush=True)

    # 模块侧先就位（串行/TCP），等待 HELLO
    if args.mode == "serial":
        side = SerialSide(args.port)
        print(f"[bridge] 串口已打开：{args.port}", flush=True)
    else:
        side = TcpSide(args.host, int(args.port))
        print(f"[bridge] TCP 已连接：{args.host}:{args.port}", flush=True)
    ok, prebuf = sync_hello(side)
    if not ok:
        print("[bridge] ✗ 未收到模块 HELLO（固件未运行或链路故障）", flush=True)
        side.close()
        transport.cancel_port_forward("127.0.0.1", REMOTE_TUNNEL_PORT)
        ssh.close()
        sys.exit(2)

    # 等云端宿主接入
    chan = transport.accept(timeout=args.duration)
    if chan is None:
        print("[bridge] 等待云端接入超时，退出", flush=True)
        side.close()
        transport.cancel_port_forward("127.0.0.1", REMOTE_TUNNEL_PORT)
        ssh.close()
        return
    chan.settimeout(8.0)
    print("[bridge] 云端宿主已接入，开始泵流", flush=True)
    # 合成 HELLO 完成租户注册（模块的 HELLO 由桥接层吸收）
    chan.sendall(HELLO_MAGIC)

    t1 = threading.Thread(target=pump_module_to_host,
                          args=(side, chan, prebuf, args.kill_after))
    t2 = threading.Thread(target=pump_host_to_module, args=(chan, side))
    t1.start(); t2.start()
    t1.join(args.duration); t2.join(args.duration)

    print("[bridge] 会话结束，清理", flush=True)
    try:
        side.close()
        chan.close()
    except Exception:
        pass
    transport.cancel_port_forward("127.0.0.1", REMOTE_TUNNEL_PORT)
    ssh.close()


if __name__ == "__main__":
    main()

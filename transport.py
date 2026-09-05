# -*- coding: utf-8 -*-
"""Ganglion 数据面传输：共享内存双槽通道 + Pipe / TCP 回环通道与基准。

- ShmWriter / ShmReader：SPSC 双槽轮换。每槽 [seq int64][len int64][payload...]，
  seq 最后发布（写定序）；同步一问一答协议下无双发竞态。
- benchmark_channel()：给定通道类型/维度/序列长度/dtype，测量逐次往返延迟分布。
- benchmark_gpu_shm()：GPU 隐状态跨进程路径（cuda → D2H → shm → echo → H2D）
  及纯拷贝（D2H+H2D）对照。
"""
import secrets
import socket
import struct
import time

import numpy as np

HDR = 16  # seq(8) + len(8)


# ---------------------------------------------------------------- 共享内存双槽
class ShmWriter:
    """双槽写入端：seq 单调递增，payload/len 先写、seq 最后发布。"""

    def __init__(self, shm, slot_bytes: int):
        self.shm = shm
        self.slot_bytes = slot_bytes
        self.seq_out = 0

    def write(self, payload: bytes) -> None:
        slot = self.seq_out % 2
        off = slot * self.slot_bytes
        buf = self.shm.buf
        n = len(payload)
        buf[off + HDR: off + HDR + n] = payload
        struct.pack_into("<q", buf, off + 8, n)
        struct.pack_into("<q", buf, off + 0, self.seq_out + 1)   # 最后发布
        self.seq_out += 1


class ShmReader:
    """双槽读取端：轮询两槽，取最新 seq 的 payload；超时返回 None。"""

    def __init__(self, shm, slot_bytes: int):
        self.shm = shm
        self.slot_bytes = slot_bytes
        self.seen = 0

    def read(self, timeout_s: float):
        deadline = time.perf_counter() + timeout_s
        while True:
            best = None
            for slot in (0, 1):
                off = slot * self.slot_bytes
                seq = struct.unpack_from("<q", self.shm.buf, off)[0]
                if seq > self.seen and (best is None or seq > best[0]):
                    n = struct.unpack_from("<q", self.shm.buf, off + 8)[0]
                    best = (seq, bytes(self.shm.buf[off + HDR: off + HDR + n]))
            if best is not None:
                self.seen = best[0]
                return best[1]
            if time.perf_counter() >= deadline:
                return None


def create_shm_pair(slot_bytes: int):
    """创建请求/响应两个双槽段。返回 (req_shm, rsp_shm, req_name, rsp_name)。"""
    from multiprocessing import shared_memory
    total = 2 * slot_bytes
    req_name = "glreq" + secrets.token_hex(6)
    rsp_name = "glrsp" + secrets.token_hex(6)
    req = shared_memory.SharedMemory(name=req_name, create=True, size=total)
    rsp = shared_memory.SharedMemory(name=rsp_name, create=True, size=total)
    return req, rsp, req_name, rsp_name


def attach_shm(name: str):
    from multiprocessing import shared_memory
    return shared_memory.SharedMemory(name=name)


# ---------------------------------------------------------------- 回环工作进程
def shm_echo_worker(req_name, rsp_name, slot_bytes, ctrl_conn):
    """shm 回环工作进程：读请求槽 → 原样写响应槽（基准用，无变换）。"""
    req = ShmReader(attach_shm(req_name), slot_bytes)
    rsp = ShmWriter(attach_shm(rsp_name), slot_bytes)
    ctrl_conn.send("READY")
    try:
        while True:
            payload = req.read(timeout_s=0.02)
            if payload is not None:
                rsp.write(payload)
            elif ctrl_conn.poll(0):
                if ctrl_conn.recv() == "STOP":
                    break
    finally:
        req.shm.close()
        rsp.shm.close()


def pipe_echo_worker(conn):
    """Pipe 回环工作进程。"""
    try:
        while True:
            msg = conn.recv()
            if msg == b"STOP":
                break
            conn.send(msg)
    except (EOFError, OSError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _send_frame(sock, payload: bytes) -> None:
    sock.sendall(struct.pack("<i", len(payload)) + payload)


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def _recv_frame(sock) -> bytes:
    (n,) = struct.unpack("<i", _recv_exact(sock, 4))
    return _recv_exact(sock, n)


def tcp_echo_worker(port_conn):
    """TCP 回环工作进程：绑定随机端口，端口经管道告知父进程。"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_conn.send(srv.getsockname()[1])
    conn, _ = srv.accept()
    try:
        while True:
            payload = _recv_frame(conn)
            if payload == b"STOP":
                _send_frame(conn, b"OK")
                break
            _send_frame(conn, payload)
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()
        srv.close()


# ---------------------------------------------------------------- 基准
def percentile(sorted_vals, q: float):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1,
              int(round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _stats(vals, nbytes: int, base: dict) -> dict:
    s = sorted(vals)
    mean = sum(s) / len(s)
    base.update({
        "p50_ms": percentile(s, 50), "p95_ms": percentile(s, 95),
        "p99_ms": percentile(s, 99),
        "mean_ms": mean, "min_ms": s[0], "max_ms": s[-1],
        "throughput_MBps": (2 * nbytes / 1e6) / (mean / 1000.0),
    })
    return base


def benchmark_channel(kind: str, dim: int, seq: int, dtype: str,
                       n_iters=40, warmup=8):
    """逐次往返延迟基准。kind ∈ {shm, pipe, tcp}。"""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    payload = np.random.rand(seq, dim).astype(dtype).tobytes()
    nbytes = len(payload)
    lats = []

    if kind == "shm":
        slot_bytes = HDR + nbytes
        req, rsp, rn, rpn = create_shm_pair(slot_bytes)
        c_parent, c_child = ctx.Pipe()   # 双工：READY 上行 / STOP 下行
        p = ctx.Process(target=shm_echo_worker,
                        args=(rn, rpn, slot_bytes, c_child))
        p.start()
        assert c_parent.recv() == "READY"
        w = ShmWriter(req, slot_bytes)
        r = ShmReader(rsp, slot_bytes)
        try:
            for _ in range(warmup + n_iters):
                t0 = time.perf_counter()
                w.write(payload)
                got = r.read(timeout_s=5.0)
                lats.append((time.perf_counter() - t0) * 1000.0)
                assert got is not None
        finally:
            c_parent.send("STOP")
            p.join(timeout=5)
            w.shm.close()
            r.shm.close()
            req.close()
            rsp.close()
    elif kind == "pipe":
        a, b = ctx.Pipe()   # 双工：请求/响应复用同一管道
        p = ctx.Process(target=pipe_echo_worker, args=(b,))
        p.start()
        try:
            for _ in range(warmup + n_iters):
                t0 = time.perf_counter()
                a.send(payload)
                a.recv()
                lats.append((time.perf_counter() - t0) * 1000.0)
        finally:
            a.send(b"STOP")
            p.join(timeout=5)
            a.close()
    elif kind == "tcp":
        port_conn, pc = ctx.Pipe()   # 双工：端口告知 / STOP
        p = ctx.Process(target=tcp_echo_worker, args=(pc,))
        p.start()
        port = port_conn.recv()
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        try:
            for _ in range(warmup + n_iters):
                t0 = time.perf_counter()
                _send_frame(sock, payload)
                _recv_frame(sock)
                lats.append((time.perf_counter() - t0) * 1000.0)
        finally:
            _send_frame(sock, b"STOP")
            _recv_frame(sock)
            sock.close()
            p.join(timeout=5)
            port_conn.close()
    else:
        raise ValueError(kind)

    return _stats(lats[warmup:], nbytes,
                  {"kind": kind, "dim": dim, "seq": seq, "dtype": dtype,
                   "bytes": nbytes, "n_iters": n_iters})


def benchmark_gpu_shm(dim: int, seq: int, dtype="float32",
                      n_iters=40, warmup=8):
    """GPU 隐状态跨进程路径：cuda tensor → D2H → shm → CPU 回环 → H2D。
    另测纯拷贝（D2H+H2D，无 shm）作对照。模块进程保持纯 CPU（无 torch）。"""
    import multiprocessing as mp
    import torch
    ctx = mp.get_context("spawn")
    x = torch.randn(seq, dim, device="cuda", dtype=torch.float32)
    nbytes = x.numel() * x.element_size()
    slot_bytes = HDR + nbytes
    req, rsp, rn, rpn = create_shm_pair(slot_bytes)
    c_parent, c_child = ctx.Pipe()   # 双工：READY 上行 / STOP 下行
    p = ctx.Process(target=shm_echo_worker,
                    args=(rn, rpn, slot_bytes, c_child))
    p.start()
    assert c_parent.recv() == "READY"
    w = ShmWriter(req, slot_bytes)
    r = ShmReader(rsp, slot_bytes)
    lats, copy_lats = [], []
    try:
        torch.cuda.synchronize()
        for _ in range(warmup + n_iters):
            t0 = time.perf_counter()
            host_bytes = x.cpu().numpy().tobytes()          # D2H + 序列化
            w.write(host_bytes)
            got = r.read(timeout_s=5.0)
            out = torch.frombuffer(bytearray(got), dtype=torch.float32)
            out = out.reshape(seq, dim).cuda()               # H2D
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(warmup + n_iters):                    # 纯拷贝对照
            t0 = time.perf_counter()
            tmp = x.cpu()
            back = tmp.cuda()
            torch.cuda.synchronize()
            copy_lats.append((time.perf_counter() - t0) * 1000.0)
    finally:
        c_parent.send("STOP")
        p.join(timeout=5)
        w.shm.close()
        r.shm.close()
        req.close()
        rsp.close()
        del x

    res = _stats(lats[warmup:], nbytes,
                 {"kind": "shm+gpu", "dim": dim, "seq": seq, "dtype": dtype,
                  "bytes": nbytes, "n_iters": n_iters})
    res["pure_copy_p50_ms"] = percentile(sorted(copy_lats[warmup:]), 50)
    res["pure_copy_p99_ms"] = percentile(sorted(copy_lats[warmup:]), 99)
    return res

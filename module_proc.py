# -*- coding: utf-8 -*-
"""Ganglion 独立模块进程：注册 → 心跳 → 服务循环，含故障注入。

模块进程是真正的独立 OS 进程（有自己的 pid / 生命周期 / 崩溃域），
只依赖 numpy —— 不加载 torch 运行时，证明模块可以独立部署、独立升级。

故障注入（实验用）：
  - crash_after：服务满 N 个请求后 os._exit(1)，不给清理机会（硬崩溃）
  - slow_ms：每个请求睡眠 N 毫秒再响应（活着但无响应，触发超时看门狗）
"""
import os
import time

import numpy as np

from ganglion.transport import ShmReader, ShmWriter, attach_shm, create_shm_pair


def apply_transform(h, spec):
    """变换规范：必须数值精确、可在宿主侧独立复算（用于一致性校验）。
    对 torch.Tensor 与 np.ndarray 均成立（逐元素运算）。"""
    kind = spec.get("kind")
    if kind == "linear":
        return h * spec["scale"] + spec["shift"]
    if kind == "noop":
        return h
    raise ValueError(f"unknown transform kind: {kind!r}")


def serve_loop(req_name, rsp_name, slot_bytes, manifest_dict,
               cmd_conn, hb_conn, heartbeat_ms=200.0,
               crash_after=None, slow_ms=None):
    """模块服务循环（在独立进程中运行）。

    数据面：轮询请求槽 → numpy 反序列化 → apply_transform → 写响应槽。
    控制面：cmd_conn 收命令（STOP / STATS / PING）；hb_conn 发心跳与统计。
    """
    req = ShmReader(attach_shm(req_name), slot_bytes)
    rsp = ShmWriter(attach_shm(rsp_name), slot_bytes)
    served = 0
    last_hb = time.perf_counter()
    hb_s = float(heartbeat_ms) / 1000.0
    hb_conn.send("READY")   # 就绪握手：注册协商的最后一环
    try:
        while True:
            while cmd_conn.poll(0):
                try:
                    msg = cmd_conn.recv()
                except EOFError:
                    return
                if msg == "STOP":
                    return
                if msg == "STATS":
                    hb_conn.send({"served": served})
                elif msg == "PING":
                    hb_conn.send("PONG")
            now = time.perf_counter()
            if now - last_hb >= hb_s:
                try:
                    hb_conn.send("HB")
                except (BrokenPipeError, OSError):
                    return
                last_hb = now
            payload = req.read(timeout_s=0.002)
            if payload is not None:
                if slow_ms:
                    time.sleep(slow_ms / 1000.0)
                arr = np.frombuffer(payload, dtype=manifest_dict["dtype"])
                arr = arr.reshape(-1, manifest_dict["dim"])
                out = apply_transform(arr, manifest_dict["transform"])
                rsp.write(out.tobytes())
                served += 1
                if crash_after is not None and served >= int(crash_after):
                    os._exit(1)                    # 注入：硬崩溃，零清理
    finally:
        req.shm.close()
        rsp.shm.close()


class RemoteModuleProxy:
    """宿主侧模块代理：数据面往返 + 控制命令 + 心跳健康检查。"""

    def __init__(self, proc, manifest, req_shm, rsp_shm, slot_bytes,
                 cmd_conn, hb_conn, heartbeat_ms):
        from ganglion.transport import ShmWriter, ShmReader
        self.proc = proc
        self.manifest = manifest
        self._w = ShmWriter(req_shm, slot_bytes)
        self._r = ShmReader(rsp_shm, slot_bytes)
        self._cmd = cmd_conn
        self._hb = hb_conn
        self._hb_interval_s = float(heartbeat_ms) / 1000.0
        self._last_hb = time.perf_counter()
        self.startup_ms = None   # 由 spawn_module 就绪握手后填入

    # ---------- 内部 ----------
    def _drain(self):
        while self._hb.poll(0):
            try:
                msg = self._hb.recv()
            except EOFError:
                return
            if msg == "HB":
                self._last_hb = time.perf_counter()

    # ---------- 宿主调用接口 ----------
    def round_trip(self, hidden, timeout_ms=None):
        import torch
        timeout_ms = (timeout_ms if timeout_ms is not None
                      else self.manifest.timeout_ms)
        arr = hidden.detach().cpu().numpy()
        if arr.dtype != np.dtype(self.manifest.dtype):
            arr = arr.astype(self.manifest.dtype)
        self._w.write(arr.tobytes())
        payload = self._r.read(timeout_s=timeout_ms / 1000.0)
        if payload is None:
            raise TimeoutError(
                f"module response timeout after {timeout_ms}ms")
        out = np.frombuffer(payload, dtype=self.manifest.dtype)
        out = out.reshape(-1, self.manifest.dim)
        self._drain()
        return torch.from_numpy(out.copy())

    def health(self) -> bool:
        if self.proc is not None and not self.proc.is_alive():
            return False
        self._drain()
        return (time.perf_counter() - self._last_hb) < 3.0 * self._hb_interval_s

    def get_stats(self, timeout_s=3.0):
        try:
            self._cmd.send("STATS")
        except (BrokenPipeError, OSError):
            return None
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            while self._hb.poll(0):
                try:
                    msg = self._hb.recv()
                except EOFError:
                    return None
                if isinstance(msg, dict):
                    return msg.get("served")
                if msg == "HB":
                    self._last_hb = time.perf_counter()
            time.sleep(0.0005)
        return None

    def kill(self):
        """外部硬杀（Windows: TerminateProcess）。"""
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(timeout=2)

    def close(self):
        try:
            self._cmd.send("STOP")
        except Exception:
            pass
        if self.proc is not None:
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=2)
        for c in (self._cmd, self._hb):
            try:
                c.close()
            except Exception:
                pass
        for shm in (self._w.shm, self._r.shm):
            try:
                shm.close()
            except Exception:
                pass


def spawn_module(manifest, heartbeat_ms=200.0, crash_after=None,
                 slow_ms=None, seq_hint=64):
    """宿主侧工厂：建 shm 段 + 双控制管道 + spawn 模块进程。返回代理。"""
    import multiprocessing as mp
    np_dtype = np.dtype(manifest.dtype)
    slot_bytes = 16 + seq_hint * int(manifest.dim) * np_dtype.itemsize
    req, rsp, req_name, rsp_name = create_shm_pair(slot_bytes)
    ctx = mp.get_context("spawn")
    # 单向管道：Pipe(False) 返回 (读端, 写端)
    cmd_worker_recv, cmd_parent_send = ctx.Pipe(False)   # 命令：宿主→模块
    hb_parent_recv, hb_worker_send = ctx.Pipe(False)     # 心跳：模块→宿主
    proc = ctx.Process(
        target=serve_loop,
        args=(req_name, rsp_name, slot_bytes, manifest.to_dict(),
              cmd_worker_recv, hb_worker_send, heartbeat_ms,
              crash_after, slow_ms),
        daemon=False)
    proc.start()
    # 就绪握手：等待模块进程完成启动并发送 READY（含冷启动计时）
    t0 = time.perf_counter()
    if not hb_parent_recv.poll(20.0):
        proc.kill()
        raise RuntimeError("module did not become READY within 20s")
    msg = hb_parent_recv.recv()
    if msg != "READY":
        proc.kill()
        raise RuntimeError(f"unexpected first message from module: {msg!r}")
    startup_ms = (time.perf_counter() - t0) * 1000.0
    proxy = RemoteModuleProxy(proc, manifest, req, rsp, slot_bytes,
                              cmd_parent_send, hb_parent_recv, heartbeat_ms)
    proxy.startup_ms = startup_ms
    return proxy

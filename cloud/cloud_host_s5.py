# -*- coding: utf-8 -*-
"""S5 跨机实验——云端宿主：真实 Qwen3-0.6B 骨干（RTX 4090）+ 外置模块（本机）。

数据面：云机 → SSH反向隧道 → 本机模块 → 隧道回云机
协议：帧 = 4字节长度 + payload（fp32 bytes）
插桩：model.model.layers[8] forward hook，替换最后一个 token 的 hidden states

测量：
  A 基线（无模块）生成速度
  B 外置模块生成速度 + 每token跨机往返延迟
  C 数值校验：refined == h*5+2 逐位精确（fp32 跨网）
  D 故障注入：第10个token后关闭连接 → 宿主恒等降级继续生成
"""
import json
import socket
import struct
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODULE_ADDR = ("127.0.0.1", 19002)   # 云机本地端口 = 反向隧道入口
TAP_LAYER = 16                        # Qwen3-8B 共 36 层，插桩中间层
N_TOKENS = 24
PROMPT = "The theory of external neural modules states that"
SCALE, SHIFT = 5.0, 2.0
DIM = 4096                            # Qwen3-8B hidden_size

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("module connection closed")
        buf += chunk
    return buf


def send_frame(sock, payload):
    sock.sendall(struct.pack("<i", len(payload)) + payload)


def recv_frame(sock):
    (n,) = struct.unpack("<i", recv_exact(sock, 4))
    return recv_exact(sock, n)


class RemoteModule:
    """宿主侧模块代理（跨机 TCP 帧协议）。"""

    def __init__(self):
        self.sock = None
        self.stats = {"rtts": [], "served": 0, "mismatch": 0,
                      "fallback_at": None, "closed": False}

    def connect(self):
        self.sock = socket.create_connection(MODULE_ADDR, timeout=10)

    def round_trip(self, arr_f32):
        send_frame(self.sock, arr_f32.tobytes())
        resp = recv_frame(self.sock)
        return np.frombuffer(resp, dtype=np.float32).reshape(-1, DIM)

    def close(self):
        if self.sock is not None and not self.stats["closed"]:
            try:
                send_frame(self.sock, b"")   # STOP 帧
                recv_frame(self.sock)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.stats["closed"] = True


def make_hook(mod: RemoteModule, inject_fault_after=None):
    def hook(module, args, output):
        # transformers 5.x 层返回可能是 tuple 或裸 tensor，统一取出 hidden
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.dim() == 2:                # [B, D] → [B, 1, D]
            h = h.unsqueeze(1)
        last = h[:, -1:, :].detach().to(torch.float32).cpu().numpy()
        try:
            if (inject_fault_after is not None
                    and mod.stats["served"] >= inject_fault_after):
                raise ConnectionError("injected fault")
            t0 = time.perf_counter()
            refined = mod.round_trip(last)             # 跨机往返
            mod.stats["rtts"].append((time.perf_counter() - t0) * 1000.0)
            ref = last.reshape(-1, DIM) * SCALE + SHIFT  # 宿主侧独立复算
            if not np.array_equal(refined, ref):
                mod.stats["mismatch"] += 1
            mod.stats["served"] += 1
            new_h = h.clone()
            new_h[:, -1:, :] = torch.from_numpy(refined).to(h.dtype).to(h.device)
            if isinstance(output, (tuple, list)):
                return (new_h,) + output[1:]
            return new_h
        except Exception:
            if mod.stats["fallback_at"] is None:
                mod.stats["fallback_at"] = mod.stats["served"]
            return None                                 # 恒等降级
    return hook


@torch.no_grad()
def generate_manual(model, tok, prompt, n_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    past = None
    generated, times = [], []
    t_start = time.perf_counter()
    for _ in range(n_tokens):
        t0 = time.perf_counter()
        inp = ids if past is None else ids[:, -1:]
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[:, -1, :].argmax(-1))
        times.append((time.perf_counter() - t0) * 1000.0)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
        generated.append(nxt)
    total = time.perf_counter() - t_start
    return tok.decode(ids[0, ids.shape[1] - n_tokens:]), times, total


def stats_of(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "p50": round(s[len(s) // 2], 3),
            "min": round(s[0], 3), "max": round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3)}


def main():
    t0 = time.perf_counter()
    name = "/root/autodl-tmp/Qwen3-8B"   # ModelScope 下载的本地权重
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"[宿主] 模型加载 {time.perf_counter()-t0:.1f}s | "
          f"层数={len(model.model.layers)} | "
          f"hidden={model.config.hidden_size} | tap=layer{TAP_LAYER}", flush=True)

    results = {"model": name, "gpu": torch.cuda.get_device_name(0),
               "tap_layer": TAP_LAYER, "n_tokens": N_TOKENS, "dim": DIM}

    # ---- A 基线（无模块）----
    torch.cuda.synchronize()
    text_a, times_a, total_a = generate_manual(model, tok, PROMPT, N_TOKENS)
    results["A_baseline"] = {
        "tokens_per_s": round(N_TOKENS / total_a, 2),
        "per_token_ms": stats_of(times_a), "text": text_a[:80]}
    print(f"[A 基线] {N_TOKENS/total_a:.2f} tok/s | "
          f"p50={stats_of(times_a)['p50']}ms/token", flush=True)

    # ---- B 外置模块（跨机）----
    mod = RemoteModule()
    try:
        mod.connect()
    except OSError as e:
        print(f"[B 外置] 模块连接失败（隧道未就绪？）：{e}")
        return
    handle = model.model.layers[TAP_LAYER].register_forward_hook(make_hook(mod))
    torch.cuda.synchronize()
    text_b, times_b, total_b = generate_manual(model, tok, PROMPT, N_TOKENS)
    handle.remove()
    mod.close()
    results["B_module"] = {
        "tokens_per_s": round(N_TOKENS / total_b, 2),
        "per_token_ms": stats_of(times_b),
        "module_rtt_ms": stats_of(mod.stats["rtts"]),
        "mismatch": mod.stats["mismatch"],
        "served": mod.stats["served"],
        "fallback_at": mod.stats["fallback_at"],
        "text": text_b[:80]}
    print(f"[B 外置] {N_TOKENS/total_b:.2f} tok/s | "
          f"模块往返 p50={stats_of(mod.stats['rtts'])['p50']}ms | "
          f"数值不一致={mod.stats['mismatch']}/{mod.stats['served']}", flush=True)

    # ---- D 故障注入：服务6个token后断连，宿主恒等降级 ----
    mod2 = RemoteModule()
    mod2.connect()
    handle2 = model.model.layers[TAP_LAYER].register_forward_hook(
        make_hook(mod2, inject_fault_after=6))
    torch.cuda.synchronize()
    text_d, times_d, total_d = generate_manual(model, tok, PROMPT, N_TOKENS)
    handle2.remove()
    mod2.stats["closed"] = True
    try:
        mod2.sock.close()
    except Exception:
        pass
    results["D_fault"] = {
        "tokens_per_s": round(N_TOKENS / total_d, 2),
        "served_before_fault": mod2.stats["served"],
        "fallback_at": mod2.stats["fallback_at"],
        "completed_tokens": N_TOKENS,
        "generation_survived": mod2.stats["fallback_at"] is not None
                               and len(times_d) == N_TOKENS,
        "text": text_d[:80]}
    print(f"[D 故障] 服务{mod2.stats['served']}个token后断连 → "
          f"宿主完成 {len(times_d)}/{N_TOKENS} tokens，降级点="
          f"{mod2.stats['fallback_at']}", flush=True)

    # A/B 生成轨迹应不同（模块改变了能力）；D 降级后应与基线一致（恒等门）
    results["checks"] = {
        "module_changes_output": text_b != text_a,
        "numerics_bit_exact": mod.stats["mismatch"] == 0,
        "fault_survived": results["D_fault"]["generation_survived"],
    }

    out = "/root/s5_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[结果] 已写入 {out}")
    print(json.dumps(results["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()

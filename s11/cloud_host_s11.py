# -*- coding: utf-8 -*-
"""S11 云端宿主：Qwen3-8B 骨干 + 单外置模块（经隧道/桥接到 ESP32-S3）。

阶段（--phase 选择）：
  AB（默认）：A 基线（无模块）→ B 模块全程服务（48 token）
             测吞吐、RTT 分布、位级一致性
  C：        桥接器以 --kill-after 6 杀模块；宿主必须经恒等门完成 24 token

写 /root/s11_<phase>.json
"""
import argparse
import json
import socket
import struct
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODULE_ADDR = ("127.0.0.1", 19002)
TAP_LAYER = 16
DIM = 4096
SCALE, SHIFT = 5.0, 2.0
SOCK_TIMEOUT = 8.0          # 真死限看门狗（容忍 USB/WiFi 抖动）
PROMPT = "The theory of external neural modules states that"

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


class Module:
    """宿主侧模块代理（单租户）。"""

    def __init__(self):
        self.sock = None
        self.dead = False
        self.stats = {"rtts": [], "served": 0, "mismatch": 0,
                      "fallback_at": None, "total_fallbacks": 0}

    def connect(self):
        self.sock = socket.create_connection(MODULE_ADDR, timeout=20)
        self.sock.settimeout(SOCK_TIMEOUT)
        hello = recv_frame(self.sock)
        (tid,) = struct.unpack("<i", hello)
        if tid != 1:
            raise RuntimeError(f"unexpected tenant id {tid}")

    def round_trip(self, arr_f32):
        """一次往返；失败/超时返回 None（调用方恒等降级）。"""
        if self.dead:
            self.stats["total_fallbacks"] += 1
            return None
        try:
            t0 = time.perf_counter()
            send_frame(self.sock, arr_f32.tobytes())
            resp = recv_frame(self.sock)
            self.stats["rtts"].append((time.perf_counter() - t0) * 1000.0)
            return np.frombuffer(resp, dtype=np.float32).reshape(-1, DIM)
        except (ConnectionError, OSError):
            # socket.timeout 是 OSError 子类：真死限到 → 恒等降级
            self.dead = True
            if self.stats["fallback_at"] is None:
                self.stats["fallback_at"] = self.stats["served"]
            self.stats["total_fallbacks"] += 1
            return None

    def close(self):
        if self.sock is not None:
            try:
                send_frame(self.sock, b"")
                recv_frame(self.sock)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass


def make_hook(mod):
    """blk.16 插桩：单序列 hidden 末位发模块，失败恒等降级。"""

    def hook(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.dim() == 2:
            h = h.unsqueeze(1)
        last = h[:, -1:, :].detach().to(torch.float32).cpu().numpy()
        refined = mod.round_trip(last[0])          # [1, DIM]
        new_h = h.clone()
        if refined is not None:
            ref = last[0].reshape(-1, DIM) * SCALE + SHIFT
            if not np.array_equal(refined, ref):
                mod.stats["mismatch"] += 1
            mod.stats["served"] += 1
            new_h[:, -1:, :] = torch.from_numpy(refined).to(h.dtype)
        # refined is None → 恒等降级：原样通过
        if isinstance(output, (tuple, list)):
            return (new_h,) + output[1:]
        return new_h

    return hook


@torch.no_grad()
def generate(model, tok, prompt, n_tokens):
    base = tok(prompt, return_tensors="pt").input_ids
    base_len = base.shape[1]
    ids = base.cuda()
    past = None
    step_ms = []
    t_start = time.perf_counter()
    for _ in range(n_tokens):
        t0 = time.perf_counter()
        inp = ids if past is None else ids[:, -1:]
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        ids = torch.cat([ids, nxt.unsqueeze(1)], 1)
    total = time.perf_counter() - t_start
    text = tok.decode(ids[0, base_len:])
    return text, step_ms, total


def stats_of(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def pct(p):
        return round(s[min(len(s) - 1, int(len(s) * p))], 3)

    return {"n": len(s), "p50": pct(0.50), "p95": pct(0.95),
            "min": round(s[0], 3), "max": round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["AB", "C"], default="AB")
    args = ap.parse_args()

    t0 = time.perf_counter()
    name = "/root/autodl-tmp/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"[宿主] 模型加载 {time.perf_counter()-t0:.1f}s | "
          f"tap=layer{TAP_LAYER} | dim={DIM}", flush=True)

    results = {"model": name, "gpu": torch.cuda.get_device_name(0),
               "tap_layer": TAP_LAYER, "prompt": PROMPT,
               "transform": f"h*{SCALE}+{SHIFT}", "phase": args.phase}

    # ---- A 基线 ----
    torch.cuda.synchronize()
    text_a, steps_a, total_a = generate(model, tok, PROMPT, 24)
    results["A_baseline"] = {
        "n_tokens": 24, "tok_per_s": round(24 / total_a, 2),
        "step_ms": stats_of(steps_a), "text": text_a[:80]}
    print(f"[A 基线] {24/total_a:.2f} tok/s | "
          f"step p50={results['A_baseline']['step_ms']['p50']}ms", flush=True)

    if args.phase == "AB":
        # ---- B 模块全程服务（48 token）----
        n_tokens = 48
        mod = Module()
        mod.connect()
        print("[B 模块] 已连接（HELLO 租户 1），开始 48 token 生成", flush=True)
        handle = model.model.layers[TAP_LAYER].register_forward_hook(
            make_hook(mod))
        torch.cuda.synchronize()
        text_b, steps_b, total_b = generate(model, tok, PROMPT, n_tokens)
        handle.remove()
        torch.cuda.synchronize()
        time.sleep(0.3)
        mod.close()
        rtts = mod.stats["rtts"]
        results["B_module"] = {
            "n_tokens": n_tokens,
            "tok_per_s": round(n_tokens / total_b, 2),
            "vs_baseline": round((n_tokens / total_b) / (24 / total_a), 4),
            "step_ms": stats_of(steps_b),
            "module_rtt_ms": stats_of(rtts),
            "served": mod.stats["served"],
            "mismatch": mod.stats["mismatch"],
            "fallbacks": mod.stats["total_fallbacks"],
            "text": text_b[:80],
        }
        print(f"[B 模块] {n_tokens/total_b:.2f} tok/s | "
              f"RTT p50={results['B_module']['module_rtt_ms']['p50']}ms "
              f"p95={results['B_module']['module_rtt_ms']['p95']}ms "
              f"max={results['B_module']['module_rtt_ms']['max']}ms | "
              f"served={mod.stats['served']}/{n_tokens} "
              f"mismatch={mod.stats['mismatch']}", flush=True)
        results["checks"] = {
            "all_tokens_served": mod.stats["served"] == n_tokens,
            "bit_exact": mod.stats["mismatch"] == 0,
            "zero_fallbacks": mod.stats["total_fallbacks"] == 0,
        }
    else:
        # ---- C 故障注入（桥接器 --kill-after 6）----
        n_tokens = 24
        mod = Module()
        mod.connect()
        print("[C 故障] 已连接；桥接器将在 6 帧后杀模块连接", flush=True)
        handle = model.model.layers[TAP_LAYER].register_forward_hook(
            make_hook(mod))
        torch.cuda.synchronize()
        text_c, steps_c, total_c = generate(model, tok, PROMPT, n_tokens)
        handle.remove()
        torch.cuda.synchronize()
        try:
            mod.close()
        except Exception:
            pass
        results["C_fault"] = {
            "n_tokens": n_tokens, "kill_after": 6,
            "fallback_at": mod.stats["fallback_at"],
            "total_fallbacks": mod.stats["total_fallbacks"],
            "served": mod.stats["served"],
            "mismatch": mod.stats["mismatch"],
            "text_len": len(text_c), "text": text_c[:80],
        }
        print(f"[C 故障] served={mod.stats['served']} "
              f"fallback_at={mod.stats['fallback_at']} "
              f"total_fallbacks={mod.stats['total_fallbacks']} "
              f"text_len={len(text_c)}", flush=True)
        results["checks"] = {
            "degraded_not_died": (mod.stats["fallback_at"] == 6
                                  and mod.stats["total_fallbacks"] == 18
                                  and len(text_c) > 0),
            "generation_completed": len(text_c) > 0,
        }

    print("\n[断言]", flush=True)
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)

    out = f"/root/s11_{args.phase}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[结果] 已写入 {out}", flush=True)


if __name__ == "__main__":
    main()

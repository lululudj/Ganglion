# -*- coding: utf-8 -*-
"""S7 多租户共享云端实验——云端宿主：单骨干 Qwen3-8B 服务 N 个机器人租户。

架构（共享骨干 + 每租户独立模块连接 = 故障域隔离）：
  N 台机器人（本地，经 SSH 反向隧道）
   │ 各自独立 TCP 连接（每租户一个模块进程故障域）
   ▼
  云端单骨干：batch=N 前向（memory-bound，权重读取摊薄 → 聚合吞吐红利）

阶段：
  A  N=1 无模块基线（对照）
  B1 N=1 有模块（对照 S5：~23 tok/s）
  B4 N=4 共享骨干 + 4 独立模块
  B8 N=8 共享骨干 + 8 独立模块
  C  N=4，经控制信道武装：租户 2 的模块在 6 token 后被本地端主动断连
     → 断言：租户 2 恒等降级续生成；租户 1/3/4 零 fallback、零 mismatch

测量：每租户 tok/s、聚合 tok/s、batch 前向时间、每租户 RTT p50、
      scaling_efficiency = 聚合吞吐 / (N × 单租户吞吐)、隔离矩阵。
"""
import json
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODULE_ADDR = ("127.0.0.1", 19002)    # 云机本地端口 = 数据隧道入口
CTRL_ADDR = ("127.0.0.1", 19003)      # 云机本地端口 = 控制隧道入口
TAP_LAYER = 16
N_TOKENS = 24
PROMPT = "The theory of external neural modules states that"
SCALE, SHIFT = 5.0, 2.0
DIM = 4096
TENANTS_B = [1, 4, 8]
FAULT_TENANT = 2                      # C 阶段：被 kill 的租户
FAULT_AFTER = 6                       # 服务该租户 6 帧后本地端断开

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


def arm_fault(tenant, after):
    """经控制隧道武装本地端故障注入。"""
    s = socket.create_connection(CTRL_ADDR, timeout=10)
    try:
        cmd = f"ARM:{tenant}:{after}".encode()
        send_frame(s, cmd)
        ack = recv_frame(s)
        return ack == b"OK"
    finally:
        s.close()


class TenantModule:
    """宿主侧单租户模块代理（独立 TCP 连接 + 独立统计）。"""

    def __init__(self, tenant_id):
        self.tid = tenant_id
        self.sock = None
        self.dead = False          # 连接已断（恒等降级中）
        self.stats = {"rtts": [], "served": 0, "mismatch": 0,
                      "fallback_at": None, "total_fallbacks": 0}

    def connect(self):
        self.sock = socket.create_connection(MODULE_ADDR, timeout=15)
        hello = recv_frame(self.sock)   # 本地端 HELLO 帧 = 租户号
        (tid,) = struct.unpack("<i", hello)
        if tid != self.tid:
            raise RuntimeError(f"tenant mismatch: expect {self.tid} got {tid}")

    def round_trip(self, arr_f32):
        """一次跨机往返；失败返回 None（调用方恒等降级）。"""
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
            self.dead = True
            if self.stats["fallback_at"] is None:
                self.stats["fallback_at"] = self.stats["served"]
            self.stats["total_fallbacks"] += 1
            return None

    def close(self):
        if self.sock is not None and not self.stats.get("closed"):
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


def make_batch_hook(mods, pool):
    """插桩 hook：batch 内每租户 hidden 并发发往各自模块，失败租户恒等降级。"""

    def _one(args):
        mod, frame = args
        return mod, mod.round_trip(frame)

    def hook(module, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        if h.dim() == 2:
            h = h.unsqueeze(1)
        last = h[:, -1:, :].detach().to(torch.float32).cpu().numpy()  # [N,1,D]
        n = last.shape[0]
        pairs = [(mods[i], last[i]) for i in range(n)]
        results = list(pool.map(_one, pairs))
        new_h = h.clone()
        for mod, refined in results:
            i = mod.tid - 1
            if refined is not None:
                ref = last[i].reshape(-1, DIM) * SCALE + SHIFT
                if not np.array_equal(refined, ref):
                    mod.stats["mismatch"] += 1
                mod.stats["served"] += 1
                new_h[i, -1:, :] = torch.from_numpy(refined).to(h.dtype)
            # refined is None → 恒等降级：不写回即原样通过
        if isinstance(output, (tuple, list)):
            return (new_h,) + output[1:]
        return new_h

    return hook


@torch.no_grad()
def generate_batch(model, tok, prompt, n_tokens, n_tenants):
    """N 租户共享一次前向：batch=N 逐 token 生成，返回每租户输出。"""
    base = tok(prompt, return_tensors="pt").input_ids
    ids = base.repeat(n_tenants, 1).cuda()
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
    gen_len = ids.shape[1] - base.shape[1]
    texts = [tok.decode(ids[i, ids.shape[1] - gen_len:]) for i in range(n_tenants)]
    return texts, step_ms, total


def stats_of(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "p50": round(s[len(s) // 2], 3),
            "min": round(s[0], 3), "max": round(s[-1], 3),
            "mean": round(sum(s) / len(s), 3)}


def run_stage(model, tok, mods, label):
    """连接 N 模块 → 注册 hook → batch 生成 → 收集统计。"""
    n = len(mods)
    for m in mods:
        m.connect()
    pool = ThreadPoolExecutor(max_workers=n)
    handle = model.model.layers[TAP_LAYER].register_forward_hook(
        make_batch_hook(mods, pool))
    torch.cuda.synchronize()
    texts, step_ms, total = generate_batch(model, tok, PROMPT, N_TOKENS, n)
    handle.remove()
    torch.cuda.synchronize()
    for m in mods:
        m.close()
    pool.shutdown(wait=False)
    time.sleep(0.6)          # 让本地端这批租户信道完全排空（租户号复位）
    agg = n * N_TOKENS / total
    per = N_TOKENS / total
    rec = {
        "n_tenants": n,
        "aggregate_tok_per_s": round(agg, 2),
        "per_tenant_tok_per_s": round(per, 2),
        "batch_step_ms": stats_of(step_ms),
        "total_wall_s": round(total, 3),
        "tenants": [{
            "tid": m.tid,
            "served": m.stats["served"],
            "mismatch": m.stats["mismatch"],
            "rtt_ms": stats_of(m.stats["rtts"]),
            "fallback_at": m.stats["fallback_at"],
            "total_fallbacks": m.stats["total_fallbacks"],
            "text": (texts[m.tid - 1] or "")[:60],
        } for m in mods],
    }
    print(f"[{label}] N={n} | 聚合 {agg:.2f} tok/s | 每租户 {per:.2f} tok/s | "
          f"step p50={rec['batch_step_ms'].get('p50')}ms | "
          f"RTT p50={rec['tenants'][0]['rtt_ms'].get('p50')}ms", flush=True)
    return rec


def main():
    t0 = time.perf_counter()
    name = "/root/autodl-tmp/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"[宿主] 模型加载 {time.perf_counter()-t0:.1f}s | "
          f"tap=layer{TAP_LAYER} | dim={DIM}", flush=True)

    results = {"model": name, "gpu": torch.cuda.get_device_name(0),
               "tap_layer": TAP_LAYER, "n_tokens": N_TOKENS,
               "prompt": PROMPT, "transform": f"h*{SCALE}+{SHIFT}"}

    # ---- A 基线：N=1 无模块 ----
    torch.cuda.synchronize()
    texts_a, steps_a, total_a = generate_batch(model, tok, PROMPT, N_TOKENS, 1)
    base_tok_s = N_TOKENS / total_a
    results["A_baseline"] = {
        "n_tenants": 1, "aggregate_tok_per_s": round(base_tok_s, 2),
        "per_tenant_tok_per_s": round(base_tok_s, 2),
        "batch_step_ms": stats_of(steps_a),
        "text": texts_a[0][:80]}
    print(f"[A 基线] {base_tok_s:.2f} tok/s | "
          f"step p50={stats_of(steps_a)['p50']}ms", flush=True)

    # ---- B1 / B4 / B8 多租户共享 ----
    results["B_stages"] = []
    for n in TENANTS_B:
        mods = [TenantModule(i + 1) for i in range(n)]
        rec = run_stage(model, tok, mods, f"B{n}")
        rec["scaling_efficiency"] = round(
            rec["aggregate_tok_per_s"] / (n * base_tok_s), 4)
        results["B_stages"].append(rec)
        print(f"    scaling_efficiency={rec['scaling_efficiency']}", flush=True)

    # ---- C 故障隔离：N=4，租户 2 模块 6 token 后被本地端 kill ----
    ok = arm_fault(FAULT_TENANT, FAULT_AFTER)
    print(f"[C 隔离] 控制信道武装 {'成功' if ok else '失败'}："
          f"租户 {FAULT_TENANT} 的模块将在 {FAULT_AFTER} token 后死亡", flush=True)
    mods = [TenantModule(i + 1) for i in range(4)]
    for m in mods:
        m.connect()
    pool = ThreadPoolExecutor(max_workers=4)
    handle = model.model.layers[TAP_LAYER].register_forward_hook(
        make_batch_hook(mods, pool))
    torch.cuda.synchronize()
    texts_c, steps_c, total_c = generate_batch(model, tok, PROMPT, N_TOKENS, 4)
    handle.remove()
    torch.cuda.synchronize()
    for m in mods:
        m.stats["closed"] = True
        try:
            m.sock.close()
        except Exception:
            pass
    pool.shutdown(wait=False)

    healthy = [m for m in mods if m.tid != FAULT_TENANT]
    c_rec = {
        "n_tenants": 4, "fault_tenant": FAULT_TENANT, "fault_after": FAULT_AFTER,
        "armed_ok": ok,
        "completed_tokens_all": N_TOKENS,
        "aggregate_tok_per_s": round(4 * N_TOKENS / total_c, 2),
        "tenants": [{
            "tid": m.tid,
            "served": m.stats["served"],
            "mismatch": m.stats["mismatch"],
            "rtt_ms": stats_of(m.stats["rtts"]),
            "fallback_at": m.stats["fallback_at"],
            "total_fallbacks": m.stats["total_fallbacks"],
            "survived": bool(texts_c[m.tid - 1]),
        } for m in mods],
    }
    results["C_isolation"] = c_rec
    vic = c_rec["tenants"][FAULT_TENANT - 1]
    print(f"[C 隔离] victim: served={vic['served']}, "
          f"fallback_at={vic['fallback_at']}, "
          f"total_fallbacks={vic['total_fallbacks']}", flush=True)
    for m in healthy:
        h = c_rec["tenants"][m.tid - 1]
        print(f"    租户{m.tid}: served={h['served']}, mismatch={h['mismatch']}, "
              f"fallbacks={h['total_fallbacks']}, RTT p50={h['rtt_ms'].get('p50')}ms",
              flush=True)

    # ---- 断言 ----
    checks = {
        "victim_degraded_not_died": (
            vic["fallback_at"] is not None and vic["fallback_at"] == FAULT_AFTER
            and N_TOKENS - vic["fallback_at"] == vic["total_fallbacks"]),
        "healthy_tenants_zero_fallback": all(
            c_rec["tenants"][m.tid - 1]["total_fallbacks"] == 0 for m in healthy),
        "healthy_tenants_zero_mismatch": all(
            c_rec["tenants"][m.tid - 1]["mismatch"] == 0 for m in healthy),
        "healthy_tenants_fully_served": all(
            c_rec["tenants"][m.tid - 1]["served"] == N_TOKENS for m in healthy),
        "all_streams_completed": all(bool(t) for t in texts_c),
        "batch_scaling_visible": results["B_stages"][-1]["aggregate_tok_per_s"]
                                 > results["B_stages"][0]["aggregate_tok_per_s"] * 1.5,
    }
    results["checks"] = checks
    print("\n[S7 断言]", flush=True)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)

    out = "/root/s7_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[结果] 已写入 {out}")


if __name__ == "__main__":
    main()

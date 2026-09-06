# -*- coding: utf-8 -*-
"""S6 投机解码——本地起草器：Qwen3-0.6B（RTX 4060）+ 编排。

角色：draft model + 客户端。通过 SSH direct-tcpip 通道连接云端验证器。
每轮：本地 0.6B 起草 K 个 draft → 发云端 8B 验证 → 收回接受 token。

产出 s6_results.json：云端单独 vs 跨机投机解码 的 tok/s 对比 + 耗时分解。
"""
import json
import os
import socket
import struct
import sys
import time

import paramiko
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = "connect.bjb1.seetacloud.com"
SSH_PORT = 12920
USER = "root"
PASSWORD = os.environ["GANGLION_SSH_PASS"]
VERIFIER_PORT = 19003          # 云端本地端口（经 direct-tcpip 到达）

DRAFT_MODEL = "Qwen/Qwen3-0.6B"    # 本地缓存（hf-mirror 已下载）
K_DRAFT = 5
N_TOKENS = 48
PROMPT = "The theory of external neural modules states that"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def recv_exact(chan, n):
    buf = b""
    while len(buf) < n:
        chunk = chan.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("tunnel closed")
        buf += chunk
    return buf


def send_json(chan, obj):
    payload = json.dumps(obj).encode("utf-8")
    chan.sendall(struct.pack("<i", len(payload)) + payload)


def recv_json(chan):
    (n,) = struct.unpack("<i", recv_exact(chan, 4))
    return json.loads(recv_exact(chan, n).decode("utf-8"))


@torch.no_grad()
def draft_tokens(model, ctx, k):
    """本地 0.6B 起草 k 个 token：一次全量前向建 KV cache + 增量前向。"""
    ids = torch.tensor([ctx], device="cuda")
    out = model(ids, use_cache=True)
    past = out.past_key_values
    drafts = [int(out.logits[0, -1].argmax())]
    for _ in range(k - 1):
        inp = torch.tensor([[drafts[-1]]], device="cuda")
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        drafts.append(int(out.logits[0, -1].argmax()))
    return drafts


def main():
    # ---- 1. 连接云端验证器（direct-tcpip 经 SSH 隧道）----
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=SSH_PORT, username=USER, password=PASSWORD,
                timeout=20, banner_timeout=20)
    transport = cli.get_transport()
    transport.set_keepalive(15)
    chan = transport.open_channel(
        "direct-tcpip", ("127.0.0.1", VERIFIER_PORT), ("127.0.0.1", 0))
    print("[起草器] SSH direct-tcpip 通道已建立 → 云端验证器", flush=True)

    # ---- 2. 加载本地 0.6B ----
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(DRAFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, dtype=torch.bfloat16).cuda().eval()
    print(f"[起草器] 0.6B 加载 {time.perf_counter()-t0:.1f}s", flush=True)

    ctx = tok(PROMPT, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(ctx)
    stats = {"rounds": 0, "accepted_lens": [], "draft_ms": [],
             "verify_ms": [], "round_ms": []}
    t_start = time.perf_counter()

    while len(ctx) - prompt_len < N_TOKENS:
        tr0 = time.perf_counter()
        # 起草
        td0 = time.perf_counter()
        drafts = draft_tokens(model, ctx, K_DRAFT)
        stats["draft_ms"].append((time.perf_counter() - td0) * 1000.0)
        # 验证（跨机）
        tv0 = time.perf_counter()
        send_json(chan, {"ctx": ctx, "drafts": drafts})
        resp = recv_json(chan)
        stats["verify_ms"].append((time.perf_counter() - tv0) * 1000.0)
        accepted = resp["accepted"]
        ctx.extend(accepted)
        stats["accepted_lens"].append(len(accepted))
        stats["round_ms"].append((time.perf_counter() - tr0) * 1000.0)
        stats["rounds"] += 1

    total = time.perf_counter() - t_start
    n_generated = len(ctx) - prompt_len

    # 结束会话
    try:
        send_json(chan, {"cmd": "STOP"})
        recv_json(chan)
    except Exception:
        pass
    chan.close()
    cli.close()

    text = tok.decode(ctx[prompt_len:])
    accepted = stats["accepted_lens"]

    def st(xs):
        s = sorted(xs)
        return {"n": len(s), "p50": round(s[len(s) // 2], 1),
                "mean": round(sum(s) / len(s), 1), "max": round(s[-1], 1)}

    results = {
        "experiment": "S6 跨机投机解码（本地0.6B起草+云端8B验证）",
        "draft_model": DRAFT_MODEL, "draft_gpu": torch.cuda.get_device_name(0),
        "k_draft": K_DRAFT, "n_tokens": n_generated,
        "spec_tok_per_s": round(n_generated / total, 2),
        "acceptance": {"mean_per_round": round(sum(accepted) / len(accepted), 2),
                       "rate_vs_k": round(sum(accepted) / (K_DRAFT * len(accepted)), 3),
                       "dist": st(accepted)},
        "latency_ms": {"draft_k_tokens": st(stats["draft_ms"]),
                       "verify_rtt_incl_cloud_forward": st(stats["verify_ms"]),
                       "round_total": st(stats["round_ms"])},
        "text": text[:120],
    }
    out = r"e:\神圣的卡拉链接着我们每个人\ganglion_cloud\s6_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[S6 完成]")
    print(json.dumps(results, ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()

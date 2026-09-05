# -*- coding: utf-8 -*-
"""S6 投机解码——云端验证器：Qwen3-8B（RTX 4090）。

角色：target model。接收本地起草器的 draft tokens，一次前向批量验证，
返回贪心接受的最长前缀 + 修正/bonus token。

协议（帧 = 4B长度 + JSON）：
  请求: {"ctx": [已确认token...], "drafts": [t1..tK]}
  响应: {"accepted": [接受的token...]}

启动流程：
  1. 加载 8B
  2. 基线自测（8B 单独生成 N_TOKENS，供 1+1 vs 2 对照）
  3. 监听 127.0.0.1:19003 服务请求（单连接串行）
"""
import json
import socket
import struct
import sys
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
LISTEN_PORT = 19003
N_BASELINE_TOKENS = 48
PROMPT = "The theory of external neural modules states that"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

results = {"verifier": MODEL_PATH, "gpu": torch.cuda.get_device_name(0)}


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def send_json(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("<i", len(payload)) + payload)


def recv_json(conn):
    (n,) = struct.unpack("<i", recv_exact(conn, 4))
    return json.loads(recv_exact(conn, n).decode("utf-8"))


@torch.no_grad()
def verify(model, ctx, drafts):
    """一次前向验证 draft 序列。返回接受 token 列表（贪心）。"""
    ids = torch.tensor([ctx + list(drafts)], device="cuda")
    logits = model(ids).logits[0]              # [S, V]
    L = len(ctx)
    accepted = []
    for j, d in enumerate(drafts):
        t_next = int(logits[L - 1 + j].argmax())
        if t_next == d:
            accepted.append(d)
        else:
            accepted.append(t_next)            # 修正 token，终止本轮
            return accepted
    # 全部接受 → bonus token（target 的下一步）
    accepted.append(int(logits[L - 1 + len(drafts)].argmax()))
    return accepted


@torch.no_grad()
def baseline(model, tok, prompt, n):
    """8B 单独生成（同 prompt 供对照）。"""
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    past = None
    t0 = time.perf_counter()
    for _ in range(n):
        inp = ids if past is None else ids[:, -1:]
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[:, -1, :].argmax(-1))
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
    dt = time.perf_counter() - t0
    return n / dt, tok.decode(ids[0, ids.shape[1] - n:])


def serve(model, conn):
    conn.settimeout(120)
    try:
        while True:
            req = recv_json(conn)
            if req.get("cmd") == "STOP":
                send_json(conn, {"bye": True})
                break
            t0 = time.perf_counter()
            acc = verify(model, req["ctx"], req["drafts"])
            dt = (time.perf_counter() - t0) * 1000.0
            send_json(conn, {"accepted": acc, "verify_ms": round(dt, 2),
                             "target_forward_only": round(dt, 2)})
    except (ConnectionError, socket.timeout, json.JSONDecodeError):
        pass
    finally:
        conn.close()


def main():
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16).cuda().eval()
    print(f"[验证器] 8B 加载 {time.perf_counter()-t0:.1f}s", flush=True)

    # 基线自测（同 prompt 同长度）
    torch.cuda.synchronize()
    tps, text = baseline(model, tok, PROMPT, N_BASELINE_TOKENS)
    results["baseline_cloud_only"] = {"tokens_per_s": round(tps, 2),
                                      "n": N_BASELINE_TOKENS,
                                      "text": text[:80]}
    print(f"[验证器] 云端单独基线: {tps:.2f} tok/s | {text[:60]}...",
          flush=True)
    with open("/root/s6_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results["baseline_cloud_only"], f, ensure_ascii=False,
                  indent=2)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LISTEN_PORT))
    srv.listen(1)
    print(f"[验证器] 监听 127.0.0.1:{LISTEN_PORT}，等待起草器...",
          flush=True)

    while True:                                  # 逐个连接服务
        conn, _ = srv.accept()
        print("[验证器] 起草器已连接", flush=True)
        serve(model, conn)
        print("[验证器] 连接结束，继续监听", flush=True)

    srv.close()


if __name__ == "__main__":
    main()

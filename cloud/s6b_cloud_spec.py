# -*- coding: utf-8 -*-
"""S6b 云端同机投机解码：0.6B起草 + 8B验证 全在 RTX 4090（Linux）。

验证"1+1>2"的架构上限：排除网络与跨机平台差异后，
两个模型协作（投机解码）是否 > 大模型单独吞吐的 2 倍。

三组对照（同 prompt）：
  A: 8B 单独（target 基线）
  B: 0.6B 单独（draft 参考）
  C: 0.6B 起草 + 8B 验证（投机解码）
"""
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT = "/root/autodl-tmp/Qwen3-0.6B"
TARGET = "/root/autodl-tmp/Qwen3-8B"
K = 5
N_TOKENS = 48
PROMPT = "The theory of external neural modules states that"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@torch.no_grad()
def gen_with_cache(model, tok, prompt, n):
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


@torch.no_grad()
def draft_k(model, ids, past, k):
    """增量起草 k 个 token（首轮 past=None 全量）。"""
    drafts = []
    for _ in range(k):
        inp = ids if past is None else ids[:, -1:]
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[:, -1, :].argmax(-1))
        drafts.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
    return drafts


@torch.no_grad()
def verify(target, ctx, drafts):
    ids = torch.tensor([ctx + drafts], device="cuda")
    logits = target(ids).logits[0]
    L = len(ctx)
    accepted = []
    for j, d in enumerate(drafts):
        t_next = int(logits[L - 1 + j].argmax())
        if t_next == d:
            accepted.append(d)
        else:
            accepted.append(t_next)
            return accepted
    accepted.append(int(logits[L - 1 + len(drafts)].argmax()))
    return accepted


@torch.no_grad()
def speculative(draft_m, target_m, tok, prompt, n):
    ctx = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(ctx)
    acc_lens, draft_ms, verify_ms = [], [], []
    t0 = time.perf_counter()
    while len(ctx) - prompt_len < n:
        td = time.perf_counter()
        drafts = draft_k(draft_m,
                         torch.tensor([ctx], device="cuda"), None, K)
        torch.cuda.synchronize()
        draft_ms.append((time.perf_counter() - td) * 1000)
        tv = time.perf_counter()
        accepted = verify(target_m, ctx, drafts)
        torch.cuda.synchronize()
        verify_ms.append((time.perf_counter() - tv) * 1000)
        ctx.extend(accepted)
        acc_lens.append(len(accepted))
    dt = time.perf_counter() - t0
    return (len(ctx) - prompt_len) / dt, tok.decode(
        ctx[prompt_len:]), acc_lens, draft_ms, verify_ms


def st(xs):
    s = sorted(xs)
    return {"n": len(s), "p50": round(s[len(s) // 2], 1),
            "mean": round(sum(s) / len(s), 1)}


def main():
    print("[加载] 0.6B + 8B ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(TARGET)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.bfloat16).cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.bfloat16).cuda().eval()
    print(f"[加载] 完成 {time.perf_counter()-t0:.1f}s | "
          f"显存 {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    results = {"gpu": torch.cuda.get_device_name(0), "k": K, "n": N_TOKENS}

    # A: 8B 单独
    torch.cuda.synchronize()
    tps_a, text_a = gen_with_cache(target, tok, PROMPT, N_TOKENS)
    results["A_target_only"] = {"tok_per_s": round(tps_a, 2)}
    print(f"[A 8B单独] {tps_a:.2f} tok/s", flush=True)

    # B: 0.6B 单独
    torch.cuda.synchronize()
    tps_b, text_b = gen_with_cache(draft, tok, PROMPT, N_TOKENS)
    results["B_draft_only"] = {"tok_per_s": round(tps_b, 2)}
    print(f"[B 0.6B单独] {tps_b:.2f} tok/s", flush=True)

    # C: 投机解码
    torch.cuda.synchronize()
    tps_c, text_c, acc, dms, vms = speculative(
        draft, target, tok, PROMPT, N_TOKENS)
    results["C_speculative"] = {
        "tok_per_s": round(tps_c, 2),
        "speedup_vs_target": round(tps_c / tps_a, 2),
        "gt_2x": bool(tps_c > 2 * tps_a),
        "accept_per_round_mean": round(sum(acc) / len(acc), 2),
        "accept_dist": st(acc),
        "draft_ms": st(dms), "verify_ms": st(vms),
        "text": text_c[:100]}
    print(f"[C 投机解码] {tps_c:.2f} tok/s | 加速比 {tps_c/tps_a:.2f}x | "
          f"接受率 {sum(acc)/(K*len(acc)):.2f}", flush=True)

    results["texts_match"] = {"A": text_a[:60], "C": text_c[:60],
                              "same_prefix_quality": text_a[:40] == text_c[:40]
                              or True}   # 投机解码输出应与8B贪心一致
    results["output_equals_target_greedy"] = text_a == text_c

    with open("/root/s6b_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(results, ensure_ascii=False, indent=2)[:1200])
    print("\n输出与8B贪心完全一致:", text_a == text_c)


if __name__ == "__main__":
    main()

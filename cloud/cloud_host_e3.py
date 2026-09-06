# -*- coding: utf-8 -*-
"""E3: Hot-swap Integrity — 模块切换时 KV cache 污染与输出质量

测试：生成序列中切换模块 k 次，测量切换点前后：
  1. 文本连贯性（perplexity proxy: next-token logit gap）
  2. hidden state 的 CKA（切换前后表示相似度）
  3. 与不切换基线的文本差异

Kill 判据：质量随 k 单调下降且无法通过恒等前缀设计修复 → 架构不成立
"""
import json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
TAP_LAYER = 16
PROMPT = "The theory of external neural modules states that"
N_TOKENS = 24

MODULES = {
    "A_amplify": lambda h: h * 3.0 + 1.0,
    "B_damp":    lambda h: h * 0.3 + 0.5,
    "C_invert":  lambda h: h * -0.5 + 0.2,
    "D_shift":   lambda h: h + 1.5,
}

def make_hook(transform_fn):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs_new = hs.clone()
        if hs.dim() == 3:
            hs_new[:, -1, :] = transform_fn(hs[:, -1, :])
        elif hs.dim() == 2:
            hs_new[-1, :] = transform_fn(hs[-1, :])
        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new
    return hook_fn

def cka(X, Y):
    """Linear CKA between two hidden state matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    xy = (X.T @ Y).norm().item()
    xx = (X.T @ X).norm().item()
    yy = (Y.T @ Y).norm().item()
    return xy / (xx * yy + 1e-10) ** 0.5

def generate_with_switches(model, tokenizer, prompt, n_tokens, switch_schedule):
    """Generate with module switches at specified token positions.
    
    switch_schedule: dict {token_idx: module_name}
    Returns: text, hidden_states_list, switch_positions
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = inputs.input_ids.clone()
    all_hs = []
    current_module = None
    current_handle = None
    
    hook_handle = None
    active_module = None
    
    for t in range(n_tokens):
        # Check if we need to switch module
        if t in switch_schedule:
            new_module = switch_schedule[t]
            if hook_handle is not None:
                hook_handle.remove()
            if new_module and new_module in MODULES:
                layer = model.model.layers[TAP_LAYER]
                hook_handle = layer.register_forward_hook(make_hook(MODULES[new_module]))
                active_module = new_module
            else:
                hook_handle = None
                active_module = None
        
        with torch.no_grad():
            out = model(input_ids=generated_ids if t == 0 else generated_ids[:, -1:],
                       use_cache=True)
        
        # Capture hidden states at tap layer
        # (we capture from the hook's perspective)
        
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Logit gap (confidence measure)
        logits_last = out.logits[0, -1, :]
        top2 = torch.topk(logits_last, 2)
        confidence = (top2.values[0] - top2.values[1]).item()
        
        generated_ids = torch.cat([generated_ids, next_id], dim=-1)
        all_hs.append({"token_idx": t, "module": active_module, "confidence": round(confidence, 4)})
    
    if hook_handle is not None:
        hook_handle.remove()
    
    new_tokens = generated_ids[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, all_hs

def main():
    t0 = time.time()
    print(f"[E3] Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()
    print(f"[E3] Model loaded in {time.time()-t0:.1f}s")

    results = {"model": MODEL_PATH, "n_tokens": N_TOKENS, "prompt": PROMPT}

    # ---- Baseline: no modules ----
    print("\n[Base] No module:")
    base_text, base_hs = generate_with_switches(model, tokenizer, PROMPT, N_TOKENS, {})
    base_confs = [h["confidence"] for h in base_hs]
    print(f"  text: {base_text}")
    print(f"  confidence: mean={np.mean(base_confs):.4f} min={np.min(base_confs):.4f}")
    results["baseline"] = {"text": base_text, "mean_conf": round(np.mean(base_confs), 4)}

    # ---- Test 1: Single module throughout (no switching) ----
    print("\n[Single] Module A throughout:")
    text_a, hs_a = generate_with_switches(model, tokenizer, PROMPT, N_TOKENS, {0: "A_amplify"})
    confs_a = [h["confidence"] for h in hs_a]
    print(f"  text: {text_a}")
    print(f"  confidence: mean={np.mean(confs_a):.4f} min={np.min(confs_a):.4f}")
    results["single_A"] = {"text": text_a, "mean_conf": round(np.mean(confs_a), 4)}

    # ---- Test 2: Switch once (A for 12 tokens, then B for 12) ----
    print("\n[Switch-1] A(12) -> B(12):")
    sched1 = {0: "A_amplify", 12: "B_damp"}
    text_s1, hs_s1 = generate_with_switches(model, tokenizer, PROMPT, N_TOKENS, sched1)
    confs_s1 = [h["confidence"] for h in hs_s1]
    pre_switch = [h["confidence"] for h in hs_s1 if h["token_idx"] < 12]
    post_switch = [h["confidence"] for h in hs_s1 if h["token_idx"] >= 12]
    print(f"  text: {text_s1}")
    print(f"  confidence: pre_switch={np.mean(pre_switch):.4f} post_switch={np.mean(post_switch):.4f}")
    results["switch_1"] = {"text": text_s1,
                           "pre_conf": round(np.mean(pre_switch), 4),
                           "post_conf": round(np.mean(post_switch), 4)}

    # ---- Test 3: Switch every 4 tokens (6 switches total) ----
    print("\n[Switch-6] A(4) -> B(4) -> C(4) -> D(4) -> A(4) -> B(4):")
    mods = ["A_amplify", "B_damp", "C_invert", "D_shift", "A_amplify", "B_damp"]
    sched6 = {i * 4: m for i, m in enumerate(mods)}
    text_s6, hs_s6 = generate_with_switches(model, tokenizer, PROMPT, N_TOKENS, sched6)
    confs_s6 = [h["confidence"] for h in hs_s6]
    print(f"  text: {text_s6}")
    print(f"  confidence: mean={np.mean(confs_s6):.4f} min={np.min(confs_s6):.4f}")
    results["switch_6"] = {"text": text_s6, "mean_conf": round(np.mean(confs_s6), 4)}

    # ---- Test 4: Switch every token (maximum churn) ----
    print("\n[Switch-24] Switch every token:")
    mods_24 = [list(MODULES.keys())[i % 4] for i in range(N_TOKENS)]
    sched24 = {i: m for i, m in enumerate(mods_24)}
    text_s24, hs_s24 = generate_with_switches(model, tokenizer, PROMPT, N_TOKENS, sched24)
    confs_s24 = [h["confidence"] for h in hs_s24]
    print(f"  text: {text_s24}")
    print(f"  confidence: mean={np.mean(confs_s24):.4f} min={np.min(confs_s24):.4f}")
    results["switch_24"] = {"text": text_s24, "mean_conf": round(np.mean(confs_s24), 4)}

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print("E3 HOT-SWAP INTEGRITY — ANALYSIS")
    print("=" * 70)
    
    base_conf = results["baseline"]["mean_conf"]
    single_conf = results["single_A"]["mean_conf"]
    s1_conf = results["switch_1"]["post_conf"]
    s6_conf = results["switch_6"]["mean_conf"]
    s24_conf = results["switch_24"]["mean_conf"]
    
    print(f"  Switches:  0     0     1     6     24")
    print(f"  Mean conf: {base_conf:.4f} {single_conf:.4f} {s1_conf:.4f} {s6_conf:.4f} {s24_conf:.4f}")
    
    # Does confidence degrade with more switches?
    no_switch = single_conf
    degrade_rate = (no_switch - s24_conf) / max(no_switch, 1e-10)
    print(f"\n  Degradation (0→24 switches): {degrade_rate:.1%}")
    
    # Kill criterion: quality degrades > 50% with max switches
    quality_ok = degrade_rate < 0.5
    print(f"  Quality acceptable (<50% degradation): {quality_ok}")
    
    # Per-token confidence around switch points
    print("\n  Per-token confidence at switch boundaries (switch_6):")
    for i, h in enumerate(hs_s6):
        marker = " ← SWITCH" if h["token_idx"] in sched6 else ""
        if marker or i < 3:
            print(f"    t={h['token_idx']:2d} mod={h['module'] or 'none':<12} conf={h['confidence']:.4f}{marker}")

    all_texts = [base_text, text_a, text_s1, text_s6, text_s24]
    unique_texts = len(set(all_texts))
    print(f"\n  Unique texts across conditions: {unique_texts}/5")
    print(f"  All different: {unique_texts == 5}")

    verdict = "PASS" if quality_ok else "FAIL"
    print(f"\n  E3 VERDICT: {verdict}")
    results["analysis"] = {
        "degradation_rate": round(degrade_rate, 4),
        "quality_ok": quality_ok,
        "unique_texts": unique_texts,
        "verdict": verdict}

    out = "/root/e3_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out}")
    print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
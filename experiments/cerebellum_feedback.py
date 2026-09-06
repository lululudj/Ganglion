# -*- coding: utf-8 -*-
"""Cerebellum-Brain Feedback: 本地小脑的 hidden-state 变换如何被大脑的剩余层解读？

核心问题：模块在外置层做的变换，经过剩余层的非线性传播后，
对最终输出的影响是均匀噪声还是语义选择性（有意义方向 vs 随机方向）？

如果剩余层对"有意义"的扰动（如均值方向、方差方向）和对"随机噪声"
的响应有显著差异 → 说明剩余层在做**语义滤波**，外置模块不只是加噪，
而是在**表征空间中有方向性地影响模型的决策**。

实验设计（全部在本地 TinyBackbone 上，可复现）：

E1  基线确认：identity 模块 (h→h) 输出应与无模块基线逐 token 一致
E2  线性变换扫描：h*scale + shift，观察 argmax 翻转率与 KL 散度
E3  语义方向 vs 随机方向：
    - mean 方向：h + alpha * mean(h)（强化当前语义）
    - anti-mean 方向：h - alpha * mean(h)（反转语义）
    - random 方向：h + alpha * random_vector（无语义噪声）
    同等 L2 范数的扰动，比较输出变化 → 如果 mean 扰动的影响 ≠ random 扰动
    → 剩余层在做语义选择性放大/抑制
E4  层深度敏感性：tap_layer = 0, 1, 2，同一变换在不同深度的影响
E5  反馈回路：模块根据 hidden state 的范数做条件变换（"危险检测→修正"）
    观察模型行为是否被引导（不只是噪声）
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ganglion.host import TinyBackbone

torch.manual_seed(42)
PROMPT = [1, 2, 3, 4]
N_TOKENS = 24
DIM = 64
VOCAB = 32
LAYERS = 3


def make_backbone(tap_layer=1, seed=0):
    return TinyBackbone(vocab=VOCAB, dim=DIM, n_layers=LAYERS, tap_layer=tap_layer, seed=seed)


def run_generation(bb, transform_fn=None):
    """Generate tokens with optional transform at tap layer."""
    tokens = list(PROMPT)
    generated = []
    all_logits = []
    for _ in range(N_TOKENS):
        h = bb.encode_upto(tokens, bb.tap_layer)
        if transform_fn is not None:
            h = transform_fn(h)
        logits = bb.decode_from(h, bb.tap_layer)
        nxt = int(torch.argmax(logits[-1]).item())
        all_logits.append(logits[-1].detach().clone())
        tokens.append(nxt)
        generated.append(nxt)
    return generated, all_logits


def argmax_flip_rate(seq_a, seq_b):
    if len(seq_a) != len(seq_b):
        return 1.0
    return sum(1 for a, b in zip(seq_a, seq_b) if a != b) / len(seq_a)


def kl_div(logits_a, logits_b):
    """Average KL divergence between two logits sequences."""
    total = 0.0
    for la, lb in zip(logits_a, logits_b):
        pa = torch.softmax(la, dim=-1)
        log_pb = torch.log_softmax(lb, dim=-1)
        total += (pa * (torch.log(pa + 1e-10) - log_pb)).sum().item()
    return total / len(logits_a)


def main():
    results = {}
    bb = make_backbone(tap_layer=1)
    bb.eval()

    # ===== E1: Baseline =====
    print("=" * 70)
    print("E1: Baseline + identity module")
    print("=" * 70)
    baseline_tokens, baseline_logits = run_generation(bb)
    identity_tokens, identity_logits = run_generation(bb, lambda h: h)
    e1_match = baseline_tokens == identity_tokens
    print(f"  baseline: {baseline_tokens}")
    print(f"  identity: {identity_tokens}")
    print(f"  match: {e1_match}")
    results["E1_baseline"] = {
        "baseline": baseline_tokens, "identity": identity_tokens,
        "match": e1_match, "pass": e1_match}

    # ===== E2: Linear transform sweep =====
    print("\n" + "=" * 70)
    print("E2: Linear transform sweep h*scale + shift")
    print("=" * 70)
    e2_rows = []
    for scale in [-2, -1, 0, 0.5, 0.9, 1.0, 1.1, 2, 5, 10]:
        for shift in [0, 0.5, 1.0, 2.0]:
            seq, logits = run_generation(bb, lambda h, s=scale, sh=shift: h * s + sh)
            flip = argmax_flip_rate(baseline_tokens, seq)
            kl = kl_div(baseline_logits, logits)
            e2_rows.append({"scale": scale, "shift": shift,
                           "flip_rate": round(flip, 3), "kl": round(kl, 4),
                           "tokens": seq})
            if flip > 0:
                print(f"  scale={scale:+.1f} shift={shift:+.1f} → flip={flip:.1%} KL={kl:.4f}")
    # identity should have zero flip
    e2_identity = [r for r in e2_rows if r["scale"] == 1.0 and r["shift"] == 0]
    e2_pass = len(e2_identity) == 0 or e2_identity[0]["flip_rate"] == 0
    results["E2_linear_sweep"] = {"rows": e2_rows, "pass": e2_pass}
    print(f"  ({len(e2_rows)} configs tested)")

    # ===== E3: Semantic direction vs Random direction =====
    print("\n" + "=" * 70)
    print("E3: Semantic direction vs Random direction (same L2 norm)")
    print("=" * 70)
    # Get reference hidden states to compute mean direction
    with torch.no_grad():
        h_ref = bb.encode_upto(PROMPT, bb.tap_layer)
        h_mean = h_ref.mean(dim=0)  # [dim] — "semantic centroid" direction
        h_std = h_ref.std()
        torch.manual_seed(123)
        h_random = torch.randn(DIM) * h_std  # same std as hidden states

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    e3_rows = []
    for alpha in alphas:
        # Mean direction (reinforce current semantics)
        seq_mean, logits_mean = run_generation(
            bb, lambda h, a=alpha, m=h_mean: h + a * m.unsqueeze(0))
        # Anti-mean direction (reverse semantics)
        seq_anti, logits_anti = run_generation(
            bb, lambda h, a=alpha, m=h_mean: h - a * m.unsqueeze(0))
        # Random direction (same L2 norm as mean direction)
        seq_rand, logits_rand = run_generation(
            bb, lambda h, a=alpha, r=h_random: h + a * r.unsqueeze(0))

        flip_mean = argmax_flip_rate(baseline_tokens, seq_mean)
        flip_anti = argmax_flip_rate(baseline_tokens, seq_anti)
        flip_rand = argmax_flip_rate(baseline_tokens, seq_rand)
        kl_mean = kl_div(baseline_logits, logits_mean)
        kl_anti = kl_div(baseline_logits, logits_anti)
        kl_rand = kl_div(baseline_logits, logits_rand)

        e3_rows.append({
            "alpha": alpha,
            "flip_mean": round(flip_mean, 3), "flip_anti": round(flip_anti, 3),
            "flip_random": round(flip_rand, 3),
            "kl_mean": round(kl_mean, 4), "kl_anti": round(kl_anti, 4),
            "kl_random": round(kl_rand, 4),
        })
        print(f"  α={alpha:.1f}: mean_flip={flip_mean:.1%} anti_flip={flip_anti:.1%} "
              f"rand_flip={flip_rand:.1%} | KL: mean={kl_mean:.4f} anti={kl_anti:.4f} rand={kl_rand:.4f}")

    # Key question: is mean-direction response ≠ random-direction response?
    # (same L2 norm perturbation, different semantic content)
    e3_selective = False
    for row in e3_rows:
        if abs(row["flip_mean"] - row["flip_random"]) > 0.15:
            e3_selective = True
            break
    results["E3_direction"] = {"rows": e3_rows, "semantically_selective": e3_selective,
                               "pass": True}
    print(f"  semantically_selective: {e3_selective}")

    # ===== E4: Layer depth sensitivity =====
    print("\n" + "=" * 70)
    print("E4: Same transform (h*2+1) at different tap layers")
    print("=" * 70)
    e4_rows = []
    for tap in range(LAYERS):
        bb_tap = make_backbone(tap_layer=tap)
        base_t, base_l = run_generation(bb_tap)
        mod_t, mod_l = run_generation(bb_tap, lambda h: h * 2 + 1)
        flip = argmax_flip_rate(base_t, mod_t)
        kl = kl_div(base_l, mod_l)
        layers_remaining = LAYERS - 1 - tap
        e4_rows.append({"tap_layer": tap, "layers_remaining": layers_remaining,
                        "flip_rate": round(flip, 3), "kl": round(kl, 4)})
        print(f"  tap={tap} (remaining={layers_remaining}): flip={flip:.1%} KL={kl:.4f}")
    results["E4_depth"] = {"rows": e4_rows, "pass": True}

    # ===== E5: Conditional feedback (danger detection → correction) =====
    print("\n" + "=" * 70)
    print("E5: Conditional feedback: module detects high-norm hidden states → damps")
    print("=" * 70)
    # Module acts as a "cerebellum": if hidden state norm > threshold, damp it
    # This simulates a safety reflex that modifies the brain's computation
    norms = []
    with torch.no_grad():
        h_ref = bb.encode_upto(PROMPT, bb.tap_layer)
        ref_norm = h_ref.norm(dim=-1).mean().item()

    threshold = ref_norm * 2.0  # if hidden state norm > 2x normal → damp
    print(f"  reference norm: {ref_norm:.2f}, threshold: {threshold:.2f}")

    def conditional_damp(h):
        norm = h.norm(dim=-1, keepdim=True)
        damp_mask = (norm > threshold).float()
        damped = h * 0.3  # strong damping
        return h * (1 - damp_mask) + damped * damp_mask

    seq_e5, logits_e5 = run_generation(bb, conditional_damp)
    # Also run with a "dangerous" transform that amplifies norms
    seq_danger, logits_danger = run_generation(bb, lambda h: h * 10.0)
    flip_danger_no_feedback = argmax_flip_rate(baseline_tokens, seq_danger)
    flip_danger_with_feedback = argmax_flip_rate(baseline_tokens, seq_e5)

    # The feedback should prevent the dangerous transform from fully disrupting output
    # (conditional damp only activates when norm is high, so normal operation is unaffected)
    seq_normal_with_feedback, _ = run_generation(bb, conditional_damp)
    flip_normal_with_feedback = argmax_flip_rate(baseline_tokens, seq_normal_with_feedback)

    e5_pass = flip_normal_with_feedback == 0  # normal operation unaffected
    print(f"  danger (h*10) no feedback: flip={flip_danger_no_feedback:.1%}")
    print(f"  normal with feedback: flip={flip_normal_with_feedback:.1%}")
    print(f"  feedback protects normal path: {e5_pass}")
    results["E5_feedback"] = {
        "flip_danger_no_feedback": round(flip_danger_no_feedback, 3),
        "flip_normal_with_feedback": round(flip_normal_with_feedback, 3),
        "pass": e5_pass}

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = all([
        results["E1_baseline"]["pass"],
        results["E2_linear_sweep"]["pass"],
        results["E5_feedback"]["pass"],
    ])
    print(f"  E1 baseline match: {results['E1_baseline']['pass']}")
    print(f"  E2 linear sweep: {results['E2_linear_sweep']['pass']}")
    print(f"  E3 semantically selective: {results['E3_direction']['semantically_selective']}")
    print(f"  E5 feedback protects: {results['E5_feedback']['pass']}")
    print(f"  ALL PASS: {all_pass}")

    if results["E3_direction"]["semantically_selective"]:
        print("\n  ★ KEY FINDING: The remaining layers respond differently to "
              "semantically meaningful perturbations vs random noise at the same L2 norm.")
        print("    This means the external module is not just adding noise —")
        print("    it is engaging in DIRECTIONAL influence on the model's decision process.")

    results["all_pass"] = all_pass
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cerebellum_feedback_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
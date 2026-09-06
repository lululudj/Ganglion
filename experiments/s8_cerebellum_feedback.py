# -*- coding: utf-8 -*-
"""S8: Cerebellum-to-Brain Feedback Loop — 小脑反哺大脑

核心假设：当多个外置模块（小脑）持续向云端骨干（大脑）的中间层
注入变换后的 hidden states 时，大脑可以通过微调其后续层来"学会"
更好地解读这些变换——即使模块本身不变，整体系统行为也会改善。

实验设计：
  Phase A  数据收集：N 个模块各跑一次生成，收集 (h_tap, h_module, target) 三元组
  Phase B  微调：冻结 tap 之前的层，只训练 tap 之后的层 + head
           损失 = 模块激活时鼓励目标 token + 基线激活时保持原 token
  Phase C  评估：同一组模块、同一组变换，微调后的骨干
           → 目标 token 达成率提升？
           → 基线（无模块）行为不破坏？

关键指标：
  target_hit_before / target_hit_after  — 模块引导目标 token 的成功率
  baseline_preserved                     — 无模块时输出不变（安全）
  module_effectiveness                   — 微调后模块的"话语权"是否增强
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ganglion.host import TinyBackbone

torch.manual_seed(42)
DEVICE = "cpu"
PROMPT = [1, 2, 3, 4]
N_TOKENS = 24
DIM = 64
VOCAB = 32
LAYERS = 3
TAP = 1
TARGET_TOKEN = 7        # 我们希望模块引导大脑输出的 token
N_MODULES = 5
EPOCHS = 200
LR = 1e-3


def make_backbone(seed=0):
    return TinyBackbone(vocab=VOCAB, dim=DIM, n_layers=LAYERS, tap_layer=TAP, seed=seed)


def generate(bb, transform_fn=None):
    """Generate tokens with optional transform at tap layer. Returns (tokens, h_taps, logits_last)."""
    tokens = list(PROMPT)
    generated = []
    h_taps = []       # hidden states at tap layer (before transform)
    h_transformed = []  # after transform (what the remaining layers actually see)
    all_logits = []
    for _ in range(N_TOKENS):
        h = bb.encode_upto(tokens, bb.tap_layer)
        h_taps.append(h.detach().clone())
        h_in = h
        if transform_fn is not None:
            h_in = transform_fn(h)
        h_transformed.append(h_in.detach().clone())
        logits = bb.decode_from(h_in, bb.tap_layer)
        nxt = int(torch.argmax(logits[-1]).item())
        all_logits.append(logits[-1].detach().clone())
        tokens.append(nxt)
        generated.append(nxt)
    return generated, h_taps, h_transformed, all_logits


def argmax_flip(seq_a, seq_b):
    return sum(1 for a, b in zip(seq_a, seq_b) if a != b) / max(len(seq_a), 1)


def target_hit_rate(seqs, target=TARGET_TOKEN):
    """Fraction of generated tokens that equal target."""
    total = sum(len(s) for s in seqs)
    hits = sum(1 for s in seqs for t in s if t == target)
    return hits / max(total, 1)


def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    results = {"device": DEVICE, "target_token": TARGET_TOKEN, "n_modules": N_MODULES}

    bb = make_backbone()
    bb.to(DEVICE)
    bb.eval()

    # ---- Define modules (each simulates a different robot's skill) ----
    module_transforms = {
        "M1_amplify":   lambda h: h * 1.5 + 0.2,
        "M2_damp":      lambda h: h * 0.5 + 0.3,
        "M3_shift":     lambda h: h + 0.8,
        "M4_invert":    lambda h: h * -0.8 + 0.1,
        "M5_strong":    lambda h: h * 3.0,
    }
    module_names = list(module_transforms.keys())

    # ---- Phase A: Baseline + Data Collection ----
    print("\n" + "=" * 70)
    print("Phase A: Baseline + module data collection")
    print("=" * 70)

    baseline_tokens, base_h, base_ht, base_logits = generate(bb)
    print(f"  baseline: {baseline_tokens}")

    # Collect training data: (h_transformed, label)
    # label = TARGET_TOKEN if from module, else baseline token
    train_h = []       # transformed hidden states (input to remaining layers)
    train_labels = []  # desired output token
    module_data = {}

    for mname in module_names:
        tfn = module_transforms[mname]
        seq, h_taps, h_trans, logits = generate(bb, tfn)
        flip = argmax_flip(baseline_tokens, seq)
        hit = sum(1 for t in seq if t == TARGET_TOKEN) / len(seq)
        module_data[mname] = {"tokens": seq, "flip_rate": round(flip, 3),
                              "target_hit": round(hit, 3)}
        print(f"  {mname}: flip={flip:.1%} target_hit={hit:.1%} tokens={seq[:8]}...")

        # Collect: transformed hidden states from last position → want TARGET_TOKEN
        for ht in h_trans:
            train_h.append(ht[-1])          # [dim] — last token's hidden state
            train_labels.append(TARGET_TOKEN)

    # Also collect baseline data: normal hidden states → want baseline token (preserve)
    for ht in base_ht:
        train_h.append(ht[-1])
        baseline_last = baseline_tokens[-1] if baseline_tokens else 4
        train_labels.append(baseline_last)  # preserve baseline behavior

    train_h = torch.stack(train_h).to(DEVICE)         # [N, dim]
    train_labels = torch.tensor(train_labels).to(DEVICE)  # [N]

    # Record pre-finetune metrics
    pre_target_hit = target_hit_rate([module_data[m]["tokens"] for m in module_names])
    pre_flips = {m: module_data[m]["flip_rate"] for m in module_names}
    print(f"\n  PRE-finetune: target_hit={pre_target_hit:.1%} "
          f"avg_flip={np.mean(list(pre_flips.values())):.1%}")

    results["phase_A"] = {
        "baseline": baseline_tokens,
        "modules": module_data,
        "pre_target_hit": round(pre_target_hit, 4),
        "pre_avg_flip": round(float(np.mean(list(pre_flips.values()))), 4),
    }

    # ---- Phase B: Fine-tune remaining layers (brain learns to cooperate) ----
    print("\n" + "=" * 70)
    print(f"Phase B: Fine-tune layers {TAP+1}..{LAYERS-1} + head ({EPOCHS} epochs)")
    print("=" * 70)

    bb_ft = copy.deepcopy(bb)
    bb_ft.train()

    # Freeze layers before tap (brain's "perception" doesn't change)
    for param in bb_ft.emb.parameters():
        param.requires_grad = False
    for i in range(TAP + 1):
        for param in bb_ft.blocks[i].parameters():
            param.requires_grad = False

    # Only layers after tap + head are trainable
    trainable = []
    for i in range(TAP + 1, LAYERS):
        trainable += list(bb_ft.blocks[i].parameters())
    trainable += list(bb_ft.head.parameters())

    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Mixed loss: module data → TARGET_TOKEN; baseline data → baseline token
    n_module_samples = len(module_names) * N_TOKENS
    n_baseline_samples = N_TOKENS

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        # Forward through remaining layers only (input = collected h_transformed)
        h = train_h
        for i in range(TAP + 1, LAYERS):
            h = bb_ft.blocks[i](h)
        logits = bb_ft.head(h)  # [N, vocab]

        # Weight module samples higher (we want the brain to learn module cooperation)
        weights = torch.ones(len(train_labels), device=DEVICE)
        weights[:n_module_samples] = 2.0   # module samples get 2x weight
        weights[n_module_samples:] = 1.0   # baseline samples get 1x weight

        loss = F.cross_entropy(logits, train_labels, reduction="none")
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            pred = logits.argmax(dim=-1)
            acc = (pred == train_labels).float().mean().item()
            print(f"  epoch {epoch+1}/{EPOCHS}: loss={loss.item():.4f} acc={acc:.1%}")

    bb_ft.eval()

    # ---- Phase C: Evaluate fine-tuned backbone ----
    print("\n" + "=" * 70)
    print("Phase C: Evaluate fine-tuned backbone (same modules, same transforms)")
    print("=" * 70)

    ft_baseline_tokens, _, _, _ = generate(bb_ft)
    baseline_preserved = baseline_tokens == ft_baseline_tokens
    print(f"  baseline (no module): {'PRESERVED' if baseline_preserved else 'BROKEN!'}")
    print(f"    before: {baseline_tokens}")
    print(f"    after:  {ft_baseline_tokens}")

    ft_module_data = {}
    for mname in module_names:
        tfn = module_transforms[mname]
        seq, _, _, _ = generate(bb_ft, tfn)
        flip = argmax_flip(baseline_tokens, seq)
        hit = sum(1 for t in seq if t == TARGET_TOKEN) / len(seq)
        ft_module_data[mname] = {"tokens": seq, "flip_rate": round(flip, 3),
                                 "target_hit": round(hit, 3)}
        print(f"  {mname}: flip={flip:.1%} target_hit={hit:.1%} tokens={seq[:8]}...")

    post_target_hit = target_hit_rate([ft_module_data[m]["tokens"] for m in module_names])
    post_flips = {m: ft_module_data[m]["flip_rate"] for m in module_names}

    print(f"\n  POST-finetune: target_hit={post_target_hit:.1%} "
          f"avg_flip={np.mean(list(post_flips.values())):.1%}")
    print(f"\n  Improvement: target_hit {pre_target_hit:.1%} → {post_target_hit:.1%} "
          f"({(post_target_hit - pre_target_hit)*100:+.1f}pp)")

    # ---- Assertions ----
    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)
    a1 = baseline_preserved
    a2 = post_target_hit > pre_target_hit
    a3 = all(post_flips[m] >= pre_flips[m] - 0.05 for m in module_names)  # modules still have effect

    print(f"  A1 baseline_preserved:       {'PASS' if a1 else 'FAIL'}")
    print(f"  A2 target_hit_improved:      {'PASS' if a2 else 'FAIL'} "
          f"({pre_target_hit:.1%} → {post_target_hit:.1%})")
    print(f"  A3 module_effectiveness:     {'PASS' if a3 else 'FAIL'}")

    all_pass = a1 and a2 and a3
    print(f"\n  ALL PASS: {all_pass}")

    results["phase_B"] = {"epochs": EPOCHS, "lr": LR, "trainable_params": sum(p.numel() for p in trainable)}
    results["phase_C"] = {
        "baseline_preserved": baseline_preserved,
        "modules": ft_module_data,
        "post_target_hit": round(post_target_hit, 4),
        "post_avg_flip": round(float(np.mean(list(post_flips.values()))), 4),
    }
    results["assertions"] = {
        "A1_baseline_preserved": a1,
        "A2_target_hit_improved": a2,
        "A3_module_effectiveness": a3,
        "all_pass": all_pass,
    }
    results["summary"] = {
        "target_hit_before": round(pre_target_hit, 4),
        "target_hit_after": round(post_target_hit, 4),
        "improvement_pp": round((post_target_hit - pre_target_hit) * 100, 2),
        "baseline_preserved": baseline_preserved,
        "elapsed_s": round(time.time() - t_start, 2),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s8_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
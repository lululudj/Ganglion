# -*- coding: utf-8 -*-
"""S8 云端宿主：Qwen3-8B + LoRA 微调 — 小脑反哺大脑（真实大模型验证）

流程：
  A. 基线生成（无模块）→ baseline_text
  B. 模块生成（h*5+2 hook）→ module_text_before
  C. LoRA 微调 layers 16-35：目标 = 模块激活时输出 TARGET_TEXT
  D. 重新生成（同一模块、同一变换）→ module_text_after
  E. 安全检查：无模块时输出 = baseline_text（不破坏）

输出：s8_cloud_results.json
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
TAP_LAYER = 16
N_TOKENS = 24
PROMPT = "The theory of external neural modules states that"
SCALE, SHIFT = 5.0, 2.0
TARGET_TEXT = " characterized by distributed specialized processing units that operate independently"
EPOCHS = 30
LR = 5e-4
LORA_R = 8


def module_transform_hook(scale, shift):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        hs_new = hs.clone()
        if hs.dim() == 3:
            hs_new[:, -1, :] = hs[:, -1, :] * scale + shift
        elif hs.dim() == 2:
            hs_new[-1, :] = hs[-1, :] * scale + shift
        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new
    return hook_fn


def generate(model, tokenizer, prompt, n_tokens, hook_handle=None):
    """Autoregressive generation with optional module hook active."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = inputs.input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        for _ in range(n_tokens):
            if past_key_values is None:
                out = model(input_ids=generated_ids, use_cache=True)
            else:
                out = model(input_ids=generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)

    new_tokens = generated_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    t0 = time.time()
    device = "cuda"
    print(f"[S8-cloud] Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager")
    model.eval()
    print(f"[S8-cloud] Model loaded in {time.time()-t0:.1f}s | dtype={model.dtype}")

    results = {"model": MODEL_PATH, "tap_layer": TAP_LAYER,
               "scale": SCALE, "shift": SHIFT, "target": TARGET_TEXT}

    # ---- Phase A: Baseline (no module) ----
    print("\n[A] Baseline generation (no module) ...")
    baseline_text = generate(model, tokenizer, PROMPT, N_TOKENS)
    print(f"  baseline: {baseline_text}")
    results["baseline_text"] = baseline_text

    # ---- Phase B: Module generation (before fine-tune) ----
    print(f"\n[B] Module generation (h*{SCALE}+{SHIFT} at layer {TAP_LAYER}) ...")
    hook_handle = model.model.layers[TAP_LAYER].register_forward_hook(
        module_transform_hook(SCALE, SHIFT))
    module_text_before = generate(model, tokenizer, PROMPT, N_TOKENS)
    hook_handle.remove()
    print(f"  module_before: {module_text_before}")
    results["module_text_before"] = module_text_before

    # Check module actually changes output
    module_changes = module_text_before != baseline_text
    print(f"  module changes output: {module_changes}")
    results["module_changes_output_before"] = module_changes

    # ---- Phase C: LoRA fine-tune layers 16-35 ----
    print(f"\n[C] LoRA fine-tune (rank={LORA_R}, {EPOCHS} epochs, lr={LR}) ...")

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(TAP_LAYER, 36)),
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # Prepare training data: BOTH module-active and module-inactive
    target_ids = tokenizer(TARGET_TEXT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    baseline_ids = tokenizer(baseline_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)

    # Hook for module-active training
    hook_train = model.base_model.model.model.layers[TAP_LAYER].register_forward_hook(
        module_transform_hook(SCALE, SHIFT))

    losses = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Pass 1: WITH module hook → target text
        full_ids_m = torch.cat([prompt_ids, target_ids], dim=-1)
        labels_m = full_ids_m.clone()
        labels_m[:, :prompt_ids.shape[1]] = -100
        out_m = model(input_ids=full_ids_m, labels=labels_m)
        loss_module = out_m.loss
        
        # Pass 2: WITHOUT module hook → baseline text (preserve)
        hook_train.remove()
        full_ids_b = torch.cat([prompt_ids, baseline_ids], dim=-1)
        labels_b = full_ids_b.clone()
        labels_b[:, :prompt_ids.shape[1]] = -100
        out_b = model(input_ids=full_ids_b, labels=labels_b)
        loss_baseline = out_b.loss
        hook_train = model.base_model.model.model.layers[TAP_LAYER].register_forward_hook(
            module_transform_hook(SCALE, SHIFT))
        
        # Combined loss: module cooperation + baseline preservation
        loss = loss_module + loss_baseline
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: loss={loss.item():.4f} (mod={loss_module.item():.4f} base={loss_baseline.item():.4f})")

    hook_train.remove()
    model.eval()
    print(f"  final loss: {losses[-1]:.4f} (start: {losses[0]:.4f})")
    results["training"] = {"epochs": EPOCHS, "lr": LR, "lora_r": LORA_R,
                           "loss_start": round(losses[0], 4),
                           "loss_end": round(losses[-1], 4)}

    # ---- Phase D: Module generation (after fine-tune) ----
    print(f"\n[D] Module generation after fine-tune ...")
    hook_handle_eval = model.base_model.model.model.layers[TAP_LAYER].register_forward_hook(
        module_transform_hook(SCALE, SHIFT))
    module_text_after = generate(model, tokenizer, PROMPT, N_TOKENS)
    hook_handle_eval.remove()
    print(f"  module_after: {module_text_after}")
    results["module_text_after"] = module_text_after

    # Check target achievement
    target_words = set(TARGET_TEXT.lower().split())
    after_words = set(module_text_after.lower().split())
    target_overlap = len(target_words & after_words) / max(len(target_words), 1)
    before_words = set(module_text_before.lower().split())
    before_overlap = len(target_words & before_words) / max(len(target_words), 1)
    print(f"  target overlap: before={before_overlap:.1%} after={target_overlap:.1%}")
    results["target_overlap_before"] = round(before_overlap, 4)
    results["target_overlap_after"] = round(target_overlap, 4)

    # ---- Phase E: Safety check (no module after fine-tune) ----
    print(f"\n[E] Safety check: baseline after fine-tune (no module) ...")
    baseline_after = generate(model, tokenizer, PROMPT, N_TOKENS)
    baseline_preserved = baseline_after == baseline_text
    print(f"  baseline_after: {baseline_after}")
    print(f"  baseline_preserved: {baseline_preserved}")
    results["baseline_after_text"] = baseline_after
    results["baseline_preserved"] = baseline_preserved

    # ---- Assertions ----
    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)
    a1 = module_changes  # module changes output before fine-tune
    a2 = target_overlap > before_overlap  # fine-tune improved target achievement
    a3 = baseline_preserved  # baseline not broken
    a4 = losses[-1] < losses[0]  # training loss decreased

    print(f"  A1 module_changes_output:    {'PASS' if a1 else 'FAIL'}")
    print(f"  A2 target_overlap_improved:  {'PASS' if a2 else 'FAIL'} "
          f"({before_overlap:.1%} → {target_overlap:.1%})")
    print(f"  A3 baseline_preserved:       {'PASS' if a3 else 'FAIL'}")
    print(f"  A4 training_loss_decreased:  {'PASS' if a4 else 'FAIL'} "
          f"({losses[0]:.4f} → {losses[-1]:.4f})")

    all_pass = a1 and a2 and a3 and a4
    print(f"\n  ALL PASS: {all_pass}")
    results["assertions"] = {
        "A1_module_changes": a1, "A2_target_improved": a2,
        "A3_baseline_preserved": a3, "A4_loss_decreased": a4,
        "all_pass": all_pass}
    results["elapsed_s"] = round(time.time() - t0, 1)

    with open("/root/s8_cloud_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to /root/s8_cloud_results.json")
    print(f"  Total time: {time.time()-t0:.1f}s")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

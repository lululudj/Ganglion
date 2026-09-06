# -*- coding: utf-8 -*-
"""E1 cloud attribution ablation for Qwen3-8B.

The task is intentionally impossible from text: a neutral prompt is paired with
a sensor value through an ABI residual write.  The target token depends only on
the sensor value.  Real pairs sensor tensor with target; Permuted destroys the
pairing while preserving both marginals.
"""
import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/root/autodl-tmp/Qwen3-8B"
TAP_LAYER = 16
PROMPT = "External neural channel:"
N_CLASSES = 4
LABELS = [" A", " B", " C", " D"]
TENSOR_SCALE = 0.35
LORA_R = 8
LORA_ALPHA = 16
LR = 5e-4
BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--arms", default="real,sensor_permuted,raw_onehot,trajectory_only")
    parser.add_argument("--output", default="/root/e1_cloud_results.json")
    args = parser.parse_args()
    if args.quick:
        args.train_samples = min(args.train_samples, 16)
        args.eval_samples = min(args.eval_samples, 24)
        args.seeds = min(args.seeds, 1)
    return args


class SensorWriteHook:
    def __init__(self, vector, prompt_len):
        self.vector = vector
        self.prompt_len = prompt_len
        self.sensor = None

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.sensor is None:
            return output
        updated = hidden.clone()
        sensor_values = self.sensor.detach().cpu().squeeze(-1)
        classes = torch.clamp((sensor_values * N_CLASSES).long(), max=N_CLASSES - 1)
        delta = self.vector.to(device=hidden.device, dtype=hidden.dtype)[classes.to(hidden.device)]
        updated[:, self.prompt_len - 1, :] += delta
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated


def bucket_sensor(sensor):
    return min(int(sensor.item() * N_CLASSES), N_CLASSES - 1)


def make_dataset(n_samples, seed):
    generator = torch.Generator().manual_seed(seed)
    sensors = torch.rand(n_samples, 1, generator=generator)
    labels = [bucket_sensor(sensor) for sensor in sensors]
    return sensors, labels


def encode_prompt(tokenizer):
    return tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids


def register_hook(model, vector, prompt_len, sensor=None):
    hook = SensorWriteHook(vector, prompt_len)
    hook.sensor = sensor
    handle = model.base_model.model.model.layers[TAP_LAYER].register_forward_hook(hook)
    return hook, handle


def compute_reference_vectors(model, tokenizer, device):
    inputs = encode_prompt(tokenizer).to(device)
    with torch.no_grad():
        output = model(input_ids=inputs, output_hidden_states=True, use_cache=False)
    reference = output.hidden_states[TAP_LAYER + 1][0, -1, :]
    rms = reference.pow(2).mean().sqrt().item()
    generator = torch.Generator(device="cpu").manual_seed(7391)
    basis = torch.randn(N_CLASSES, reference.shape[0], generator=generator)
    basis /= basis.norm(dim=1, keepdim=True)
    basis *= math.sqrt(reference.shape[0]) * rms * TENSOR_SCALE
    class_basis = basis.to(device=device, dtype=model.dtype)

    raw_basis = torch.zeros(N_CLASSES, reference.shape[0])
    raw_basis[torch.arange(N_CLASSES), torch.arange(N_CLASSES)] = 1.0
    raw_basis *= math.sqrt(reference.shape[0]) * rms * TENSOR_SCALE
    raw_basis = raw_basis.to(device=device, dtype=model.dtype)
    return class_basis, raw_basis, rms


def make_model(model_path, device):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )


def train_arm(model, tokenizer, arm, sensors, hook_sensors, labels, class_vector, raw_vector, device, epochs, lr):
    prompt_ids = encode_prompt(tokenizer).to(device)
    prompt_len = prompt_ids.shape[1]
    target_ids = [
        tokenizer(LABELS[label], return_tensors="pt", add_special_tokens=False).input_ids[0]
        for label in labels
    ]
    target_lengths = {target.shape[0] for target in target_ids}
    if len(target_lengths) != 1 or next(iter(target_lengths)) != 1:
        raise RuntimeError(f"Target labels must be exactly one token, got {target_lengths}")

    target_token_ids = [target.item() for target in target_ids]

    full_ids = torch.cat(
        [prompt_ids.expand(len(sensors), -1),
         torch.tensor(target_token_ids, device=device).unsqueeze(1)],
        dim=1,
    )
    full_labels = full_ids.clone()
    full_labels[:, :prompt_len] = -100

    hook, handle = None, None
    train_vector = None
    if arm in {"real", "sensor_permuted"}:
        train_vector = class_vector
    elif arm == "raw_onehot":
        train_vector = raw_vector
    if train_vector is not None:
        hook, handle = register_hook(model, train_vector, prompt_len, hook_sensors)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    losses = []
    order = list(range(len(sensors)))
    for _epoch in range(epochs):
        random.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            if hook is not None:
                hook.sensor = hook_sensors[batch_indices].to(device)
            input_batch = full_ids[batch_indices]
            label_batch = full_labels[batch_indices]
            output = model(input_ids=input_batch, labels=label_batch, use_cache=False)
            output.loss.backward()
            optimizer.step()
            epoch_losses.append(output.loss.item())
        losses.append(statistics.fmean(epoch_losses))
    if handle is not None:
        handle.remove()
    if hook is not None:
        hook.sensor = None
    return losses


def evaluate_arm(model, tokenizer, vector, sensors, labels, device, module_active=True, limit_examples=0):
    prompt_ids = encode_prompt(tokenizer).to(device)
    prompt_len = prompt_ids.shape[1]
    hook, handle = (None, None)
    if module_active:
        hook, handle = register_hook(model, vector, prompt_len)
    correct = 0
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sensors), BATCH_SIZE):
            batch_indices = list(range(start, min(start + BATCH_SIZE, len(sensors))))
            if hook is not None:
                hook.sensor = sensors[batch_indices].to(device)
            batch = prompt_ids.expand(len(batch_indices), -1).contiguous()
            logits = model(input_ids=batch, use_cache=False).logits[:, prompt_len - 1, :]
            predicted = logits.argmax(-1).tolist()
            label_token_ids = [
                tokenizer(LABELS[labels[index]], add_special_tokens=False).input_ids[0]
                for index in batch_indices
            ]
            predictions.extend(predicted)
            correct += sum(prediction == target for prediction, target in zip(predicted, label_token_ids))
    if handle is not None:
        handle.remove()
    if hook is not None:
        hook.sensor = None
    return correct / len(sensors), predictions


def mean_std(values):
    return statistics.fmean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def main():
    args = parse_args()
    requested_arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    valid_arms = {"real", "sensor_permuted", "raw_onehot", "trajectory_only"}
    if set(requested_arms) - valid_arms:
        raise ValueError(f"Unknown arms: {set(requested_arms) - valid_arms}")

    started = time.time()
    device = "cuda"
    print(f"[E1] Loading {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = make_model(MODEL_PATH, device)
    model.eval()
    class_vector, raw_vector, hidden_rms = compute_reference_vectors(model, tokenizer, device)
    delta_rms = class_vector.pow(2).mean().sqrt().item()
    print(f"[E1] reference hidden RMS={hidden_rms:.4f} delta RMS={delta_rms:.4f}", flush=True)

    runs = []
    arm_accuracies = {arm: [] for arm in requested_arms}
    arm_no_module_accuracies = {arm: [] for arm in requested_arms}
    for seed_index in range(args.seeds):
        run_seed = args.base_seed + seed_index
        data_seed = 1000 + seed_index
        train_sensors, train_labels = make_dataset(args.train_samples, data_seed)
        eval_sensors, eval_labels = make_dataset(args.eval_samples, data_seed + 500)
        permutation = torch.randperm(len(train_sensors), generator=torch.Generator().manual_seed(data_seed + 900))
        permuted_sensors = train_sensors[permutation]

        for arm in requested_arms:
            print(f"\n[run {seed_index + 1}/{args.seeds}] arm={arm} seed={run_seed}", flush=True)
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            random.seed(run_seed)
            peft_model = get_peft_model(
                model,
                LoraConfig(
                    r=LORA_R,
                    lora_alpha=LORA_ALPHA,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    layers_to_transform=list(range(TAP_LAYER, model.config.num_hidden_layers)),
                    task_type="CAUSAL_LM",
                ),
            )
            peft_model.print_trainable_parameters()
            current_hook_sensors = train_sensors
            if arm == "sensor_permuted":
                current_hook_sensors = permuted_sensors
            losses = train_arm(
                peft_model,
                tokenizer,
                arm,
                train_sensors,
                current_hook_sensors,
                train_labels,
                class_vector,
                raw_vector,
                device,
                args.epochs,
                LR,
            )
            eval_vector = class_vector
            if arm == "raw_onehot":
                eval_vector = raw_vector
            module_accuracy, predictions = evaluate_arm(
                peft_model,
                tokenizer,
                eval_vector,
                eval_sensors,
                eval_labels,
                device,
                module_active=True,
            )
            no_module_accuracy, _ = evaluate_arm(
                peft_model,
                tokenizer,
                eval_vector,
                eval_sensors,
                eval_labels,
                device,
                module_active=False,
            )
            print(
                f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}; "
                f"module_acc={module_accuracy:.4f}; no_module_acc={no_module_accuracy:.4f}",
                flush=True,
            )
            runs.append({
                "arm": arm,
                "seed": run_seed,
                "loss_start": losses[0],
                "loss_end": losses[-1],
                "module_active_accuracy": module_accuracy,
                "no_module_accuracy": no_module_accuracy,
                "predictions": predictions if len(predictions) <= 64 else predictions[:64],
            })
            arm_accuracies[arm].append(module_accuracy)
            arm_no_module_accuracies[arm].append(no_module_accuracy)
            peft_model = peft_model.unload()
            model = peft_model
            model.eval()
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    summary = {}
    for arm in requested_arms:
        module_mean, module_std = mean_std(arm_accuracies[arm])
        no_module_mean, no_module_std = mean_std(arm_no_module_accuracies[arm])
        summary[arm] = {
            "module_active_accuracy_mean": module_mean,
            "module_active_accuracy_std": module_std,
            "no_module_accuracy_mean": no_module_mean,
            "no_module_accuracy_std": no_module_std,
            "module_active_runs": arm_accuracies[arm],
            "no_module_runs": arm_no_module_accuracies[arm],
        }
    chance = 1.0 / N_CLASSES
    primary_gap = (
        summary["real"]["module_active_accuracy_mean"]
        - summary["sensor_permuted"]["module_active_accuracy_mean"]
        if "real" in summary and "sensor_permuted" in summary else None
    )
    trajectory_gap = (
        summary["real"]["module_active_accuracy_mean"]
        - summary["trajectory_only"]["module_active_accuracy_mean"]
        if "real" in summary and "trajectory_only" in summary else None
    )
    raw_gap = (
        summary["real"]["module_active_accuracy_mean"]
        - summary["raw_onehot"]["module_active_accuracy_mean"]
        if "real" in summary and "raw_onehot" in summary else None
    )
    abi_signal_effective = (
        primary_gap is not None
        and primary_gap >= 0.15
        and summary["real"]["module_active_accuracy_mean"] > chance + 0.15
        and summary["real"]["no_module_accuracy_mean"] <= chance + 0.10
    )
    module_is_noop = (
        raw_gap is not None
        and raw_gap <= 0.05
        and summary["raw_onehot"]["module_active_accuracy_mean"] > chance + 0.15
    )
    conclusion = (
        "PASS: paired class-basis tensor carries usable task information beyond permuted control; "
        "raw-onehot also solves it, so this experiment supports bandwidth/encoding, not external computation."
        if abi_signal_effective and module_is_noop
        else "PASS: paired tensor signal carries usable task information beyond permuted and raw controls."
        if abi_signal_effective
        else "FAIL/NONDESCRIPT: paired tensor signal is not distinguishable from controls under this setup."
    )
    decision = {
        "chance_accuracy": chance,
        "primary_gap_real_minus_sensor_permuted": primary_gap,
        "real_minus_trajectory": trajectory_gap,
        "real_minus_raw_onehot": raw_gap,
        "abi_signal_effective": abi_signal_effective,
        "module_is_noop": module_is_noop,
        "conclusion": conclusion,
    }
    results = {
        "experiment": "E1_cloud_attribution_ablation",
        "model": MODEL_PATH,
        "tap_layer": TAP_LAYER,
        "task": "sensor-only four-class next-token accuracy",
        "prompt": PROMPT,
        "labels": LABELS,
        "tensor_scale": TENSOR_SCALE,
        "reference_hidden_rms": hidden_rms,
        "delta_rms": delta_rms,
        "tensor_encoding": "class-basis lookup versus fixed one-hot projection at matched norm",
        "config": vars(args),
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "lr": LR, "batch_size": BATCH_SIZE},
        "runs": runs,
        "summary": summary,
        "decision": decision,
        "elapsed_s": round(time.time() - started, 1),
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print("\n[E1 SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(json.dumps(decision, indent=2))
    print(f"Results saved to {args.output}; elapsed={results['elapsed_s']}s", flush=True)
    return abi_signal_effective


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 2)

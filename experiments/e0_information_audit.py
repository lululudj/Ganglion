# -*- coding: utf-8 -*-
"""E0: Information Audit"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ganglion.host import TinyBackbone

torch.manual_seed(42)
DIM = 64
LAYERS = 3
TAP = 1
VOCAB = 32
N_SAMPLES = 500

def closed_form_r2(X, Y):
    Xb = np.hstack([X.numpy(), np.ones((X.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(Xb, Y.numpy(), rcond=None)
    Yp = Xb @ W
    ssr = ((Yp - Y.numpy())**2).sum()
    sst = ((Y.numpy() - Y.numpy().mean(axis=0))**2).sum()
    return 1 - ssr / max(sst, 1e-10)

def mlp_probe_r2(X, Y, epochs=300, lr=1e-3, hidden=128):
    probe = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, X.shape[1]))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(probe(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(X)
        ssr = ((pred - Y)**2).sum().item()
        sst = ((Y - Y.mean(dim=0))**2).sum().item()
        return 1 - ssr / max(sst, 1e-10)

def main():
    torch.manual_seed(42)
    bb = TinyBackbone(vocab=VOCAB, dim=DIM, n_layers=LAYERS, tap_layer=TAP, seed=0)
    bb.eval()

    all_h = []
    for seed in range(N_SAMPLES):
        torch.manual_seed(seed)
        tokens = torch.randint(0, VOCAB, (8,)).tolist()
        with torch.no_grad():
            h = bb.encode_upto(tokens, TAP)
        all_h.append(h[-1])
    H = torch.stack(all_h)
    print(f"Hidden states: {H.shape}, mean_norm={H.norm(dim=-1).mean():.4f}, std={H.std():.4f}")

    results = {}

    # M1: Linear h*5+2 -> delta = 4H + 2 (linear in H, closed-form should give R2=1)
    delta1 = H * 4 + 2
    r2_lin = closed_form_r2(H, delta1)
    r2_mlp = mlp_probe_r2(H, delta1)
    print(f"\n[M1] Linear h*5+2 (delta=4H+2):")
    print(f"  closed-form linear R2 = {r2_lin:.6f}")
    print(f"  MLP probe R2 = {r2_mlp:.6f}")
    results["M1_linear"] = {"r2_closed_form": round(r2_lin, 6), "r2_mlp": round(r2_mlp, 6)}

    # M2: Sensor-conditioned h*(1+s), s~U(0,1) PRIVILEGED
    torch.manual_seed(999)
    s = torch.rand(H.shape[0], 1)
    H_m2 = H * (1 + s)
    delta2 = H_m2 - H  # = H * s
    r2_lin2 = closed_form_r2(H, delta2)
    r2_mlp2 = mlp_probe_r2(H, delta2)
    print(f"\n[M2] Sensor-conditioned h*(1+s):")
    print(f"  closed-form linear R2 = {r2_lin2:.6f}")
    print(f"  MLP probe R2 = {r2_mlp2:.6f}")
    results["M2_sensor"] = {"r2_closed_form": round(r2_lin2, 6), "r2_mlp": round(r2_mlp2, 6)}

    # M3: Nonlinear MLP module
    torch.manual_seed(123)
    mod = nn.Sequential(nn.Linear(DIM, 32), nn.Tanh(), nn.Linear(32, DIM))
    with torch.no_grad():
        H_m3 = mod(H)
    delta3 = H_m3 - H
    r2_lin3 = closed_form_r2(H, delta3)
    r2_mlp3 = mlp_probe_r2(H, delta3)
    print(f"\n[M3] Nonlinear MLP module:")
    print(f"  closed-form linear R2 = {r2_lin3:.6f}")
    print(f"  MLP probe R2 = {r2_mlp3:.6f}")
    results["M3_nonlinear"] = {"r2_closed_form": round(r2_lin3, 6), "r2_mlp": round(r2_mlp3, 6)}

    # Summary
    print("\n" + "=" * 70)
    print("E0 INFORMATION AUDIT")
    print("=" * 70)
    print(f"{'Module':<25} {'Linear R2':>10} {'MLP R2':>10} {'Verdict'}")
    print("-" * 70)
    for name, r in results.items():
        l, m = r["r2_closed_form"], r["r2_mlp"]
        if l > 0.95:
            verdict = "CLOSED (no new info)"
        elif m > 0.95:
            verdict = "CLOSED (MLP decodes)"
        else:
            verdict = "OPEN (privileged info)"
        print(f"{name:<25} {l:>10.4f} {m:>10.4f} {verdict}")

    m1_closed = results["M1_linear"]["r2_closed_form"] > 0.95
    m2_open = results["M2_sensor"]["r2_mlp"] < 0.95

    print(f"\n  M1 linear = information-closed: {m1_closed}")
    print(f"  M2 sensor-conditioned = privileged: {m2_open}")

    verdict = "PASS" if m1_closed and m2_open else "FAIL"
    print(f"  E0 VERDICT: {verdict}")
    results["verdict"] = verdict

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e0_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
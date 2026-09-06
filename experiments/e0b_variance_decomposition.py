# -*- coding: utf-8 -*-
"""E0b: Variance Decomposition — delta 的方差中有多少来自 h、多少来自 s、多少是交互项？

修复 E0 的两个问题：
1. MLP probe 训练集 R2=1.0 是过拟合（训练=测试集泄漏）→ 改用 train/test split
2. 只报一个 R2 不够 → 分解 delta 的方差来源

方差分解：
  delta 的总方差 = var(delta)
  h 单独解释的方差：R2(delta ~ h)
  s 单独解释的方差：R2(delta ~ s)
  h + s 联合解释：R2(delta ~ h + s)
  交互项：R2(delta ~ h * s) - R2(delta ~ h + s)
  残差 = 1 - R2(delta ~ h * s)
"""
import json, os, sys
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
N_SAMPLES = 1000
TEST_RATIO = 0.3

def closed_form_r2(X, Y):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(Xb, Y, rcond=None)
    Yp = Xb @ W
    ssr = ((Yp - Y)**2).sum()
    sst = ((Y - Y.mean(axis=0))**2).sum()
    return 1 - ssr / max(sst, 1e-10)

def mlp_probe_test_r2(X_train, Y_train, X_test, Y_test, epochs=300, lr=1e-3, hidden=128):
    probe = nn.Sequential(nn.Linear(X_train.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, X_train.shape[1]))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.MSELoss()(probe(X_train), Y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(X_test)
        ssr = ((pred - Y_test)**2).sum().item()
        sst = ((Y_test - Y_test.mean(dim=0))**2).sum().item()
        return 1 - ssr / max(sst, 1e-10)

def main():
    torch.manual_seed(42)
    bb = TinyBackbone(vocab=VOCAB, dim=DIM, n_layers=LAYERS, tap_layer=TAP, seed=0)
    bb.eval()

    # Collect hidden states
    all_h = []
    for seed in range(N_SAMPLES):
        torch.manual_seed(seed)
        tokens = torch.randint(0, VOCAB, (8,)).tolist()
        with torch.no_grad():
            h = bb.encode_upto(tokens, TAP)
        all_h.append(h[-1])
    H = torch.stack(all_h)
    print(f"Hidden states: {H.shape}")

    # Privileged sensor signal
    torch.manual_seed(999)
    S = torch.rand(N_SAMPLES, 1)  # s ~ U(0,1), unknown to backbone

    # Train/test split
    idx = torch.randperm(N_SAMPLES)
    n_test = int(N_SAMPLES * TEST_RATIO)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    H_train, H_test = H[train_idx], H[test_idx]
    S_train, S_test = S[train_idx], S[test_idx]

    results = {}

    # ===== Module types =====
    modules = {
        "M1_linear_h5p2": lambda h, s: h * 5 + 2,
        "M2_sensor_h1ps": lambda h, s: h * (1 + s),
        "M3_nonlinear_mlp": None,  # special case
    }

    for name, tfn in modules.items():
        print(f"\n{'='*70}")
        print(f"[{name}]")
        print(f"{'='*70}")

        if name == "M3_nonlinear_mlp":
            torch.manual_seed(123)
            mod = nn.Sequential(nn.Linear(DIM, 32), nn.Tanh(), nn.Linear(32, DIM))
            with torch.no_grad():
                delta_all = mod(H) - H
            delta_train = delta_all[train_idx]
            delta_test = delta_all[test_idx]
            S_train_ = torch.zeros_like(S_train)  # no sensor
            S_test_ = torch.zeros_like(S_test)
        else:
            delta_all = tfn(H, S) - H
            delta_train = delta_all[train_idx]
            delta_test = delta_all[test_idx]
            S_train_ = S_train
            S_test_ = S_test

        # Convert to numpy for closed-form
        H_tr_np, H_te_np = H_train.numpy(), H_test.numpy()
        S_tr_np, S_te_np = S_train_.numpy(), S_test_.numpy()
        D_tr_np, D_te_np = delta_train.numpy(), delta_test.numpy()

        # ---- Variance decomposition ----
        # 1. delta ~ h alone
        r2_h = closed_form_r2(H_tr_np, D_tr_np)  # train R2
        r2_h_test = closed_form_r2(H_te_np, D_te_np)  # test R2 (same model, test data)

        # Actually, closed_form_r2 fits on the input data. For test R2,
        # we need to fit on train and evaluate on test.
        # Let me do this properly:
        Xb_train = np.hstack([H_tr_np, np.ones((H_tr_np.shape[0], 1))])
        W_h, _, _, _ = np.linalg.lstsq(Xb_train, D_tr_np, rcond=None)
        Xb_test = np.hstack([H_te_np, np.ones((H_te_np.shape[0], 1))])
        D_pred_h = Xb_test @ W_h
        ss_res_h = ((D_pred_h - D_te_np)**2).sum()
        ss_tot = ((D_te_np - D_te_np.mean(axis=0))**2).sum()
        r2_h_test = 1 - ss_res_h / max(ss_tot, 1e-10)

        # 2. delta ~ s alone
        Xb_s_train = np.hstack([S_tr_np, np.ones((S_tr_np.shape[0], 1))])
        W_s, _, _, _ = np.linalg.lstsq(Xb_s_train, D_tr_np, rcond=None)
        Xb_s_test = np.hstack([S_te_np, np.ones((S_te_np.shape[0], 1))])
        D_pred_s = Xb_s_test @ W_s
        ss_res_s = ((D_pred_s - D_te_np)**2).sum()
        r2_s_test = 1 - ss_res_s / max(ss_tot, 1e-10)

        # 3. delta ~ h + s (additive)
        X_add_train = np.hstack([H_tr_np, S_tr_np, np.ones((H_tr_np.shape[0], 1))])
        W_add, _, _, _ = np.linalg.lstsq(X_add_train, D_tr_np, rcond=None)
        X_add_test = np.hstack([H_te_np, S_te_np, np.ones((H_te_np.shape[0], 1))])
        D_pred_add = X_add_test @ W_add
        ss_res_add = ((D_pred_add - D_te_np)**2).sum()
        r2_add_test = 1 - ss_res_add / max(ss_tot, 1e-10)

        # 4. delta ~ h + s + h*s (interaction)
        H_times_S_tr = H_tr_np * S_tr_np  # [N, dim] * [N, 1] = [N, dim]
        H_times_S_te = H_te_np * S_te_np
        X_full_train = np.hstack([H_tr_np, S_tr_np, H_times_S_tr, np.ones((H_tr_np.shape[0], 1))])
        W_full, _, _, _ = np.linalg.lstsq(X_full_train, D_tr_np, rcond=None)
        X_full_test = np.hstack([H_te_np, S_te_np, H_times_S_te, np.ones((H_te_np.shape[0], 1))])
        D_pred_full = X_full_test @ W_full
        ss_res_full = ((D_pred_full - D_te_np)**2).sum()
        r2_full_test = 1 - ss_res_full / max(ss_tot, 1e-10)

        # 5. MLP probe (proper train/test)
        r2_mlp_test = mlp_probe_test_r2(H_train, delta_train, H_test, delta_test)

        # Interaction variance
        interaction_var = r2_full_test - r2_add_test
        residual_var = 1 - r2_full_test

        print(f"  Test R2(delta ~ h alone):        {r2_h_test:.4f}")
        print(f"  Test R2(delta ~ s alone):        {r2_s_test:.4f}")
        print(f"  Test R2(delta ~ h + s):          {r2_add_test:.4f}")
        print(f"  Test R2(delta ~ h + s + h*s):    {r2_full_test:.4f}")
        print(f"  Test R2(delta ~ MLP probe):      {r2_mlp_test:.4f}")
        print(f"  ---")
        print(f"  Variance from h alone:           {r2_h_test:.1%}")
        print(f"  Variance from s alone:           {r2_s_test:.1%}")
        print(f"  Variance from interaction h*s:   {interaction_var:.1%}")
        print(f"  Residual (unexplained):          {residual_var:.1%}")

        if name == "M1_linear_h5p2":
            info = "CLOSED" if r2_h_test > 0.95 else "OPEN"
            privileged_pct = max(0, 1 - r2_h_test)
        elif name == "M2_sensor_h1ps":
            info = "OPEN" if r2_h_test < 0.95 else "CLOSED"
            privileged_pct = max(0, 1 - r2_h_test)
        else:
            info = "OPEN" if r2_mlp_test < 0.95 else "CLOSED"
            privileged_pct = max(0, 1 - r2_mlp_test)

        print(f"  Information: {info} (privileged variance: {privileged_pct:.1%})")

        results[name] = {
            "r2_h_alone": round(r2_h_test, 4),
            "r2_s_alone": round(r2_s_test, 4),
            "r2_additive": round(r2_add_test, 4),
            "r2_full_interaction": round(r2_full_test, 4),
            "r2_mlp_test": round(r2_mlp_test, 4),
            "interaction_variance": round(interaction_var, 4),
            "residual_variance": round(residual_var, 4),
            "privileged_pct": round(privileged_pct, 4),
            "information": info,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("E0b VARIANCE DECOMPOSITION — SUMMARY")
    print("=" * 70)
    print(f"{'Module':<25} {'h alone':>8} {'s alone':>8} {'h*s交互':>8} {'残差':>8} {'特权%':>8} {'判定':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<25} {r['r2_h_alone']:>8.4f} {r['r2_s_alone']:>8.4f} "
              f"{r['interaction_variance']:>8.4f} {r['residual_variance']:>8.4f} "
              f"{r['privileged_pct']:>8.1%} {r['information']:>8}")

    # Verdict
    m1_closed = results["M1_linear_h5p2"]["r2_h_alone"] > 0.95
    m2_open = results["M2_sensor_h1ps"]["r2_h_alone"] < 0.95
    m3_closed = results["M3_nonlinear_mlp"]["r2_mlp_test"] > 0.95

    print(f"\n  M1 (linear) = information-closed: {m1_closed}")
    print(f"  M2 (sensor-conditioned) = privileged: {m2_open}")
    print(f"  M3 (nonlinear f(h)) = information-closed: {m3_closed}")

    # E0b verdict
    if m1_closed and m3_closed and m2_open:
        verdict = "PASS: all deterministic f(h) modules are CLOSED. Only f(h,s) is OPEN."
    elif m1_closed and m2_open:
        verdict = "PARTIAL: linear confirmed closed, but M3 needs investigation."
    else:
        verdict = "FAIL: unexpected results, check methodology."

    print(f"\n  E0b VERDICT: {verdict}")
    results["verdict"] = verdict

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e0b_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
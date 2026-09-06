# -*- coding: utf-8 -*-
"""E0c v2: Conditional task — labels depend on sensor value"""
import json, os, sys, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ganglion.host import TinyBackbone

torch.manual_seed(42)
DIM, LAYERS, TAP, VOCAB = 64, 3, 1, 32
N, N_GEN, TEST_RATIO = 400, 48, 0.3
EPOCHS, LR = 150, 1e-3

def make_bb(seed=0): return TinyBackbone(vocab=VOCAB, dim=DIM, n_layers=LAYERS, tap_layer=TAP, seed=seed)

def finetune_eval(bb, X_tr, Y_tr, X_te, Y_te):
    bb_ft = copy.deepcopy(bb); bb_ft.train()
    for p in bb_ft.emb.parameters(): p.requires_grad = False
    for i in range(TAP+1):
        for p in bb_ft.blocks[i].parameters(): p.requires_grad = False
    tr = []
    for i in range(TAP+1, LAYERS): tr += list(bb_ft.blocks[i].parameters())
    tr += list(bb_ft.head.parameters())
    opt = torch.optim.Adam(tr, lr=LR)
    for ep in range(EPOCHS):
        opt.zero_grad()
        h = X_tr
        for i in range(TAP+1, LAYERS): h = bb_ft.blocks[i](h)
        F.cross_entropy(bb_ft.head(h), Y_tr).backward(); opt.step()
    bb_ft.eval()
    with torch.no_grad():
        h = X_te
        for i in range(TAP+1, LAYERS): h = bb_ft.blocks[i](h)
        return (bb_ft.head(h).argmax(-1) == Y_te).float().mean().item()

def main():
    torch.manual_seed(42)
    bb = make_bb(); bb.eval()
    print("="*70); print("E0c v2: Conditional labels (s>0.5→7, s<=0.5→8)"); print("="*70)

    # ===== Smoke test =====
    print("\n--- Smoke: synthetic worlds ---")
    H = torch.randn(N, DIM) * 0.3
    S = torch.rand(N, 1)
    labels = (S.squeeze() > 0.5).long() + 7  # 7 or 8
    idx = torch.randperm(N); n_te = int(N*TEST_RATIO); te, tr = idx[:n_te], idx[n_te:]
    Y_tr, Y_te = labels[tr], labels[te]

    v_sig = torch.randn(DIM) * 0.3
    # World A: residual write delta = s * v (s carries the class info)
    delta_A = S * v_sig.unsqueeze(0)
    S_shuf = S[torch.randperm(N)]
    delta_A_shuf = S_shuf * v_sig.unsqueeze(0)

    hit_real_A = finetune_eval(bb, H[tr]+delta_A[tr], Y_tr, H[te]+delta_A[te], Y_te)
    hit_shuf_A = finetune_eval(bb, H[tr]+delta_A_shuf[tr], Y_tr, H[te]+delta_A_shuf[te], Y_te)
    hit_honly_A = finetune_eval(bb, H[tr], Y_tr, H[te], Y_te)
    gap_A = hit_real_A - hit_shuf_A
    print(f"  A(s=signal): real={hit_real_A:.4f} shuf={hit_shuf_A:.4f} h_only={hit_honly_A:.4f} gap={gap_A:+.4f}")

    # World B: delta = f(h) only, s irrelevant
    delta_B = H * 0.5
    hit_real_B = finetune_eval(bb, H[tr]+delta_B[tr], Y_tr, H[te]+delta_B[te], Y_te)
    hit_honly_B = finetune_eval(bb, H[tr], Y_tr, H[te], Y_te)
    gap_B = hit_real_B - hit_honly_B
    print(f"  B(s=noise): real={hit_real_B:.4f} h_only={hit_honly_B:.4f} gap={gap_B:+.4f}")

    smoke = gap_A > 0.1 and abs(gap_B) < 0.1
    print(f"  Smoke: {'PASS' if smoke else 'FAIL'}")
    if not smoke:
        print("  Instrument can't discriminate. ABORT.")
        return

    # ===== Real data =====
    print("\n--- Real: sensor-conditioned module on backbone ---")
    torch.manual_seed(0)
    tokens = [1,2,3,4]
    S_gen = torch.rand(N_GEN, 1)
    labels_gen = (S_gen.squeeze() > 0.5).long() + 7
    H_orig_l, H_trans_l = [], []
    v_rw = torch.randn(DIM) * 0.3
    for t in range(N_GEN):
        h = bb.encode_upto(tokens, TAP)
        H_orig_l.append(h[-1].detach().clone())
        h_in = h + S_gen[t] * v_rw.unsqueeze(0)
        H_trans_l.append(h_in[-1].detach().clone())
        logits = bb.decode_from(h_in, TAP)
        tokens.append(int(torch.argmax(logits[-1]).item()))
    H_orig = torch.stack(H_orig_l); H_trans = torch.stack(H_trans_l)

    idx2 = torch.randperm(N_GEN); n_te2 = int(N_GEN*TEST_RATIO); te2, tr2 = idx2[:n_te2], idx2[n_te2:]
    Y2_tr, Y2_te = labels_gen[tr2], labels_gen[te2]

    hit_real = finetune_eval(bb, H_trans[tr2], Y2_tr, H_trans[te2], Y2_te)
    S_shuf2 = S_gen[torch.randperm(N_GEN)]
    # Rebuild H_trans with shuffled sensor
    H_trans_shuf_l = []
    tokens2 = [1,2,3,4]
    for t in range(N_GEN):
        h = bb.encode_upto(tokens2, TAP)
        h_in = h + S_shuf2[t] * v_rw.unsqueeze(0)
        H_trans_shuf_l.append(h_in[-1].detach().clone())
        logits = bb.decode_from(h_in, TAP)
        tokens2.append(int(torch.argmax(logits[-1]).item()))
    H_trans_shuf = torch.stack(H_trans_shuf_l)
    hit_shuf = finetune_eval(bb, H_trans_shuf[tr2], Y2_tr, H_trans_shuf[te2], Y2_te)
    hit_honly = finetune_eval(bb, H_orig[tr2], Y2_tr, H_orig[te2], Y2_te)

    print(f"\n  real:     {hit_real:.4f}")
    print(f"  shuffled: {hit_shuf:.4f}")
    print(f"  h_only:   {hit_honly:.4f}")
    g1 = hit_real - hit_shuf; g2 = hit_real - hit_honly
    print(f"  real-shuffled: {g1:+.4f}")
    print(f"  real-h_only:   {g2:+.4f}")

    sv = "PRIVILEGED" if g1 > 0.1 else ("NOISE" if abs(g1) <= 0.1 else "HARMFUL")
    mv = "INJECTS" if g2 > 0.1 else "NO INJECTION"
    print(f"\n  Sensor: {sv}")
    print(f"  Module: {mv}")

    results = {"smoke": {"A_gap": round(gap_A,4), "B_gap": round(gap_B,4), "pass": smoke},
               "real": {"real": round(hit_real,4), "shuffled": round(hit_shuf,4),
                        "h_only": round(hit_honly,4), "gap_rs": round(g1,4),
                        "gap_rh": round(g2,4), "sensor": sv, "module": mv}}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e0c_results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
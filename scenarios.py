# -*- coding: utf-8 -*-
"""Ganglion 验证场景 S0-S4（进程级，真实独立进程）。

S0  ABI 注册协商：兼容接受；维度/dtype/层/超时/缺字段 → 原因码拒绝
S1  端到端外置服务：独立进程逐 token 处理 hidden states，跨进程数值一致
S2  故障隔离（三种故障模式）：外部硬杀 / 模块自杀 / 活着但无响应
S3  热换一致性：A→B 边界换挡 + 崩溃 + 热插入恢复，全程版本归因
S4  跨进程传输基准：shm/pipe/tcp × dim × seq × dtype + GPU 变体
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ganglion.abi import HostContract, ModuleManifest, negotiate
from ganglion.host import GanglionHost, TinyBackbone
from ganglion.module_proc import spawn_module

PROMPT = [1, 2, 3]


def _make_host(dim=64, seed=7):
    torch.manual_seed(0)
    bb = TinyBackbone(vocab=32, dim=dim, n_layers=3, tap_layer=1, seed=seed)
    return GanglionHost(
        bb, HostContract(layer="blk.1.out", dim=dim, dtype="float32"),
        seed=seed)


def _case(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": str(detail)[:300]}


def _manifest(mid, dim, timeout_ms, transform, version="1.0",
              dtype="float32", layer="blk.1.out"):
    return ModuleManifest(mid, version, layer, dim, dtype,
                          timeout_ms, transform)


# ================================================================ S0
def scenario_s0():
    cases = []
    c = HostContract(layer="blk.1.out", dim=768, dtype="float32")
    tf = {"kind": "linear", "scale": 1.5, "shift": 0.1}

    ok, code, _ = negotiate(c, _manifest("m1", 768, 100, tf))
    cases.append(_case("兼容清单被接受", ok and code == "OK", code))

    ok, code, _ = negotiate(c, _manifest("m2", 1024, 100, tf))
    cases.append(_case("维度不匹配被拒 E_DIM_MISMATCH",
                       (not ok) and code == "E_DIM_MISMATCH", code))

    ok, code, _ = negotiate(c, _manifest("m3", 768, 100, tf,
                                          dtype="float16"))
    cases.append(_case("dtype不匹配被拒 E_DTYPE_MISMATCH",
                       (not ok) and code == "E_DTYPE_MISMATCH", code))

    m4 = _manifest("m4", 768, 100, tf)
    m4.consumes_layer = "blk.99.out"
    ok, code, _ = negotiate(c, m4)
    cases.append(_case("未知层被拒 E_LAYER_UNKNOWN",
                       (not ok) and code == "E_LAYER_UNKNOWN", code))

    m5 = _manifest("m5", 768, 0, tf)
    ok, code, _ = negotiate(c, m5)
    cases.append(_case("非法超时被拒 E_TIMEOUT_INVALID",
                       (not ok) and code == "E_TIMEOUT_INVALID", code))

    m6 = _manifest("m6", 768, 100, tf)
    m6.dim = None
    ok, code, _ = negotiate(c, m6)
    cases.append(_case("清单缺字段被拒 E_FIELD_MISSING",
                       (not ok) and code == "E_FIELD_MISSING", code))

    return {"scenario": "S0_ABI注册协商", "cases": cases,
            "pass": all(x["pass"] for x in cases)}


# ================================================================ S1
def scenario_s1(dim=64, n_tokens=12):
    cases = []
    host = _make_host(dim)
    base = host.generate(PROMPT, n_tokens)

    tf = {"kind": "linear", "scale": 5.0, "shift": 2.0}
    proxy = spawn_module(_manifest("mA", dim, 2000, tf), seq_hint=64)
    try:
        host2 = _make_host(dim)
        host2.attach(proxy)
        out = host2.generate(PROMPT, n_tokens)
        served = proxy.get_stats()

        cases.append(_case("模块是真实独立进程(pid隔离)",
                           proxy.proc.pid != os.getpid(),
                           f"module_pid={proxy.proc.pid}"))
        cases.append(_case("模块冷启动就绪握手",
                           proxy.startup_ms is not None
                           and proxy.startup_ms < 15000,
                           f"spawn→READY={proxy.startup_ms:.0f}ms"))
        cases.append(_case("每个token都由模块服务",
                           served == n_tokens, f"served={served}/{n_tokens}"))
        cases.append(_case("全部token状态MODULE_OK",
                           all(r.status == "MODULE_OK" for r in host2.trace)))
        exact = all(torch.allclose(
            r.refined_ref, r.hidden_ref * tf["scale"] + tf["shift"],
            atol=1e-5) for r in host2.trace)
        cases.append(_case("跨进程变换数值一致(fp32精确)",
                           exact, "refined == h*5.0+2.0 for all tokens"))
        diff = sum(a != b for a, b in zip(out, base))
        logits_changed = any(
            not torch.allclose(
                host2.bb.decode_from(r.refined_ref, 1),
                host2.bb.decode_from(r.hidden_ref, 1))
            for r in host2.trace)
        cases.append(_case("插上模块改变模型输出分布(logits级)",
                           logits_changed,
                           f"argmax翻转{diff}/{n_tokens}(32词表随机骨干"
                           f"上贪心token可保持稳定)"))

        host2.detach()
        out_after = host2.generate(PROMPT, n_tokens)
        cases.append(_case("拔掉模块恢复基线输出",
                           out_after == base,
                           "detach后输出与无模块基线逐token一致"))
    finally:
        proxy.close()
    return {"scenario": "S1_端到端外置服务", "cases": cases,
            "pass": all(x["pass"] for x in cases)}


# ================================================================ S2
def _s2_run(dim, n_tokens, fault_at, timeout_ms, mode, tf):
    """单次 S2 运行。mode ∈ {kill, crash, slow}。返回 (gen, host, detect_ms)。"""
    kwargs = {}
    if mode == "crash":
        kwargs["crash_after"] = fault_at
    if mode == "slow":
        kwargs["slow_ms"] = 2000
    proxy = spawn_module(_manifest("mA", dim, timeout_ms, tf),
                          seq_hint=64, **kwargs)
    host = _make_host(dim)
    try:
        host.attach(proxy)
        host.tokens = list(PROMPT)
        gen = []
        for t in range(n_tokens):
            if t == fault_at and mode == "kill":
                proxy.kill()
            gen.append(host.step())
    finally:
        proxy.close()
    return gen, host, host.detect_ms


def scenario_s2(dim=64, n_tokens=12, fault_at=5, timeout_ms=300):
    cases = []
    tf = {"kind": "linear", "scale": 1.5, "shift": 0.1}

    for mode, label in (
            ("kill", "外部硬杀(TerminateProcess)"),
            ("crash", "模块自杀(os._exit注入)"),
            ("slow", "活着但无响应(超时看门狗)")):
        gen, host, detect_ms = _s2_run(dim, n_tokens, fault_at,
                                       timeout_ms, mode, tf)
        tr = host.trace
        # 检测语义：响应截止期看门狗是可靠界（实测 Windows 上 os._exit
        # 后 is_alive() 可见性延迟 ~125ms，存活轮询不可靠）。
        # kill 模式走 kill+join 即时路径；crash/slow 走超时路径。
        if mode == "slow":
            pre_ok = tr[0].status == "FALLBACK"   # 首 token 即超时降级
            pre_name = "首token即超时降级"
            post = tr
        else:
            pre_ok = all(r.status == "MODULE_OK" for r in tr[:fault_at])
            pre_name = "故障前模块正常服务"
            post = tr[fault_at:]
        post_degraded = all(r.status in ("FALLBACK", "PASSTHROUGH")
                            for r in post)
        identity = all(torch.equal(r.refined_ref, r.hidden_ref)
                       for r in post)
        detect_ok = (detect_ms is not None
                     and 0 <= detect_ms <= timeout_ms + 200)

        cases.append(_case(f"[{label}] 宿主生成未中断",
                           len(gen) == n_tokens,
                           f"{len(gen)}/{n_tokens} tokens"))
        cases.append(_case(f"[{label}] {pre_name}", pre_ok))
        cases.append(_case(f"[{label}] 故障后全部降级直通", post_degraded))
        cases.append(_case(f"[{label}] 降级token恒等门(refined==hidden)",
                           identity))
        cases.append(_case(f"[{label}] 看门狗检测在超时预算内",
                           detect_ok, f"detect={detect_ms:.1f}ms "
                           f"(契约界={timeout_ms}ms+200ms开销)"))

    # 外部硬杀模式两次运行输出完全一致（fail-closed 的确定性）
    g1, _, _ = _s2_run(dim, n_tokens, fault_at, timeout_ms, "kill", tf)
    g2, _, _ = _s2_run(dim, n_tokens, fault_at, timeout_ms, "kill", tf)
    cases.append(_case("故障降级路径确定性(两次运行一致)", g1 == g2))

    return {"scenario": "S2_故障隔离", "cases": cases,
            "pass": all(x["pass"] for x in cases)}


# ================================================================ S3
def scenario_s3(dim=64, n_tokens=16, swap_at=5, kill_at=10, insert_at=12):
    cases = []
    ta = {"kind": "linear", "scale": 2.0, "shift": 0.0}
    tb = {"kind": "linear", "scale": -1.0, "shift": 0.5}
    tc = {"kind": "linear", "scale": 0.5, "shift": 0.25}
    pa = spawn_module(_manifest("A", dim, 2000, ta), seq_hint=64)
    pb = spawn_module(_manifest("B", dim, 2000, tb, version="2.0"),
                      seq_hint=64)
    pc = spawn_module(_manifest("C", dim, 2000, tc, version="3.0"),
                      seq_hint=64)
    host = _make_host(dim)
    try:
        host.attach(pa)
        host.tokens = list(PROMPT)
        for t in range(n_tokens):
            if t == swap_at:
                host.request_swap(pb)     # 计划内热换：边界生效
            if t == kill_at:
                pb.kill()                # 计划内故障：硬杀 B
            if t == insert_at:
                host.request_swap(pc)    # 修复性热插入：C 接管
            host.step()

        tr = host.trace
        versions = [r.version for r in tr]

        cases.append(_case("换挡前缀全部由A服务",
                           all(v == "A" for v in versions[:swap_at])))
        cases.append(_case("换挡后缀全部由B服务",
                           all(v == "B" for v in versions[swap_at:kill_at])))
        cases.append(_case("计划内换挡零间隙(kill前无直通token)",
                           all(v is not None for v in versions[:kill_at])))
        # 修复间隙：kill B 后到 C 接管前，恰 1 个直通 token
        gap = sum(1 for v in versions if v is None)
        cases.append(_case("故障修复间隙恰1个token", gap == 1,
                           f"gap={gap} (token {kill_at} FALLBACK, "
                           f"token {insert_at - 1} PASSTHROUGH)"))
        cases.append(_case("C接管后缀全部MODULE_OK",
                           all(v == "C" for v in versions[insert_at:])
                           and all(r.status == "MODULE_OK"
                                   for r in tr[insert_at:])))
        # 无混合 token：每个 refined 恰匹配 A/B/C/恒等 之一
        no_mix = True
        for r in tr:
            h = r.hidden_ref
            m = [torch.allclose(r.refined_ref, h * ta["scale"] + ta["shift"]),
                 torch.allclose(r.refined_ref, h * tb["scale"] + tb["shift"]),
                 torch.allclose(r.refined_ref, h * tc["scale"] + tc["shift"]),
                 torch.equal(r.refined_ref, h)]
            if sum(bool(x) for x in m) != 1:
                no_mix = False
        cases.append(_case("无混合版本token(每个输出恰归因一个来源)",
                           no_mix))
        # 模块侧真实服务计数：A/C 存活用 stats；B 被硬杀后管道断开
        # （stats 不可得是预期），其服务计数由宿主 trace 证明。
        sa, sb, sc = pa.get_stats(), pb.get_stats(), pc.get_stats()
        b_via_trace = sum(1 for v in versions[swap_at:kill_at] if v == "B")
        cases.append(_case("三模块服务计数精确",
                           sa == swap_at
                           and b_via_trace == kill_at - swap_at
                           and sc == n_tokens - insert_at,
                           f"A={sa}(stats) B={b_via_trace}(trace,B被杀"
                           f"后stats不可得) C={sc}"))
        # 三个独立 pid
        cases.append(_case("三模块三独立进程",
                           len({pa.proc.pid, pb.proc.pid, pc.proc.pid,
                                os.getpid()}) == 4,
                           f"pids={pa.proc.pid},{pb.proc.pid},{pc.proc.pid}"))
    finally:
        for p in (pa, pb, pc):
            p.close()
    return {"scenario": "S3_热换一致性", "cases": cases,
            "pass": all(x["pass"] for x in cases)}


# ================================================================ S4
def scenario_s4(n_iters=40):
    from ganglion.transport import benchmark_channel, benchmark_gpu_shm
    rows = []
    for kind in ("shm", "pipe", "tcp"):
        for dtype in ("float32", "float16"):
            for dim in (768, 2048, 4096):
                for seq in (64, 256, 1024):
                    rows.append(benchmark_channel(
                        kind, dim, seq, dtype, n_iters=n_iters, warmup=8))
    gpu_rows = []
    if torch.cuda.is_available():
        for dim in (768, 2048, 4096):
            for seq in (64, 256, 1024):
                gpu_rows.append(benchmark_gpu_shm(
                    dim, seq, "float32", n_iters=n_iters, warmup=8))

    cases = [
        _case("CPU通道基准完成54项配置", len(rows) == 54, f"{len(rows)} rows"),
        _case("GPU变体基准完成9项配置", len(gpu_rows) == 9,
              f"{len(gpu_rows)} rows"),
        _case("全部配置含p50/p95/p99/吞吐",
              all(("p50_ms" in r and "p95_ms" in r and "p99_ms" in r
                   and "throughput_MBps" in r) for r in rows + gpu_rows)),
    ]
    # 摘要：每通道最差 p99（信息性，不做机器相关 PASS/FAIL 断言）
    summary = {}
    for kind in ("shm", "pipe", "tcp"):
        sub = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "worst_p99_ms": max(r["p99_ms"] for r in sub),
            "best_p99_ms": min(r["p99_ms"] for r in sub),
            "max_throughput_MBps": max(r["throughput_MBps"] for r in sub),
        }
    if gpu_rows:
        summary["shm+gpu"] = {
            "worst_p99_ms": max(r["p99_ms"] for r in gpu_rows),
            "best_p99_ms": min(r["p99_ms"] for r in gpu_rows),
        }
    return {"scenario": "S4_跨进程传输基准", "cases": cases, "rows": rows,
            "gpu_rows": gpu_rows, "summary": summary,
            "pass": all(x["pass"] for x in cases)}

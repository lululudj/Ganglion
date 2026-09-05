# -*- coding: utf-8 -*-
"""Ganglion 最小验证系统 — 一键编排 S0-S4 → 控制台 PASS/FAIL 矩阵 + results.json。

用法:  python ganglion/run_validation.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ganglion.scenarios import (scenario_s0, scenario_s1, scenario_s2,
                                scenario_s3, scenario_s4)


def main():
    t0 = time.perf_counter()
    results = {}
    for name, fn in (("S0", scenario_s0), ("S1", scenario_s1),
                     ("S2", scenario_s2), ("S3", scenario_s3),
                     ("S4", scenario_s4)):
        print(f"[{name}] 运行中 ...", flush=True)
        t1 = time.perf_counter()
        try:
            results[name] = fn()
        except Exception:
            import traceback
            results[name] = {"scenario": name, "pass": False, "cases": [],
                             "error": traceback.format_exc()[-2000:]}
        print(f"[{name}] 完成 {time.perf_counter() - t1:.1f}s", flush=True)

    print()
    print("=" * 72)
    print("Ganglion 最小验证 — PASS/FAIL 矩阵")
    print("=" * 72)
    total_cases = total_pass = 0
    for k in ("S0", "S1", "S2", "S3", "S4"):
        r = results[k]
        sp = r.get("pass", False)
        cases = r.get("cases", [])
        cp = sum(1 for c in cases if c["pass"])
        total_cases += len(cases)
        total_pass += cp
        print(f"\n[{k}] {r.get('scenario', '')} —— "
              f"{'PASS' if sp else 'FAIL'} ({cp}/{len(cases)} cases)")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        for c in cases:
            mark = "PASS" if c["pass"] else "FAIL"
            line = f"  [{mark}] {c['name']}"
            if c.get("detail"):
                line += f"  —  {c['detail']}"
            print(line)
    print()
    marks = "/".join("P" if results[k].get("pass") else "F"
                     for k in ("S0", "S1", "S2", "S3", "S4"))
    print(f"总计: {total_pass}/{total_cases} cases  |  场景 {marks}  |  "
          f"总耗时 {time.perf_counter() - t0:.1f}s")

    # S4 摘要
    s4 = results.get("S4", {})
    if s4.get("summary"):
        print("\nS4 传输基准摘要（每通道最差 p99 / 峰值吞吐）:")
        for kind, s in s4["summary"].items():
            line = f"  {kind:>8}: worst_p99={s['worst_p99_ms']:.3f}ms"
            if "max_throughput_MBps" in s:
                line += f"  peak={s['max_throughput_MBps']:.0f}MB/s"
            print(line)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()

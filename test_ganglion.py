# -*- coding: utf-8 -*-
"""Ganglion 最小验证系统 — 单元级语义测试（TDD 先行）。

覆盖四类核心语义的进程内可测部分：
  1. ABI 注册协商（兼容接受 / 维度/dtype/层/超时不兼容拒绝 + 原因码）
  2. fallback 门（模块死亡 → 恒等直通，输出与无模块基线一致）
  3. 热换 token 原子性（版本归因唯一、无混合 token、换挡间隙 ≤ 1 token）
  4. 变换规范确定性（linear spec 数值精确）

进程级场景（S1 端到端 / S2 故障隔离 / S4 传输基准）由 run_validation.py 驱动，
使用真实独立进程，不在此重复。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ganglion.abi import HostContract, ModuleManifest, negotiate
from ganglion.host import GanglionHost, TinyBackbone


def make_contract(dim=64):
    return HostContract(layer="blk.1.out", dim=dim, dtype="float32")


def make_manifest(mid="m1", dim=64, dtype="float32", layer="blk.1.out",
                  scale=1.5, shift=0.1, timeout_ms=200, version="1.0"):
    return ModuleManifest(
        module_id=mid, version=version, consumes_layer=layer, dim=dim,
        dtype=dtype, timeout_ms=timeout_ms,
        transform={"kind": "linear", "scale": scale, "shift": shift},
    )


class InProcProxy:
    """进程内假模块：仅用于宿主语义测试。真实独立进程由 module_proc.py 提供。"""

    def __init__(self, manifest, dead=False):
        self.manifest = manifest
        self.dead = dead
        self.calls = 0

    def round_trip(self, hidden, timeout_ms=None):
        self.calls += 1
        t = self.manifest.transform
        return hidden * t["scale"] + t["shift"]

    def health(self):
        return not self.dead

    def close(self):
        pass


class TestABINegotiation(unittest.TestCase):
    """S0：注册协商——契约匹配则接受，不匹配则拒绝并给出原因码。"""

    def test_compatible_manifest_accepted(self):
        ok, code, _ = negotiate(make_contract(), make_manifest())
        self.assertTrue(ok)
        self.assertEqual(code, "OK")

    def test_dim_mismatch_rejected_with_code(self):
        ok, code, detail = negotiate(make_contract(dim=64), make_manifest(dim=128))
        self.assertFalse(ok)
        self.assertEqual(code, "E_DIM_MISMATCH")
        self.assertIn("dim", detail.lower())

    def test_dtype_mismatch_rejected_with_code(self):
        ok, code, _ = negotiate(make_contract(), make_manifest(dtype="float16"))
        self.assertFalse(ok)
        self.assertEqual(code, "E_DTYPE_MISMATCH")

    def test_unknown_layer_rejected_with_code(self):
        ok, code, _ = negotiate(make_contract(), make_manifest(layer="blk.99.out"))
        self.assertFalse(ok)
        self.assertEqual(code, "E_LAYER_UNKNOWN")

    def test_invalid_timeout_rejected_with_code(self):
        ok, code, _ = negotiate(make_contract(), make_manifest(timeout_ms=0))
        self.assertFalse(ok)
        self.assertEqual(code, "E_TIMEOUT_INVALID")

    def test_missing_field_rejected_with_code(self):
        m = make_manifest()
        m.dim = None  # 模拟清单缺字段
        ok, code, _ = negotiate(make_contract(), m)
        self.assertFalse(ok)
        self.assertEqual(code, "E_FIELD_MISSING")


class TestHostSemantics(unittest.TestCase):
    """宿主核心语义：直通、模块全程服务、故障降级、热换原子性。"""

    def _make_host(self):
        torch.manual_seed(0)
        bb = TinyBackbone(vocab=32, dim=64, n_layers=3, tap_layer=1, seed=7)
        return GanglionHost(bb, make_contract(dim=64), seed=7)

    def test_baseline_generate_without_module(self):
        host = self._make_host()
        toks = host.generate(prompt=[1, 2, 3], n_tokens=8)
        self.assertEqual(len(toks), 8)
        self.assertTrue(all(r.version is None for r in host.trace))
        self.assertTrue(all(r.status == "PASSTHROUGH" for r in host.trace))

    def test_attached_module_processes_every_token(self):
        host = self._make_host()
        proxy = InProcProxy(make_manifest())
        host.attach(proxy)
        host.generate([1, 2, 3], n_tokens=8)
        self.assertEqual(proxy.calls, 8)
        self.assertTrue(all(r.status == "MODULE_OK" for r in host.trace))
        self.assertTrue(all(r.version == "m1" for r in host.trace))

    def test_fallback_gate_is_identity_when_module_dead(self):
        host = self._make_host()
        proxy = InProcProxy(make_manifest())
        host.attach(proxy)
        proxy.dead = True
        host.generate([1, 2, 3], n_tokens=8)
        # 死模块：0 次真实调用；首个 token 即降级，之后直通
        self.assertEqual(proxy.calls, 0)
        self.assertEqual(self.trace_status(host)[0], "FALLBACK")
        self.assertTrue(all(s in ("FALLBACK", "PASSTHROUGH") for s in self.trace_status(host)))
        # 降级后的输出必须与无模块基线逐 token 一致（fail-closed 且确定）
        base = self._make_host().generate([1, 2, 3], n_tokens=8)
        self.assertEqual(host.generated, base)

    @staticmethod
    def trace_status(host):
        return [r.status for r in host.trace]

    def test_hot_swap_token_atomicity_no_mixed_tokens(self):
        host = self._make_host()
        ma = make_manifest(mid="A", scale=2.0, shift=0.0)
        mb = make_manifest(mid="B", scale=-1.0, shift=0.5)
        pa, pb = InProcProxy(ma), InProcProxy(mb)
        host.attach(pa)
        host.tokens = [1, 2, 3]
        for t in range(10):
            if t == 5:
                host.request_swap(pb)  # 换挡请求：在 token 边界生效
            host.step()

        versions = [r.version for r in host.trace]
        self.assertIn("A", versions)
        self.assertIn("B", versions)
        # 换挡发生在 token 边界：间隙 ≤ 1 token → 版本索引差 ≤ 2
        idx_a = [i for i, r in enumerate(host.trace) if r.version == "A"]
        idx_b = [i for i, r in enumerate(host.trace) if r.version == "B"]
        self.assertLessEqual(min(idx_b) - max(idx_a), 2)
        self.assertLessEqual(len([v for v in versions if v is None]), 1)  # 换挡间隙无模块 ≤1 token

        # 无混合：每个 token 的 refined 必须精确匹配 A 或 B 之一的变换，且仅一个
        for r in host.trace:
            h = r.hidden_ref
            match_a = torch.allclose(r.refined_ref, h * 2.0, atol=1e-6)
            match_b = torch.allclose(r.refined_ref, h * -1.0 + 0.5, atol=1e-6)
            self.assertTrue(match_a ^ match_b,
                            f"token {r.token_idx} 出现混合版本或未知变换")


class TestTransformSpec(unittest.TestCase):
    """模块侧变换规范：必须数值精确、可独立复算（用于一致性校验）。"""

    def test_linear_transform_exact(self):
        from ganglion.module_proc import apply_transform
        h = torch.randn(4, 64)
        out = apply_transform(h, {"kind": "linear", "scale": 1.5, "shift": 0.1})
        self.assertTrue(torch.equal(out, h * 1.5 + 0.1))

    def test_noop_transform_exact(self):
        from ganglion.module_proc import apply_transform
        h = torch.randn(4, 64)
        out = apply_transform(h, {"kind": "noop"})
        self.assertTrue(torch.equal(out, h))

    def test_unknown_transform_rejected(self):
        from ganglion.module_proc import apply_transform
        with self.assertRaises(ValueError):
            apply_transform(torch.randn(2, 64), {"kind": "nonexistent"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

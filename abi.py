# -*- coding: utf-8 -*-
"""Ganglion ABI：外置神经模块的接口契约与注册协商。

问题定义核心：模块以 manifest 声明自己消费哪一层、什么 shape/dtype 的
hidden states、响应超时预算；宿主以 HostContract 声明插桩点。
negotiate() 是双方握手：兼容 → OK；任何不兼容 → 原因码，宿主拒绝挂载。

注意：dtype 限 float32/float16（numpy 传输层约束，bfloat16 暂不支持）。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

OK = "OK"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_DTYPE_MISMATCH = "E_DTYPE_MISMATCH"
E_LAYER_UNKNOWN = "E_LAYER_UNKNOWN"
E_TIMEOUT_INVALID = "E_TIMEOUT_INVALID"
E_FIELD_MISSING = "E_FIELD_MISSING"

SUPPORTED_DTYPES = ("float32", "float16")


@dataclass
class HostContract:
    """宿主侧契约：插桩层标识、hidden state 维度与 dtype。"""

    layer: str
    dim: int
    dtype: str = "float32"
    max_seq: int = 4096


@dataclass
class ModuleManifest:
    """模块侧清单：身份、消费层、接口规格、超时预算、变换规范。"""

    module_id: str
    version: str
    consumes_layer: str
    dim: int
    dtype: str
    timeout_ms: int
    transform: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "consumes_layer": self.consumes_layer,
            "dim": self.dim,
            "dtype": self.dtype,
            "timeout_ms": self.timeout_ms,
            "transform": self.transform,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleManifest":
        return cls(
            module_id=d["module_id"], version=d["version"],
            consumes_layer=d["consumes_layer"], dim=d["dim"],
            dtype=d["dtype"], timeout_ms=d["timeout_ms"],
            transform=d.get("transform", {}),
        )


def negotiate(contract: HostContract,
              manifest: ModuleManifest) -> Tuple[bool, str, str]:
    """注册协商。返回 (是否接受, 原因码, 人类可读细节)。"""
    required = [
        ("module_id", manifest.module_id), ("version", manifest.version),
        ("consumes_layer", manifest.consumes_layer), ("dim", manifest.dim),
        ("dtype", manifest.dtype), ("timeout_ms", manifest.timeout_ms),
    ]
    for name, val in required:
        if val is None or val == "":
            return False, E_FIELD_MISSING, f"manifest field '{name}' missing"
    if manifest.dtype not in SUPPORTED_DTYPES:
        return False, E_DTYPE_MISMATCH, (
            f"unsupported dtype '{manifest.dtype}' "
            f"(host supports {SUPPORTED_DTYPES})")
    if manifest.consumes_layer != contract.layer:
        return False, E_LAYER_UNKNOWN, (
            f"layer mismatch: module consumes '{manifest.consumes_layer}' "
            f"but host taps '{contract.layer}'")
    if int(manifest.dim) != int(contract.dim):
        return False, E_DIM_MISMATCH, (
            f"dim mismatch: module expects {manifest.dim}, "
            f"host provides {contract.dim}")
    if manifest.dtype != contract.dtype:
        return False, E_DTYPE_MISMATCH, (
            f"dtype mismatch: module expects {manifest.dtype}, "
            f"host provides {contract.dtype}")
    if int(manifest.timeout_ms) <= 0:
        return False, E_TIMEOUT_INVALID, (
            f"timeout_ms must be positive, got {manifest.timeout_ms}")
    return True, OK, "contract satisfied"


class ABIError(Exception):
    """协商失败异常。"""

    def __init__(self, code: str, detail: str):
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail

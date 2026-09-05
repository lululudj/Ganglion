# -*- coding: utf-8 -*-
"""Ganglion 宿主：骨干插桩 + 看门狗 + fallback 门 + 热换管理 + 逐 token trace。

核心语义：
  - token 原子性：token t 开工时绑定的模块版本完整处理 token t；
    request_swap 的换挡在下一个 token 边界生效，任何 token 不会看到混合版本。
  - fail-closed：模块心跳死亡或响应超时 → 该 token 起降级为恒等直通
    （refined == hidden），生成继续，绝不中断（熔断直至热插入恢复）。
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from ganglion.abi import HostContract, negotiate, ABIError


class TinyBackbone(nn.Module):
    """微型骨干：固定种子初始化的小型堆叠网络（CPU，可复现）。

    tap_layer 指定 hidden states 提取层：该层输出即外置模块的输入。
    """

    def __init__(self, vocab=64, dim=64, n_layers=3, tap_layer=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.vocab = vocab
        self.dim = dim
        self.tap_layer = tap_layer
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(dim, vocab)

    def encode_upto(self, tokens: List[int], k: int) -> torch.Tensor:
        """前向到第 k 层（含），返回该层输出 hidden states [seq, dim]。"""
        h = self.emb(torch.tensor(tokens, dtype=torch.long))
        for i in range(k + 1):
            h = self.blocks[i](h)
        return h

    def decode_from(self, h: torch.Tensor, k: int) -> torch.Tensor:
        """从第 k 层输出继续前向到 logits [seq, vocab]。"""
        for i in range(k + 1, len(self.blocks)):
            h = self.blocks[i](h)
        return self.head(h)


@dataclass
class TokenRecord:
    """逐 token 追踪：版本归因、状态、延迟、hidden/refined 引用（审计证据）。"""

    token_idx: int
    version: Optional[str]        # 服务该 token 的模块 id；None=直通
    status: str                   # MODULE_OK / FALLBACK / PASSTHROUGH
    t_module_ms: float
    t_total_ms: float
    hidden_ref: torch.Tensor
    refined_ref: torch.Tensor
    wall_ts: float = 0.0


class GanglionHost:
    """宿主运行时。step() = 一个完整 token 周期。"""

    def __init__(self, backbone: TinyBackbone, contract: HostContract, seed=0):
        self.bb = backbone
        self.contract = contract
        self.active = None        # 当前模块代理
        self.pending = None       # 待换入模块（下一个 token 边界原子生效）
        self.trace: List[TokenRecord] = []
        self.events: List[tuple] = []   # (事件名, 参数..., 时间戳)
        self.tokens: List[int] = []
        self.generated: List[int] = []
        self._detect_ms: Optional[float] = None   # 最近一次故障检测耗时

    # ---------- 模块生命周期 ----------
    def attach(self, proxy):
        ok, code, detail = negotiate(self.contract, proxy.manifest)
        if not ok:
            raise ABIError(code, detail)
        self.active = proxy
        self.events.append(("ATTACH", proxy.manifest.module_id, time.time()))

    def request_swap(self, proxy):
        """请求热换：协商通过后登记 pending，下一个 token 边界原子换挡。"""
        ok, code, detail = negotiate(self.contract, proxy.manifest)
        if not ok:
            raise ABIError(code, detail)
        self.pending = proxy
        self.events.append(("SWAP_REQUESTED", proxy.manifest.module_id,
                             time.time()))

    def detach(self):
        if self.active is not None:
            self.events.append(("DETACH", self.active.manifest.module_id,
                                time.time()))
        self.active = None

    def _apply_pending(self):
        if self.pending is None:
            return
        old = self.active.manifest.module_id if self.active is not None else None
        new = self.pending.manifest.module_id
        self.active = self.pending
        self.pending = None
        self.events.append(("SWAP_APPLIED", old, new, time.time()))

    # ---------- 生成 ----------
    def step(self) -> int:
        """一个 token 周期：边界换挡 → 前向至插桩层 → 模块往返或恒等直通 →
        续前向 → 贪心解码。token 开工时绑定的版本完整服务该 token。"""
        self._apply_pending()
        idx = len(self.generated)
        t0 = time.perf_counter()
        h = self.bb.encode_upto(self.tokens, self.bb.tap_layer)
        refined = h
        status = "PASSTHROUGH"
        version = None
        t_module_ms = 0.0
        if self.active is not None:
            version = self.active.manifest.module_id
            t1 = time.perf_counter()
            try:
                if not self.active.health():
                    raise RuntimeError("module heartbeat dead")
                refined = self.active.round_trip(
                    h, timeout_ms=self.active.manifest.timeout_ms)
                status = "MODULE_OK"
            except Exception as e:
                # fail-closed：本 token 起降级为恒等直通，生成不中断
                status = "FALLBACK"
                refined = h
                t_module_ms = (time.perf_counter() - t1) * 1000.0
                self._detect_ms = t_module_ms
                self.events.append(("FALLBACK", version, type(e).__name__,
                                    time.perf_counter()))
                self.active = None   # 熔断：后续 token 直至热插入恢复
            else:
                t_module_ms = (time.perf_counter() - t1) * 1000.0
        logits = self.bb.decode_from(refined, self.bb.tap_layer)
        nxt = int(torch.argmax(logits[-1]).item())
        self.trace.append(TokenRecord(
            idx, version, status, t_module_ms,
            (time.perf_counter() - t0) * 1000.0,
            h.detach().clone(), refined.detach().clone(), time.time()))
        self.tokens.append(nxt)
        self.generated.append(nxt)
        return nxt

    def generate(self, prompt, n_tokens):
        self.tokens = list(prompt)
        self.generated = []
        self.trace = []
        for _ in range(n_tokens):
            self.step()
        return list(self.generated)

    # ---------- 观测 ----------
    @property
    def detect_ms(self) -> Optional[float]:
        return self._detect_ms

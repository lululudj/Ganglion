# Ganglion: External Neural Modules
## A Minimal Validation of Running Neural Capability Units as Independent Processes

**Version**: Draft v0.1
**Release Record (see §0 below)**: First publicly released 2026-09-05 18:23:36 (UTC+8)
**Experimental Environment**: NVIDIA RTX 4060 Laptop (8 GB), Windows 11, Python 3.12.10, PyTorch 2.12.1+cu126, NumPy 2.3.5
**Validation Result**: 40/40 assertions passed; code and full experimental data in `results.json`

---

## Abstract

Mainstream capability-extension methods for large language models (Adapters, LoRA, MoE experts) all place the extension unit inside the host model's process and computation graph: a module failure is a model failure, and module replacement requires service interruption. This paper proposes and validates a different problem setting: the **External Neural Module (ENM)** — a capability unit implemented as a service with its own independent OS-process lifetime, processing the host model's hidden states in real time at per-token granularity over shared memory, with runtime hot-swapping. We define four core semantics for this setting (ABI registration negotiation, fault isolation, hot-swap consistency, transport latency bounds), implement a minimal validation system named **Ganglion**, and complete 5 scenarios with 40 reproducible assertions in a real multiprocess environment — all passing. Measured results: inter-process hidden-state round-trip latency at typical load (768×64, fp16) is p50 = 0.014 ms; after a hard module crash the host degrades with zero interruption; planned hot-swaps take effect atomically at token boundaries with no mixed-version outputs throughout. We additionally report two previously unpublished empirical findings: module cold-start cost ≈ 1.35 s, and a process-termination visibility delay of ≈ 125 ms on Windows — the latter directly invalidates liveness polling as a failure detector and establishes the **response-deadline watchdog** as the reliable fault-detection semantics for external modules.

**Keywords**: external neural module; fault isolation; hot-swapping; inter-process communication; parameter-efficient fine-tuning; system reliability

---

## 0. Release Record and Priority Claim

> **This work was first made public on September 5, 2026 (Beijing time, UTC+8).**

Because the problem definition proposed here (§3) was, to the best of our knowledge after extensive literature and web search, not formally published anywhere prior to this release (see §2 for the closest neighboring work and the gaps between them), we record verifiable timestamps to establish priority:

| Event | Timestamp (UTC+8) | Verifiable Evidence |
|---|---|---|
| Full validation run completed (40/40) | 2026-09-05 ≈ 10:44 | `results.json` in-repo; local system clock cross-checked against GitLink/GitHub server timestamps |
| **First public release — GitLink** | **2026-09-05 18:23:36** | Commit `5e69c52` on https://www.gitlink.org.cn/lulululudj/Ganglion (server-side git timestamp) |
| GitHub public mirror | 2026-09-05 ≈ 19:10 | https://github.com/lululudj/Ganglion, branch `master`, HEAD `5e69c52` |
| This English draft | 2026-09-05 ≈ 19:30 | Latest commit on both remotes |

All timestamps are independently verifiable: `git log` on either public repository, or the platforms' REST APIs (`/repos/lululudj/Ganglion/commits`). The full commit chain (`187e758` → `46265b8` → `5e69c52`) was authored 2026-09-05 18:22–18:23 (UTC+8).

**Nearest prior art and why it does not preempt this work** (full analysis in §2): dLoRA (OSDI 2024) hot-swaps LoRA *adapters as data* inside a single serving process; llama.cpp router mode (released 2025-12-11) runs *whole models* as isolated processes; Petals (JMLR 2023) distributes *layers of a single model* across processes. None of these treats a *capability module* as an independently deployed process that consumes and returns *hidden states* on a per-token data plane, and none defines the four semantics of §3. As of the release date, an extensive search (12+ targeted queries over one year of literature) found no publication defining this problem setting.

---

## 1 Introduction

### 1.1 Status Quo: Extension Units Live Inside the Model's Process

To give large language models domain or task capabilities, the community has developed a rich family of modular extension methods: Adapters insert small modules between layers [1]; LoRA modifies weights with low-rank deltas [2]; MoE places expert subnetworks under a router [3,4]; modular networks and model stitching explored compositionality [5,6]; HuggingGPT-style systems let an LLM orchestrate external models [7]; RETRO introduced external memory [8]. These methods differ in direction but share one unstated constraint: **the extension unit is co-located with the host model — same process, same computation graph, same failure domain.**

This constraint has three systemic consequences:
1. **Fault coupling**: a module defect (numerical overflow, infinite loop, OOM) kills the host directly;
2. **Expensive replacement**: updating a module requires stopping service and reloading weights;
3. **Homogeneous deployment**: module and host must share runtime, framework version, and even hardware.

### 1.2 The Gap: Externality Was Never Defined as a Research Problem

The systems community has produced all the "parts": S-LoRA/dLoRA reuse adapters as service-layer *data* [9,10]; Petals lets a single model's *layers* flow across nodes [11]; EdgeMoE-family work loads expert *weights* on demand [12–14]; GPU FaaS provides process-level isolation infrastructure [15,16]; llama.cpp's router mode (2025-12) runs whole models as isolated processes with millisecond switching. **But no work has organized these parts into a unified problem definition**: if a capability unit is a true independent process — with its own pid, lifetime, failure domain, and deployment unit — communicating with the host through a well-defined data-plane contract that processes hidden states per token, what should the system's *semantics* be?

### 1.3 Contributions

1. **Problem definition** (§3): the first formalization of external neural modules as four verifiable semantics — ABI registration negotiation, fault isolation, hot-swap consistency (token atomicity), transport latency bounds;
2. **Minimal system** (§4): Ganglion — a three-process architecture (host / module / module′) with a manifest negotiation protocol, dual-slot shared-memory data plane, heartbeat + watchdog control plane, and an identity fallback gate;
3. **Validation methodology and results** (§5): a 5-scenario, 40-assertion PASS/FAIL matrix, all passing; latency distributions over 63 benchmark configurations;
4. **Two new empirical findings** (§6): the module cold-start bound (~1.35 s) and the process-termination visibility delay (~125 ms), with their direct design implications for fault-detection semantics.

This paper is deliberately positioned as a **minimal validation**: the backbone is a tiny randomly initialized network and the module transform is an independently recomputable linear operator — we deliberately remove the "model capability" variable so that **the system semantics themselves** are the object under test. If the semantics hold in this minimal system, they are necessary conditions for scaling to real models.

---

## 2 Related Work

| Direction | Representative Work | Relation to ENM |
|---|---|---|
| In-process modules | Adapter [1], LoRA [2], MoE [3,4] | Module inside the computation graph; no process isolation |
| Adapter serving | S-LoRA [9], dLoRA [10], LoRAX, Punica | Adapter is **data**, reused in-process; dLoRA swaps dynamically but within one process's graph |
| Whole-model process isolation | llama.cpp router mode (2025-12), Llama-Swap | Swaps **entire models** (prompt-level, text I/O); no hidden-state data plane, no token-level semantics; a crashed model's in-flight requests **fail** rather than degrade |
| Layer distribution | Petals [11], HALO [17] | Hidden states cross processes/nodes, but the unit is a *layer of one model*, not a capability module; no hot-swap or fault semantics |
| Expert on-demand loading | EdgeMoE [12], MoE-Infinity [13], OD-MoE [14] | Expert *weights* swap in/out, but experts remain under the internal router |
| Model orchestration | HuggingGPT [7], Neural Module Networks [6] | Invoke external models' *outputs*; do not process hidden states in real time |
| Retrieval augmentation | RETRO [8] | External memory, not external computation |
| GPU process infrastructure | GPU FaaS [15], NVIDIA MPS [20] | Isolation mechanisms without a neural-module abstraction |
| Serverless LoRA | Predictive-LoRA [16] | Cold-start cost optimization; no real-time data-plane protocol |

**Gap confirmation**: none of the above simultaneously satisfies (a) module as independent process; (b) per-token hidden-state data plane; (c) hot-swap consistency semantics; (d) fault-isolation semantics. This is the combination Ganglion fills.

---

## 3 Problem Definition: Four Semantics of External Neural Modules

**Setting**: a host process $H$ contains a backbone $f_\theta$ with a tap at layer $k$ extracting hidden states $h_t \in \mathbb{R}^{n_t \times d}$ ($t$ = token index); an external module process $M$ runs as a service. Define:

**Semantics 1 (ABI registration negotiation).** The module declares, via a manifest, its consumed layer, dimension, dtype, timeout budget, and transform spec; the host declares its tap point via a HostContract. Negotiation is binary: accept, or **reject with a reason code** (E_DIM_MISMATCH / E_DTYPE_MISMATCH / E_LAYER_UNKNOWN / E_TIMEOUT_INVALID / E_FIELD_MISSING). Registration is complete only after a READY handshake.

**Semantics 2 (Fault isolation, fail-closed).** On any module-process fault (external kill, suicide, unresponsiveness), the host does not crash; from the detection point on, every subsequent token satisfies the identity fallback
$$h_t^{out} = h_t \quad (\text{refined} \equiv \text{hidden}),$$
and the degraded path is **deterministically reproducible**. The host continues generating until a new module is hot-inserted.

**Semantics 3 (Hot-swap consistency, token atomicity).** The module version bound at the start of token $t$ **fully serves token $t$**; a swap command takes effect atomically at the next token boundary. No "mixed-version token" may exist — every token's output must be attributable to exactly one transform source.

**Semantics 4 (Transport latency bounds).** The data-plane round-trip latency $\ell(t)$ has a measurable distribution; at typical load, $\ell$ should be negligible relative to backbone forward time, with p50/p95/p99 and throughput reported per channel.

---

## 4 Ganglion System Design

```
┌────────────────────────────┐                 ┌──────────────────┐
│ HOST process               │  Data plane     │ MODULE process   │
│  tiny backbone (3-layer MLP)│  (SHM dual-slot)│  manifest + READY│
│  tap at layer 1 → h_t      │ ──hidden_t──▶   │  heartbeat 200ms │
│  watchdog (300ms deadline)│ ◀──refined_t──  │  serve loop     │
│  identity gate / swap mgr │                 │  (numpy only)    │
│  per-token trace (audit)   │  Control plane   └──────────────────┘
└────────────────────────────┘  (one-way pipes)
```

**Data plane**: `multiprocessing.shared_memory`, dual-slot rotation (SPSC). Each slot is laid out `[seq:8B][len:8B][payload]`; the sequence field is published last to enforce single-writer ordering. The module process depends **only on NumPy** — fully decoupled from the host's torch runtime, which is direct evidence of heterogeneous deployment.

**Control plane**: two one-way pipes — commands (host→module: STOP/STATS/PING) and heartbeat (module→host: HB at 200 ms). Registration completes with the module's READY message (timed, giving the cold-start metric).

**Host kernel**: `step()` is one token cycle — boundary swap → forward to tap layer → module round-trip (300 ms budget) or identity pass-through → forward to logits → decode. Every round-trip records a TokenRecord (version attribution, status, latency, hidden/refined references), forming a full audit trail.

**Fault injection**: three modes — `kill` (host-side TerminateProcess), `crash` (module executes `os._exit(1)` after serving N requests, zero cleanup), `slow` (2 s sleep per request: alive but unresponsive).

---

## 5 Minimal Validation Experiments

### 5.1 Validation Matrix (40/40 passed)

| Scenario | Target | Assertions | Result |
|---|---|---|---|
| S0 | ABI negotiation: 6 verdicts incl. 5 rejection codes | 6 | ✅ 6/6 |
| S1 | End-to-end external service | 7 | ✅ 7/7 |
| S2 | Fault isolation: 3 fault modes × 5 checks + determinism | 16 | ✅ 16/16 |
| S3 | Hot-swap consistency: A→B→(kill B)→C | 8 | ✅ 8/8 |
| S4 | Transport benchmark: 54 CPU + 9 GPU configs | 3 | ✅ 3/3 |
| Unit tests | TDD-first (red before green) | 13 | ✅ 13/13 |

**S1 key evidence**: module is a real independent process (pid isolation); 12/12 tokens served by the module; cross-process transform numerically exact (`refined == h×5.0+2.0`, fp32 lossless); after detach, output matches the no-module baseline **token-for-token** (removal reversibility).

**S2 key evidence**: under all three fault modes the host completed 12/12 tokens (zero interruption); all post-fault tokens identity pass-through; two independent runs of the kill mode produce identical outputs (degradation determinism). Detection latency: kill path 0.0 ms (immediate); crash/slow path 300.1 ms (contract deadline 300 ms).

**S3 key evidence** (16 tokens; planned swap A→B at t=5; kill B at t=10; hot-insert C at t=12):
- prefix of 5 tokens fully served by A, middle 5 by B, **zero gap at the planned swap**;
- exactly 1 token in the degraded window after B's death (FALLBACK), then all-MODULE_OK under C;
- **no mixed-version token anywhere**: each token's output matches exactly one of {A, B, C, identity};
- exact service counts: A=5 (module-side stats), B=5 (host-trace-proven; module dead, stats unavailable), C=4.

### 5.2 S4 Transport Benchmark (RTX 4060 Laptop, Windows 11, loopback echo)

**Channel comparison** (dim=4096, seq=1024, fp32, 16 MB/frame):

| Channel | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| **shm dual-slot** | **8.32 ms** | 11.14 ms | **12.25 ms** | **3882 MB/s** |
| tcp loopback | 19.41 ms | 41.32 ms | 60.52 ms | 1475 MB/s |
| pipe | 34.87 ms | 35.50 ms | 36.15 ms | 964 MB/s |

**Per-token typical load** (shm, dim=768, seq=64):
- fp16 (196 KB): **p50 = 0.014 ms, p99 = 0.027 ms**
- fp32 (384 KB): p50 = 0.023 ms, p99 = 0.027 ms; peak small-payload throughput **16,749 MB/s**

**GPU hidden-state path** (cuda → D2H → shm → CPU module → H2D, fp32):

| Config | Total p50 | Pure copy (D2H+H2D) p50 | Copy share |
|---|---|---|---|
| 768×1024 (3 MB) | 3.65 ms | 0.72 ms | 20% |
| 4096×1024 (16 MB) | 18.19 ms | 3.06 ms | 17% |

**Conclusion**: per-token cross-process round trips are in the microseconds — negligible against a real LLM's forward pass (tens of ms); the latency cost of externality appears mainly at large frames (full-sequence rewrites), where shm holds a 4× advantage over pipe. CUDA IPC is unavailable on Windows (recorded as a platform limitation); GPU paths are costed with D2H+H2D, expected to improve further under Linux + CUDA IPC (to be verified).

---

## 6 Discussion: Two New Empirical Findings

### 6.1 Module Cold-Start Bound ≈ 1.35 s (spawn → READY)

Measured cost of Windows spawn semantics + NumPy import. Implication: the lower bound of hot-plug insertion latency is seconds — **production systems need process-pool prewarming** (consistent with the motivation of serverless-LoRA cold-start optimization [16], but here measured in the independent-process + real-time data-plane setting). The READY handshake must be a mandatory part of registration — otherwise the health-check grace period races cold start and mistakes "still booting" for "dead" (we hit this exact trap during development).

### 6.2 Process-Termination Visibility Delay ≈ 125 ms → Liveness Polling Cannot Be the Detector

After a module executes `os._exit(1)`, the parent's `is_alive()` needs **125.17 ms** (two independent measurements: 125.17/126.48 ms) to observe termination (Windows ExitProcess DLL teardown and handle cleanup). Design implication:

> **Process-liveness polling is not a timely failure detector; the response-deadline watchdog is the reliable fault-detection bound for external modules.**

Ganglion's detection semantics follow: fault detection does not depend on "is it alive" but on "no delivery by the contract deadline ⇒ dead." All three fault modes' measured detection latencies fall within the contract bound (§5.1). We found no prior literature reporting this magnitude of exit-visibility measurement or its direct constraint on detector choice.

---

## 7 Limitations

1. **Tiny backbone** (3-layer MLP, dim=64, random init): deliberate (isolates the semantics variable), but latency-share conclusions must be re-measured on a real LLM;
2. **Linear transform**: validates the data plane and consistency semantics, not module expressiveness; real modules (LoRA cartridges, diffusion heads) will change the latency budget structure;
3. **Swap is token-boundary synchronous**: the race between a swap command and in-flight requests — the deeper consistency proof — is future work;
4. **Single Windows machine**: CUDA IPC, fork cold-start, and Linux scheduling not covered;
5. **SPSC single module**: multi-module fan-out and aggregate bandwidth untested.

---

## 8 Conclusion and Future Work

This paper turned "external neural modules" from a verbal concept into **a research problem with a formal definition, a runnable system, and experimental data**: all four semantics passed 40/40 assertions in a real multiprocess environment; per-token cross-process round-trip latency p50 = 0.014 ms shows the cost of externality is negligible at typical load; and two new empirical findings (cold-start bound, exit-visibility delay) directly constrain system design in this direction.

**Future work**: (a) swap/in-flight-request race consistency (CSP/TLA+ formalization); (b) re-measuring latency share on a real Qwen3-0.6B backbone; (c) multi-module fan-out and per-layer modules' aggregate bandwidth; (d) Linux + CUDA IPC platform comparison; (e) DreamHead (our diffusion-head work) as the first real module workload.

---

## References

[1] Houlsby N, et al. Parameter-Efficient Transfer Learning for NLP. ICML 2019.
[2] Hu E J, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.
[3] Shazeer N, et al. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
[4] Jiang A Q, et al. Mixtral of Experts. arXiv:2401.04088, 2024.
[5] Lenc K, Vedaldi A. Understanding Image Representations by Measuring Their Fidelity and Equivalence. ICCV 2015.
[6] Andreas J, et al. Neural Module Networks. CVPR 2016.
[7] Shen Y, et al. HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. NeurIPS 2023. arXiv:2303.17580.
[8] Borgeaud S, et al. Improving Language Models by Retrieving from Trillions of Tokens. ICML 2022. arXiv:2112.04426.
[9] Sheng Y, et al. S-LoRA: Serving Thousands of Concurrent LoRA Adapters. MLSys 2024. arXiv:2311.03285.
[10] Liu X, et al. dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving. OSDI 2024.
[11] Borzunov A, et al. Petals: Collaborative Inference and Fine-tuning of Large Models. JMLR 2023. arXiv:2209.01188.
[12] Li A, et al. EdgeMoE: Fast MoE-based LLM Inference with Edge Computing. arXiv:2301.02628.
[13] Xue F, et al. MoE-Infinity: Offloading-Efficient MoE Model Serving. arXiv:2406.00451.
[14] OD-MoE: On-Demand Expert Loading. arXiv:2512.03927, 2025.
[15] Master N, et al. Towards GPU-Enabled Serverless Function as a Service. arXiv:2303.05601.
[16] Predictive-LoRA Serverless Serving. arXiv:2512.20210, 2025.
[17] Zheng P, et al. HALO: Semantic-Aware Distributed LLM Inference in Lossy Edge Network. arXiv:2601.11676, 2026.
[18] Nie S, et al. Large Language Diffusion Models. arXiv:2502.09992, 2025.
[19] Dream Team. Dream 7B: Diffusion Reasoning Models. arXiv 2025.
[20] NVIDIA. Multi-Process Service (MPS) Documentation.

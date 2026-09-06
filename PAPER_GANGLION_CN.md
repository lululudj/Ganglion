# Ganglion：外置神经模块
## ——将神经网络能力单元作为独立进程的最小验证

**版本**：草稿 v0.2（2026-09-05，新增 §5.3 跨机真实大模型验证与 §5.4 协作模式选择准则）
**实验环境**：本地 NVIDIA RTX 4060 Laptop (8GB) + 云端 RTX 4090 (24GB)，Windows 11 / Ubuntu 22.04，Python 3.12，PyTorch 2.12/2.8，Qwen3-8B
**验证结果**：同机 40/40 + 跨机 3/3 断言通过；代码与全部实验数据见 `results.json` / `ganglion_cloud/`

---

## 摘要

主流模型能力扩展方法（Adapter、LoRA、MoE 专家）均将扩展单元置于模型进程内部的计算图中，模块故障即模型故障，模块更换需中断服务。本文提出并验证一个不同的问题设定：**外置神经模块（External Neural Module, ENM）**——将能力单元实现为拥有独立操作系统进程生命周期的服务，通过共享内存或网络在每 token 的粒度上实时处理宿主模型的 hidden states，支持运行时热插拔。我们定义了该设定的四项核心语义（ABI 注册协商、故障隔离、热换一致性、传输延迟界），实现最小验证系统 **Ganglion**：在同机多进程环境完成 5 组场景 40 项可复核断言（全部通过），并将验证升级到**真实大模型跨机部署**——云端 RTX 4090 运行 Qwen3-8B 主力骨干、本地 RTX 4060 Laptop（8GB，跑不动 bf16 8B）作为外置模块经 SSH 隧道跨公网协作：吞吐仅降 2.7%（22.52 vs 23.15 tok/s），fp32 跨网 24/24 位级精确，断连后宿主零中断完成降级。基于同一实验床的对照实验进一步给出**协作模式选择准则**：节点间每步成本比 r 与接受率 α 决定拓扑选择——实测投机流水线因成本比不满足盈利条件而 0.34–0.71x，而模块外置 0.97x 近无损。此外报告两个此前未见发表的实证发现：模块进程冷启动开销约 1.35s；Windows 下进程终止的父进程可见性延迟约 125ms——后者直接否定"存活轮询"作为故障检测机制，确立**响应截止期看门狗**为外置模块故障检测的可靠语义。

**关键词**：外置神经模块；故障隔离；热插拔；进程间通信；参数高效微调；系统可靠性

---

## 1 引言

### 1.1 现状：能力扩展单元全部住在模型进程里

为使大语言模型获得领域或任务能力，社区发展出大量模块化扩展方法：Adapter 在层间插入小模块 [1]；LoRA 以低秩增量修改权重 [2]；MoE 将专家子网络置于路由器之下 [3,4]；模块化网络与模型拼接探索了组合性 [5,6]；HuggingGPT 类工作让 LLM 编排外部模型 [7]；RETRO 引入外部记忆 [8]。这些方法的方向各异，但共享一个未言明的约束：**扩展单元与宿主模型同进程、同计算图、同故障域**。

该约束带来三个系统性后果：
1. **故障耦合**：模块缺陷（数值溢出、死循环、显存越界）直接杀死宿主；
2. **更换昂贵**：更新模块需停止服务、重载权重；
3. **部署同构**：模块与宿主必须共享运行时、框架版本乃至硬件。

### 1.2 问题：外置性从未被作为研究问题定义

系统侧社区已做出全部"零件"：S-LoRA/dLoRA 将 adapter 作为服务层数据动态复用 [9,10]，Petals 让单模型的层跨节点流动 [11]，EdgeMoE 系工作按需加载专家权重 [12–14]，GPU FaaS 提供进程级隔离基础设施 [15,16]。但**没有工作把这些零件组织为一个统一的问题定义**：如果能力单元是一个真正的独立进程——有自己的 pid、生命周期、故障域与部署单元，通过定义良好的数据面契约逐 token 处理宿主 hidden states——系统的语义应该是什么？

### 1.3 贡献

本文贡献四项：

1. **问题定义**（§3）：首次将"外置神经模块"形式化为四项可验证语义——ABI 注册协商、故障隔离、热换一致性（token 原子性）、传输延迟界；
2. **最小系统**（§4）：实现 Ganglion——三进程架构（宿主/模块/模块'），含 manifest 协商协议、双槽共享内存数据面、心跳与看门狗控制面、恒等回退门；
3. **验证方法论与结果**（§5）：7 场景（S0–S6）的 PASS/FAIL 矩阵与传输基准；跨机真实大模型（Qwen3-8B）验证四语义无衰减（§5.3）；多租户共享云端验证 batch 红利与租户级隔离同时成立（§5.5）；
4. **协作模式选择准则**（§5.4）：节点间每步成本比 $r$ 与接受率 $\alpha$ 决定拓扑选择的定量判据——外置模块 vs 投机流水的分界；
5. **两个新实证发现**（§6）：模块冷启动界（~1.35s）与进程终止可见性延迟（~125ms）及其对故障检测语义的设计约束。

本文定位为**最小验证（minimal validation）**：骨干为微型随机初始化网络，变换为可独立复算的线性算子——我们刻意消除"模型能力"变量，使**系统语义本身**成为被测对象。语义若在此最小系统成立，即为后续真实模型规模化的必要条件。

---

## 2 相关工作

| 方向 | 代表工作 | 与 ENM 的关系 |
|---|---|---|
| 进程内模块 | Adapter [1], LoRA [2], MoE [3,4] | 模块在计算图内；无进程隔离 |
| Adapter 服务化 | S-LoRA [9], dLoRA [10], LoRAX, Punica | adapter 是**数据**，进程内复用；dLoRA 支持动态换入换出但仍在同进程计算图 |
| 整模型进程隔离 | llama.cpp router mode (2025-12), Llama-Swap | 换的是**整个模型**（prompt 级，文本 I/O）；无 hidden-state 数据面、无 token 级语义；崩溃模型的在飞请求**失败**而非降级 |
| 层分布 | Petals [11], HALO [17] | hidden states 跨进程/节点流动，但拆分对象是**单模型的层**，非能力模块；无热换/故障语义 |
| 专家按需加载 | EdgeMoE [12], MoE-Infinity [13], OD-MoE [14] | 专家**权重**换入换出，但专家仍受内部路由器支配 |
| 模型编排 | HuggingGPT [7], Neural Module Networks [6] | 调用外部模型的**输出**，非实时处理 hidden states |
| 检索增强 | RETRO [8] | 外部记忆，非外部计算 |
| GPU 进程基础设施 | GPU FaaS [15], NVIDIA MPS | 提供隔离机制，无神经模块抽象 |
| Serverless LoRA | Predictive-LoRA [16] | 关注冷启动成本优化，无实时数据面协议 |
| 边云协同推理 | Splitwise [21], HybridFlow [22], PRISM [23], ECKGF [24], DSD [25] | 层拆分/子任务路由/请求级切换/投机验证——全部为**请求或层粒度**，无模块化双向 hidden-state 数据面 |
| 机器人×云端大模型 | GemmaSense [26], VLM-on-Edge [27], EdgeBot-Reflex [28] | 请求级路由到云 API（无 hidden-state 通路）/ 层卸载（无模块语义）/ 纯本地双模型仲裁（无云端） |
| hidden states 安全 | 状态反演 [29], activation steering 攻击 [30], LoRA 后门 [31,32], 拆分隐私 [33-35] | 证明 hidden-state 通道的攻击面真实存在；但**无人研究模块化外置架构的安全属性**（可拆卸后门、逐 token 归因、恒等回退） |

**空白确认**：上述任一工作均未同时满足——(a) 模块为独立进程；(b) 逐 token 处理 hidden states 的数据面；(c) 热换一致性语义；(d) 故障隔离语义——四项条件的组合。Ganglion 填补该空白。

---

## 3 问题定义：外置神经模块的四项语义

**设定**：宿主进程 $H$ 含骨干模型 $f_\theta$，在指定层 $k$ 插桩提取 hidden states $h_t \in \mathbb{R}^{n_t \times d}$（$t$ 为 token 序号）；外置模块进程 $M$ 以服务形式运行。定义以下四项语义：

**语义 1（ABI 注册协商）** 模块以清单 manifest 声明消费层、维度、dtype、超时预算与变换规范；宿主以契约 HostContract 声明插桩点。协商结果为二值：接受，或**带原因码拒绝**（E_DIM_MISMATCH / E_DTYPE_MISMATCH / E_LAYER_UNKNOWN / E_TIMEOUT_INVALID / E_FIELD_MISSING）。协商通过含就绪握手（READY）方视为注册完成。

**语义 2（故障隔离，fail-closed）** 模块进程任意故障（外部终止、自杀、无响应）时，宿主进程不中断；自检测点起，每个后续 token 的输出满足恒等回退：
$$h_t^{out} = h_t \quad (\text{refined} \equiv \text{hidden})$$
且降级路径**确定可复现**。宿主继续生成，直至新模块热插入。

**语义 3（热换一致性，token 原子性）** token $t$ 开工时绑定的模块版本**完整服务 token $t$**；换挡命令在下一个 token 边界原子生效。全程不存在"混合版本 token"——每个 token 的输出必须精确归因于唯一的变换源。

**语义 4（传输延迟界）** 数据面往返延迟 $\ell(t)$ 具有可测分布；系统应在典型负载下使 $\ell$ 相对骨干前向耗时可忽略，并报告各通道的 p50/p95/p99 与吞吐。

---

## 4 Ganglion 系统设计

```
┌────────────────────────────┐                 ┌──────────────────┐
│ HOST 进程                   │  数据面(SHM双槽) │ MODULE 进程       │
│  微型骨干(3层MLP,dim=64)     │ ──hidden_t──▶   │  manifest+READY  │
│  第1层插桩提取 h_t          │ ◀──refined_t──  │  心跳(200ms)      │
│  看门狗(响应截止期300ms)     │                 │  服务循环(numpy) │
│  恒等回退门 / swap管理器     │  控制面(单向管道) └──────────────────┘
│  逐token trace(审计)        │ ◀──HB──▶ 命令/统计
└────────────────────────────┘
```

**数据面**：`multiprocessing.shared_memory` 双槽轮换（SPSC）。每槽布局 `[seq:8B][len:8B][payload]`，序号字段最后发布实现单写者定序。宿主写入请求槽，模块轮询读取、numpy 反序列化、施加变换、写响应槽。模块进程**仅依赖 numpy**——与宿主的 torch 运行时完全解耦，构成异构部署的直接证据。

**控制面**：两条单向管道——命令通道（宿主→模块：STOP/STATS/PING）与心跳通道（模块→宿主：HB，200ms 周期）。注册协商以模块发送 READY 完成（含就绪握手计时，即冷启动指标）。

**宿主内核**：`step()` 为一个 token 周期——边界换挡 → 前向至插桩层 → 模块往返（超时预算 300ms）或恒等直通 → 续前向 → 解码。每次往返记录 TokenRecord（版本归因、状态、延迟、hidden/refined 引用），构成全程审计轨迹。

**故障注入**：三种模式——`kill`（宿主侧 `TerminateProcess`）、`crash`（模块服务 N 请求后 `os._exit(1)`，零清理）、`slow`（每请求睡眠 2s，活着但无响应）。

---

## 5 最小验证实验

### 5.1 验证矩阵（40/40 通过）

| 场景 | 验证目标 | 断言数 | 结果 |
|---|---|---|---|
| S0 | ABI 协商：6 种判定含 5 种拒绝原因码 | 6 | ✅ 6/6 |
| S1 | 端到端外置服务 | 7 | ✅ 7/7 |
| S2 | 故障隔离：3 故障模式 × 5 检查 + 确定性 | 16 | ✅ 16/16 |
| S3 | 热换一致性：A→B→(kill B)→C 全程 | 8 | ✅ 8/8 |
| S4 | 传输基准：54 CPU 配置 + 9 GPU 配置 | 3 | ✅ 3/3 |
| S5 | 跨机真实大模型：云端 Qwen3-8B + 本地外置模块（吞吐/数值/故障三查） | 3 | ✅ 3/3 |
| S6 | 协作模式成本比准则：投机流水三层对照 | — | 数据性结论 |
| 单元测试 | TDD 先行（先失败后通过） | 13 | ✅ 13/13 |

**S1 关键证据**：模块为真实独立进程（pid 隔离）；12/12 token 全部由模块服务；跨进程变换数值精确一致（`refined == h×5.0+2.0`，fp32 无损）；detach 后输出与无模块基线**逐 token 一致**（拔除可逆性）。

**S2 关键证据**：三种故障模式下宿主均完成 12/12 token 生成（零中断）；故障后全部 token 恒等直通；硬杀模式两次独立运行输出完全一致（降级路径确定性）。检测延迟：kill 即时路径 0.0ms；crash/slow 走契约超时路径 300.1ms（契约界 300ms）。

**S3 关键证据**（16 token，t=5 计划换挡 A→B，t=10 硬杀 B，t=12 热插入 C）：
- 换挡前缀 5 token 全由 A 服务，后缀 5 token 全由 B 服务，**计划内换挡零间隙**；
- B 被杀后恰 1 个 token 处于降级间隙（FALLBACK），C 接管后全部 MODULE_OK；
- **全程无混合版本 token**：每 token 输出精确归因于唯一变换源（A/B/C/恒等 之一且仅一）；
- 服务计数精确：A=5（模块侧统计）、B=5（宿主 trace 证明，模块已死不可统计）、C=4。

### 5.2 S4 传输基准（RTX 4060 Laptop, Windows 11, 回环 echo）

**通道横向对比**（dim=4096, seq=1024, fp32, 16MB/帧）：

| 通道 | p50 | p95 | p99 | 吞吐 |
|---|---|---|---|---|
| **shm 双槽** | **8.32ms** | 11.14ms | **12.25ms** | **3882 MB/s** |
| tcp 回环 | 19.41ms | 41.32ms | 60.52ms | 1475 MB/s |
| pipe | 34.87ms | 35.50ms | 36.15ms | 964 MB/s |

**逐 token 典型负载**（shm, dim=768, seq=64）：
- fp16（196KB）：**p50=0.014ms, p99=0.027ms**
- fp32（384KB）：p50=0.023ms, p99=0.027ms；小负载峰值吞吐 **16749 MB/s**

**GPU 隐状态路径**（cuda → D2H → shm → CPU 模块 → H2D，fp32）：

| 配置 | 总 p50 | 纯拷贝(D2H+H2D) p50 | 拷贝占比 |
|---|---|---|---|
| 768×1024 (3MB) | 3.65ms | 0.72ms | 20% |
| 4096×1024 (16MB) | 18.19ms | 3.06ms | 17% |

**结论**：逐 token 粒度的跨进程往返在微秒级，相对真实 LLM 的单步前向（数十毫秒）完全可忽略；外置性的延迟代价主要出现在大帧（整序列重写）场景，此时 shm 相对 pipe 有 4 倍优势。CUDA IPC 在 Windows 不可用（平台限制，如实记录），GPU 路径以 D2H+H2D 计入成本——该成本占 17-20%，在 Linux + CUDA IPC 下预期进一步降低（待验证）。

### 5.3 S5：跨机验证——本地弱设备作为云端真实大模型的外置模块

S0–S4 验证了语义，但骨干是微型网络。S5 将其升级到**真实大模型跨机部署**：云端 RTX 4090 运行 **Qwen3-8B**（36 层，hidden 4096，bf16）作为主力骨干，在第 16 层插桩；本地 RTX 4060 Laptop（8GB，跑不动 bf16 8B）作为外置模块进程（numpy 实现），数据面经 SSH 反向隧道跨公网传输，每 token 取最后位置 hidden state（4096 维 fp32，16KB/帧）。

**S5 结果（三项检查全部通过）**：

| 指标 | 数值 |
|---|---|
| A 基线（云端单独 8B） | 23.15 tok/s（p50 23.6ms/token） |
| B 外置模块（云端骨干+本地模块） | 22.52 tok/s（**吞吐仅降 2.7%**） |
| 模块跨机往返 RTT | p50=8.34ms（min 8.01 / max 22.7） |
| 跨机数值一致性 | **24/24 位级精确**（mismatch=0，fp32 跨公网零损坏） |
| D 故障注入（服务 6 token 后断连） | **24/24 生成完成**，降级点=6，零中断 |

S5 的三个核心发现：(a) **跨机外置几乎免费**——RTT（8.3ms）相对真实 LLM 前向（23.6ms）占 35% 等待，但因传输与计算部分重叠，吞吐损失仅 2.7%；模型越大前向越慢，外置成本占比越小；(b) **四语义跨机无衰减**——位级数值一致、fail-closed 降级、能力可逆改变全部在跨机环境成立；(c) 外置模块真实改变模型行为——同一 prompt 下，8B 基线续写"the brain is not a single, unified system"，挂上本地模块后续写"the brain is divided into different modules"。

### 5.4 S6：协作模式选择准则——外置模块 vs 投机流水线的成本比判据

跨机协作存在两种拓扑：**模块外置**（S5：弱设备作为能力模块插桩）与**投机流水**（本地小模型起草 + 云端大模型批量验证）。为回答"1+1 是否大于 2"，我们在同一实验床上测得三层对照数据：

| 配置 | 吞吐 | 相对云端单独 |
|---|---|---|
| 云端 8B 单独（4090） | 27.4–29.2 tok/s | 1.0x |
| 云端同机投机（0.6B 起草+8B 验证，同在 4090） | 19.5 tok/s | 0.71x |
| 跨机投机（本地 Windows 0.6B 起草+云端验证） | 9.4 tok/s | 0.34x |
| **跨机模块外置（S5）** | **22.5 tok/s** | **0.97x** |

逐层瓶颈定位给出投机解码失败的精确原因——**每步成本比**。设 draft 每步耗时 $d$、target 每步耗时 $t$、接受率 $\alpha$、每轮起草 $K$ 个 token，投机流水盈利的近似条件为：

$$r = d/t < (1-\alpha) \cdot K/(K+1)$$

实测：云端 0.6B 的 $d$=18ms、8B 的 $t$=36ms（比值 0.5，不满足）；本地 Windows 0.6B 的 $d$=38ms——**比云端 8B 每步还慢**（28 层 × 300+ kernel launches 的 WDDM 调度开销），比值 1.06，完全不满足。附带发现：裸 transformers 循环下小模型速度优势出不来（0.6B 在 4090 上仅 55 tok/s，Python 框架开销主导）；专业投机解码系统（vLLM + CUDA Graph 将 $d$ 压到 <5ms）是使该拓扑盈利的工程前提。

**协作模式选择准则**（本文的定量贡献）：节点间每步成本比 $r$ 与接受率 $\alpha$ 决定拓扑选择——$r$ 小于阈值时选投机流水，否则选**模块外置**（吞吐近无损、能力空间相乘、可用性 $1-p^2$）。弱设备作为能力模块的边际价值远超其单独算力价值——**外置架构是弱节点的放大器**。

### 5.5 S7：多租户共享云端——batch 红利与租户级故障隔离同时成立

S5/S6 回答了"一台机器人 + 一个云端"；S7 回答"**N 台机器人共享一个云端**"——这是外置架构走向机群（fleet）的关键一步。实验设置：云端单骨干 Qwen3-8B 以 **batch=N 前向**同时服务 N 个租户，每租户一条独立模块连接（独立 TCP 连接 + 独立故障域），数据面仍经 SSH 反向隧道逐 token 往返。三项结果：

| 配置 | 聚合吞吐 | 每租户 | 租户 RTT p50 |
|---|---|---|---|
| A 基线（单租户无模块） | 24.03 tok/s | — | — |
| B1 N=1 + 模块 | 26.12 tok/s | 26.12 tok/s | 10.2ms |
| **B4 N=4 共享** | **75.20 tok/s（3.1×）** | 18.80 tok/s | 9.9–21.2ms |
| **B8 N=8 共享** | **142.77 tok/s（5.5×）** | 17.85 tok/s | 9.9–23.3ms |
| C N=4 + kill 租户2 | 76.34 tok/s | — | victim 降级，健康租户不变 |

**发现一：batch 红利显著**。batch=1 前向是 memory-bound（权重读取主导），8 租户合并前向后聚合吞吐达 **5.5×**，每租户仍保有独占时 68% 的速度（17.85 vs 26.12 tok/s）——对决策驱动机型（每 640ms 一次规划、每次数十 token）每租户吞吐仍超需求一个数量级。单位算力边际成本随 N 递减：从 1→8 租户，每新增租户的边际吞吐代价从 100% 降至 ~32%。

**发现二：租户级隔离是零损伤的**。故障注入：租户 2 的模块进程（本地端信道）在服务 6 token 后被主动关闭。结果：victim 当步恒等降级并**完成全部 24 token 生成**（fallback 点=6，降级 18 步）；三个健康租户 **served 24/24、0 fallback、0 mismatch、RTT 分布不变**；隔离期间聚合吞吐 76.34 tok/s 甚至不低于无故障对照（75.20）——victim 跳过模块往返的时间略补偿了故障开销。**一个租户的死亡对同骨干其他租户零可测影响。**

**发现三：共享引入相关故障语义**。单租户模块死亡 = 独立事件（S7-C 证实）；骨干失联 = 全体同时降级（fail-closed 保证"同时变笨"而非"同时死机"）。这把多租户外置系统与单租户在可用性模型上区分开：前者故障相关性由骨干决定，后者由模块进程决定——容量规划（临界租户数 N*，由排队延迟发散点决定）与骨干冗余成为新的设计变量。

S7 代码与完整数据见 `cloud/cloud_host_s7.py`、`cloud/s7_results.json`（6/6 断言通过）。

---

## 6 讨论：两个新实证发现

### 6.1 模块冷启动界 ≈ 1.35s（spawn→READY）

Windows spawn 语义 + numpy 导入的实测开销。含义：热插拔的"插入延迟"下界为秒级，**真正的生产系统需要进程池预热**（与 serverless LoRA 冷启动优化 [16] 的动机一致，但我们给出的是独立进程+实时数据面设定下的实测值）。就绪握手（READY）应作为注册协商的强制组成部分——否则健康检查宽限期与冷启动竞态，会把"还在启动"误判为"已死亡"（我们调试中实际踩到此坑）。

### 6.2 进程终止可见性延迟 ≈ 125ms → 存活轮询不可作为检测机制

模块执行 `os._exit(1)` 后，父进程 `is_alive()` 需 **125.17ms**（两次独立测量 125.17/126.48ms）才能观察到终止（Windows ExitProcess 的 DLL 卸载与句柄清理开销）。该发现的设计含义：

> **进程存活轮询不是及时的故障探测器；响应截止期看门狗才是外置模块故障检测的可靠界。**

Ganglion 的检测语义由此确定：故障检测不依赖"看它活着吗"，而依赖"到契约截止期没交货即判死"。实测三种故障模式的检测延迟全部落在契约界内（§5.1）。我们未见此前文献报告该量级的 exit 可见性测量及其对故障检测机制选择的直接约束。

---

## 7 局限性

1. **骨干为微型网络**（3 层 MLP, dim=64, 随机初始化）：刻意为之（隔离系统语义变量），但意味着延迟占比结论需在真实 LLM 上复测；
2. **变换为线性算子**：验证的是数据面与一致性语义，非模块的表达能力；真实模块（LoRA 卡带、扩散头）的计算耗时将改变延迟预算结构；
3. **换挡为 token 边界同步驱动**：尚未处理"swap 命令与在飞请求并发到达"的竞态——这是热换一致性更深的证明，留作下一步；
4. **单机 Windows**：CUDA IPC、fork 冷启动、Linux 调度行为未覆盖；
5. **SPSC 单模块**：未测多模块扇出与聚合带宽。

---

## 8 结论与未来工作

本文将"外置神经模块"从口头概念转化为**有形式定义、可运行系统与实验数据的研究问题**：四项语义在真实多进程环境下 40/40 断言通过；逐 token 跨进程往返延迟 p50=0.014ms 证明外置性的延迟代价在典型负载下可忽略；两个新实证发现（冷启动界、exit 可见性延迟）直接约束该方向的系统设计。

**未来工作**：(a) 换挡与在飞请求的竞态一致性（CSP/TLA+ 形式化）；(b) Linux + CUDA IPC 平台对照；(c) 多模块扇出与每层一模块的聚合带宽；(d) 将 DreamHead（我们的扩散头工作）作为首个真实模块负载；(e) **具身智能应用**：机器人本体作为外置模块连接云端超大模型——反射层本地毫秒级响应、感知模块经数据面上行、fail-closed 保证断网降级不致盲，四语义与机器人分层智能需求逐条对应；(f) **安全语义**：可逆攻击面（revocable attack surface）的形式化——外置后门可拆卸（vs 权重投毒不可逆）、manifest 签名扩展、refined states 异常检测。

---

## 参考文献

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
[17] Niraula N, et al. PHATGOOSE: Routing Among Specialized Experts. arXiv:2402.05859.
[18] Nie S, et al. Large Language Diffusion Models. arXiv:2502.09992, 2025.
[19] Dream Team. Dream 7B: Diffusion Reasoning Models. arXiv 2025.
[20] NVIDIA. Multi-Process Service (MPS) Documentation.
[21] Younesi A, et al. Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL. arXiv:2512.23310, 2025.
[22] Dong J, et al. HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference. ICML 2026. arXiv:2512.22137.
[23] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference. AAAI 2026.
[24] Edge-cloud Collaborative Knowledge Graph Fusion for Real-Time Robot Q&A. Discover Internet of Things, 2026.
[25] Yu F, et al. DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving. arXiv:2511.21669, 2025.
[26] Geedh P. GemmaSense: Hybrid Edge-Cloud Vision-Language System for Robotics. GitHub, 2026.
[27] Ahmad S, et al. Vision-Language Models on the Edge for Real-Time Robotic Perception. arXiv:2601.14921, 2026.
[28] Vijaykumar S. EdgeBot-Reflex: Heterogeneous Dual-Model Robot Navigation. GitHub, 2026.
[29] Dong T, et al. Depth Gives a False Sense of Privacy: LLM Internal States Inversion. arXiv:2507.16372, 2025.
[30] Korznikov A, et al. The Rogue Scalpel: Activation Steering Compromises LLM Safety. arXiv:2509.22067, 2025.
[31] Chen L, et al. Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models. arXiv:2512.19297, 2025.
[32] Puertolas Merenciano D, et al. Weight Space Detection of Backdoors in LoRA Adapters. arXiv:2602.15195, 2026.
[33] Duan L, et al. PrivaScissors: Enhance the Privacy of Collaborative Inference through the Lens of Mutual Information. arXiv:2306.07973, 2023.
[34] Feature Sniffer: A Stealthy Inference Attacks Framework on Split Learning. Springer, 2023.
[35] Semantic Neighbor Swapping for Privacy-Aware Edge-Cloud Inference. Journal of Cloud Computing, 2026.

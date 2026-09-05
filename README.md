# Ganglion — 外置神经模块最小验证系统 / External Neural Modules — Minimal Validation

> **首次公开发布 / First public release: 2026-09-05 18:23:36 (UTC+8)** — GitLink commit `5e69c52`，GitHub 镜像同日。优先权时间戳详见英文论文 §0。

**Ganglion（神经节）**：生物学上，神经节是位于中枢之外的独立神经处理单元——损坏不致命、宿主照常运行、可移植替换。

本项目将这一语义引入大模型系统：**把神经网络能力单元作为独立进程运行**，通过共享内存在每个 token 的粒度上实时处理宿主模型的 hidden states，支持运行时热插拔与故障隔离。

## 四项核心语义

| 语义 | 一句话定义 | 验证场景 |
|---|---|---|
| ABI 注册协商 | 模块以 manifest 声明消费层/维度/dtype/超时，宿主静态校验，不兼容带原因码拒绝 | S0 |
| 故障隔离（fail-closed） | 模块进程任意崩溃，宿主不中断，恒等直通降级，确定可复现 | S2 |
| 热换一致性（token 原子性） | 换挡在 token 边界原子生效，任何 token 不见混合版本 | S3 |
| 传输延迟界 | 逐 token 跨进程 hidden-state 往返延迟分布（shm/pipe/tcp/GPU） | S4 |

## 实验结果（RTX 4060 Laptop + 云端 RTX 4090）

- **同机 40/40 断言全部通过**（S0 ABI / S1 端到端 / S2 故障隔离 / S3 热换一致性 / S4 基准）
- 逐 token 跨进程往返（768×64 fp16）：**p50 = 0.014ms**
- **S5 跨机真实大模型**：云端 Qwen3-8B 主力骨干 + 本地 4060 外置模块（SSH 隧道跨公网）——吞吐仅降 **2.7%**，fp32 跨网 **24/24 位级精确**，断连后**零中断降级**
- **S6 协作模式准则**：投机流水线 0.34–0.71x（每步成本比不满足盈利条件）vs 模块外置 **0.97x** 近无损——弱设备当能力模块的边际价值远超其单独算力（外置架构是弱节点的放大器）
- 计划内热换 token 边界零间隙；16 token 全程无混合版本输出
- 通道吞吐：shm 双槽 16749 MB/s（峰值小负载）> tcp > pipe

## 两个论文级实证发现

1. **模块冷启动 ≈ 1.35s**（Windows spawn + numpy 导入）→ 生产系统需进程池预热；READY 就绪握手必须作为注册协商的组成部分
2. **进程终止可见性延迟 ≈ 125ms**（os._exit 后父进程 is_alive() 的观察延迟）→ 存活轮询不是及时探测器，**响应截止期看门狗才是可靠检测界**

## 目录结构

```
ganglion/
  abi.py             # HostContract / ModuleManifest / negotiate() + 5 种原因码
  transport.py       # shm 双槽 / Pipe / TCP 通道 + GPU(D2H/H2D) 变体基准
  host.py            # 宿主：骨干插桩 / 看门狗 / 恒等回退门 / token 原子换挡 / 逐 token trace
  module_proc.py     # 独立模块进程：READY 握手 / 心跳 / 服务循环 + 故障注入
  scenarios.py       # S0-S4 验证场景（每项 PASS/FAIL + 证据细节）
  run_validation.py  # 一键编排 → 控制台 PASS/FAIL 矩阵
  test_ganglion.py   # 13 项单元测试（TDD 先行）
  results.json       # 全部实验数据（40 用例 + 63 项基准行）
  PAPER_GANGLION_CN.md  # 论文草稿（中文版，v0.2 含跨机验证）
  PAPER_GANGLION_EN.md  # Paper draft (English, v0.2 with §0 release/priority timestamps)
  cloud/             # S5/S6 跨机实验：云端 8B 宿主 + 本地模块/起草器 + 全部结果 JSON
```

## 快速开始

```bash
# 运行完整验证（约 2 分钟）
python ganglion/run_validation.py

# 只跑单元测试
python ganglion/test_ganglion.py
```

依赖：Python 3.12+，numpy，torch（GPU 可选——S4 的 GPU 变体自动跳过）。

## 许可与引用

实验代码与论文草稿仅供研究使用。引用格式见论文草稿 §1。

# Ganglion — 外置神经模块最小验证系统

**Ganglion（神经节）**：生物学上，神经节是位于中枢之外的独立神经处理单元——损坏不致命、宿主照常运行、可移植替换。

本项目将这一语义引入大模型系统：**把神经网络能力单元作为独立进程运行**，通过共享内存在每个 token 的粒度上实时处理宿主模型的 hidden states，支持运行时热插拔与故障隔离。

## 四项核心语义

| 语义 | 一句话定义 | 验证场景 |
|---|---|---|
| ABI 注册协商 | 模块以 manifest 声明消费层/维度/dtype/超时，宿主静态校验，不兼容带原因码拒绝 | S0 |
| 故障隔离（fail-closed） | 模块进程任意崩溃，宿主不中断，恒等直通降级，确定可复现 | S2 |
| 热换一致性（token 原子性） | 换挡在 token 边界原子生效，任何 token 不见混合版本 | S3 |
| 传输延迟界 | 逐 token 跨进程 hidden-state 往返延迟分布（shm/pipe/tcp/GPU） | S4 |

## 实验结果（RTX 4060 Laptop, Windows 11）

- **40/40 断言全部通过**（S0 ABI 6 项 / S1 端到端 7 项 / S2 故障隔离 16 项 / S3 热换一致性 8 项 / S4 基准 3 项）
- 逐 token 跨进程往返（768×64 fp16）：**p50 = 0.014ms**
- 模块硬崩溃后宿主零中断，恒等门降级，检测延迟全部落在契约超时界内
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
  PAPER_GANGLION_CN.md  # 论文草稿（中文版）
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

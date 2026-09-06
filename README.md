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
- **S7 多租户共享云端**（6/6 断言通过）：单骨干 batch=N 服务 8 租户——聚合吞吐 **142.8 tok/s（5.5×）**，每租户仍保有 17.9 tok/s；kill 单租户模块 → victim 恒等降级续生成，**健康租户 0 fallback / 0 mismatch / 24/24 服务**——共享红利与租户级故障隔离同时成立
- 计划内热换 token 边界零间隙；16 token 全程无混合版本输出
- 通道吞吐：shm 双槽 16749 MB/s（峰值小负载）> tcp > pipe

## 两个论文级实证发现

1. **模块冷启动 ≈ 1.35s**（Windows spawn + numpy 导入）→ 生产系统需进程池预热；READY 就绪握手必须作为注册协商的组成部分
2. **进程终止可见性延迟 ≈ 125ms**（os._exit 后父进程 is_alive() 的观察延迟）→ 存活轮询不是及时探测器，**响应截止期看门狗才是可靠检测界**

## 人形机器人演示 Demo（中英双语）

`demo/` 目录是纯静态 HTML 交互演示——同一张地图、同一个机器人本体，实时可视化四语义在具身场景下的行为：

- 单机器人中文版：[demo/robot_demo_cn.html](demo/robot_demo_cn.html) · English: [demo/robot_demo_en.html](demo/robot_demo_en.html)
- **多机器人共享云端（S7）**：[demo/swarm_demo_cn.html](demo/swarm_demo_cn.html) · English: [demo/swarm_demo_en.html](demo/swarm_demo_en.html)——三台机器人共享单骨干 batch 前向，可视化租户级隔离（kill R2）与相关故障（断网）的传播差异

| 操作 | 可观察的现象 |
|---|---|
| 插入模块 A「谨慎导航」 | 冷启动 READY 握手（1.35s）→ 路线绕开全部危险区 |
| 热换模块 B「激进导航」 | 决策边界原子换挡（白色菱形标记），路线质变为贴边抄近道 |
| 崩溃注入 / 断网 | fail-closed 恒等降级：**变笨不死**，运动流零中断，恢复后自动重升 |
| 决策审计轨迹 | 逐步归因：MODULE_OK / FALLBACK / PASSTHROUGH + RTT |
| （swarm）kill R2 模块 | **租户级隔离**：R2 红轨迹降级，R1/R3 照常服务（S7-C 实测零损伤） |
| （swarm）断网 | **相关故障**：全部租户同时降级——与单租户死亡的传播差异 |

## 目录结构

```
Ganglion/                # 仓库根目录（clone 后本目录即工作目录）
  abi.py             # HostContract / ModuleManifest / negotiate() + 5 种原因码
  transport.py       # shm 双槽 / Pipe / TCP 通道 + GPU(D2H/H2D) 变体基准
  host.py            # 宿主：骨干插桩 / 看门狗 / 恒等回退门 / token 原子换挡 / 逐 token trace
  module_proc.py     # 独立模块进程：READY 握手 / 心跳 / 服务循环 + 故障注入
  scenarios.py       # S0-S4 验证场景（每项 PASS/FAIL + 证据细节）
  run_validation.py  # 一键编排 → 控制台 PASS/FAIL 矩阵
  test_ganglion.py   # 13 项单元测试（TDD 先行）
  results.json       # 全部实验数据（40 用例 + 63 项基准行）
  PAPER_GANGLION_CN.md  # 论文草稿（中文版，v0.3 含 S7 多租户共享云端）
  PAPER_GANGLION_EN.md  # Paper draft (English, v0.3 with §5.5 multi-tenant shared cloud)
  cloud/             # S5/S6/S7 跨机实验：云端 8B 宿主 + 本地模块/起草器 + 全部结果 JSON
  s11/               # S11 MCU 实验：ESP32-S3 固件 + 桥接器 + 真机结果（SSH 凭证经环境变量注入）
  demo/              # 机器人交互演示（单机 + swarm 双主题 × 中英双语）
```

## 快速开始

```bash
# 运行完整验证（约 2 分钟；在仓库根目录下执行）
python run_validation.py

# 只跑单元测试
python test_ganglion.py
```

依赖：Python 3.12+，numpy，torch（GPU 可选——S4 的 GPU 变体自动跳过）。

## 许可与引用

本仓库以 [Apache License 2.0](LICENSE) 发布（含专利授权条款）。引用格式见论文草稿 §1。

## 已验证边界（诚实声明）

- **已验证的终端形态**：笔记本级（RTX 4060，Python 模块进程，跨公网 SSH 隧道，实测吞吐损耗 2.7%）；**微控制器级（S11）**：ESP32-S3（双核 240MHz、8MB PSRAM、USB-Serial/JTAG），C++ 固件实现完整模块协议端点——48/48 token 位级精确、零降级、kill 注入后精确恒等降级（fail-closed 语义在 MCU 上同样成立）
- **S11 的延迟代价**：USB-CDC 传输 + 主机侧分块限速（绕开 HWCDC RX 溢出）使模块 RTT p50 ≈ 331ms（对比 S5 numpy 模块的 8.3ms）——这是当前 USB 传输路径的工程开销，非架构极限；WiFi TCP 与 TinyUSB CDC 是已明确的优化路径（固件已支持，未实测）
- **固件层面三个实测发现**：① ESP32-S3 USB-Serial/JTAG 外设在持续大块流下丢字节（RX 环形缓冲溢出），需请求-响应整帧收发 + 主机分块限速；② Xtensa 编译器会把 `h*5+2` 收缩为单次舍入 FMA（与 numpy 两次舍入差 1 ULP，~20% 元素），固件需显式两步舍入对齐参考实现；③ `readBytes` 忙等会饿死 idle 任务触发 Task WDT 中途重启（表现为响应流混入 HELLO），等待路径必须 `delay(1)` 让出 CPU

# -*- coding: utf-8 -*-
"""Ganglion：外置神经模块最小验证系统。

语义四支柱（每个都有可验证的实验场景）：
  1. ABI 注册协商      —— manifest 声明消费层/shape/dtype/超时，宿主静态校验
  2. 故障隔离          —— 模块进程崩溃，宿主不崩、恒等降级继续生成
  3. 热换一致性        —— token 原子换挡，任何 token 不见混合版本
  4. 跨进程传输基准    —— hidden states 逐 token 往返延迟分布（shm/pipe/tcp/GPU）
"""
__version__ = "0.1.0"

# -*- coding: utf-8 -*-
"""S5 跨机实验编排器：本地模块服务(后台) + 云端8B宿主 + 结果回收。"""
import json
import subprocess
import sys
import threading
import time

import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOST = "connect.bjb1.seetacloud.com"
PORT = 12920
USER = "root"
PASSWORD = "***REMOVED-SSH-PASSWORD***"

# ---------- 1. 本地模块服务后台启动 ----------
proc = subprocess.Popen(
    [sys.executable, r"e:\神圣的卡拉链接着我们每个人\ganglion_cloud\local_module_server.py", "240"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    encoding="utf-8", errors="replace")

time.sleep(6)   # 等隧道就绪

# 后台线程持续打印本地输出
def pipe_output():
    for line in proc.stdout:
        print(f"[本地] {line.rstrip()}", flush=True)

t = threading.Thread(target=pipe_output, daemon=True)
t.start()

# ---------- 2. 云端执行宿主 ----------
print("=" * 60, flush=True)
print("[编排] 云端宿主启动（加载 Qwen3-8B 约 30-90s）...", flush=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
            timeout=20, banner_timeout=20)
_, stdout, stderr = cli.exec_command(
    "cd /root && /root/miniconda3/bin/python cloud_host_s5.py 2>&1",
    timeout=900)
channel = stdout.channel
while True:
    if channel.recv_ready():
        line = channel.recv(4096).decode("utf-8", "replace")
        print(f"[云端] {line.rstrip()}", flush=True)
    if channel.exit_status_ready() and not channel.recv_ready():
        break
    time.sleep(0.1)
code = channel.recv_exit_status()
print(f"[编排] 云端脚本退出码 {code}", flush=True)

# ---------- 3. 拉取结果 ----------
sftp = cli.open_sftp()
try:
    with sftp.file("/root/s5_results.json") as f:
        results = json.load(f)
    print("\n[编排] ===== S5 实验结果 JSON =====")
    print(json.dumps(results, ensure_ascii=False, indent=2)[:3000])
except FileNotFoundError:
    print("[编排] 未找到结果文件（宿主可能失败）")
sftp.close()
cli.close()

# ---------- 4. 收尾本地模块服务 ----------
time.sleep(2)
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
print("\n[编排] 全部完成", flush=True)

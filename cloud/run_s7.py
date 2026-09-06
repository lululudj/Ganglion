# -*- coding: utf-8 -*-
"""S7 编排器（本地运行）：一键执行跨机多租户实验。

流程：
  1. SFTP 上传 cloud_host_s7.py 到云机 /root/
  2. 子进程启动 local_module_server_s7.py（数据隧道 19002 + 控制隧道 19003）
  3. SSH 阻塞执行云机 python cloud_host_s7.py（模型加载约 1-2 分钟）
  4. 拉回 /root/s7_results.json 到本地 cloud/ 目录
"""
import os
import subprocess
import sys
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = "connect.bjb1.seetacloud.com"
PORT = 12920
USER = "root"
PASSWORD = os.environ["GANGLION_SSH_PASS"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                timeout=20, banner_timeout=20)

    # 1. 上传云端脚本
    sftp = cli.open_sftp()
    local_path = os.path.join(HERE, "cloud_host_s7.py")
    sftp.put(local_path, "/root/cloud_host_s7.py")
    print(f"[编排] 已上传 {local_path} → /root/cloud_host_s7.py", flush=True)

    # 2. 启动本地模块服务（后台子进程）
    local_server = os.path.join(HERE, "local_module_server_s7.py")
    proc = subprocess.Popen([sys.executable, local_server, "600"],
                            cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    print("[编排] 本地模块服务已启动（600s）", flush=True)
    time.sleep(6)   # 等隧道就绪

    # 3. 运行云端实验（阻塞，流式输出）
    print("[编排] 启动云端宿主实验 ...", flush=True)
    stdin, stdout, stderr = cli.exec_command(
        "cd /root && /root/miniconda3/bin/python cloud_host_s7.py 2>&1",
        timeout=900)
    for line in iter(stdout.readline, ""):
        if not line:
            break
        print("  " + line.rstrip(), flush=True)
    rc = stdout.channel.recv_exit_status()
    print(f"[编排] 云端实验退出码 {rc}", flush=True)

    # 4. 拉回结果
    try:
        sftp.get("/root/s7_results.json", os.path.join(HERE, "s7_results.json"))
        print("[编排] 结果已拉回 cloud/s7_results.json", flush=True)
    except FileNotFoundError:
        print("[编排] ⚠ 云端结果文件不存在（实验可能失败）", flush=True)

    # 5. 停止本地服务
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=5)
        print("[本地服务输出末尾]")
        for line in (out or "").splitlines()[-25:]:
            print("  " + line, flush=True)
    except subprocess.TimeoutExpired:
        proc.kill()

    sftp.close()
    cli.close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""停验证器 → 上传并运行 s6b 同机投机解码。"""
import sys
import time

import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect("connect.bjb1.seetacloud.com", port=12920, username="root",
            password="***REMOVED-SSH-PASSWORD***", timeout=20)

# 正则技巧停掉验证器（避免 pkill 自杀）
_, o, _ = cli.exec_command("pkill -f 's6_veri[f]ier'; sleep 2; "
                            "nvidia-smi --query-gpu=memory.used "
                            "--format=csv,noheader")
print("[停验证器] 显存:", o.read().decode().strip())

sftp = cli.open_sftp()
sftp.put(r"e:\神圣的卡拉链接着我们每个人\ganglion_cloud\s6b_cloud_spec.py",
         "/root/s6b_cloud_spec.py")
sftp.close()
print("[上传] s6b_cloud_spec.py 完成")

_, stdout, _ = cli.exec_command(
    "/root/miniconda3/bin/python /root/s6b_cloud_spec.py 2>&1", timeout=600)
chan = stdout.channel
while True:
    if chan.recv_ready():
        print(chan.recv(4096).decode("utf-8", "replace").rstrip(), flush=True)
    if chan.exit_status_ready() and not chan.recv_ready():
        break
    time.sleep(0.1)
print(f"[退出码] {chan.recv_exit_status()}")

# 拉结果
_, o, _ = cli.exec_command("cat /root/s6b_results.json 2>/dev/null")
res = o.read().decode("utf-8", "replace").strip()
if res:
    with open(r"e:\神圣的卡拉链接着我们每个人\ganglion_cloud\s6b_results.json",
              "w", encoding="utf-8") as f:
        f.write(res)
    print("[结果已拉回本地]")
cli.close()

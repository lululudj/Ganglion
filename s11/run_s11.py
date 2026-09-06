# -*- coding: utf-8 -*-
"""S11 一键编排器（本地运行）。

用法：
  python run_s11.py --mode mock     # 协议链路验证（mock TCP 模拟 MCU）
  python run_s11.py --mode esp32    # 真实硬件（USB 串行 COM3）

流程：
  1. SFTP 上传 cloud_host_s11.py 到云机
  2. 启动模块侧（mock 进程 或 ESP32 直连）+ 桥接器（正常）
  3. 云端跑 AB 阶段（基线 + 模块全程服务 48 token）→ 拉回 JSON
  4. 重启桥接器（--kill-after 6 故障注入）
  5. 云端跑 C 阶段（24 token，第 6 帧后模块死亡）→ 拉回 JSON
  6. 合并 → s11/s11_results.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
SSH_HOST = "connect.bjb1.seetacloud.com"
SSH_PORT = 12920
SSH_USER = "root"
SSH_PASS = os.environ["GANGLION_SSH_PASS"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_cloud_phase(cli, sftp, phase):
    print(f"\n[编排] ===== 云端 {phase} 阶段 =====", flush=True)
    stdin, stdout, stderr = cli.exec_command(
        f"cd /root && /root/miniconda3/bin/python cloud_host_s11.py "
        f"--phase {phase} 2>&1", timeout=1200)
    for line in iter(stdout.readline, ""):
        if not line:
            break
        print("  " + line.rstrip(), flush=True)
    rc = stdout.channel.recv_exit_status()
    local = os.path.join(HERE, f"s11_{phase}.json")
    sftp.get(f"/root/s11_{phase}.json", local)
    print(f"[编排] {phase} 结果已拉回 {local}（退出码 {rc}）", flush=True)
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "esp32"], default="esp32")
    ap.add_argument("--com", default="COM3")
    args = ap.parse_args()

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS,
                timeout=20, banner_timeout=20)
    sftp = cli.open_sftp()
    sftp.put(os.path.join(HERE, "cloud_host_s11.py"), "/root/cloud_host_s11.py")
    print("[编排] 已上传 cloud_host_s11.py", flush=True)

    # 模块侧准备
    mock_proc = None
    if args.mode == "mock":
        mock_proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "mock_esp32.py"), "--port", "3333"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace")
        time.sleep(2)
        bridge_args = ["--mode", "tcp", "--host", "127.0.0.1", "--port", "3333"]
        print("[编排] mock 模块已启动（TCP 3333）", flush=True)
    else:
        bridge_args = ["--mode", "serial", "--port", args.com]
        print(f"[编排] ESP32 直连模式（串行 {args.com}）", flush=True)

    def start_bridge(extra=None):
        cmd = [sys.executable, os.path.join(HERE, "bridge.py")] + bridge_args + \
              ["--duration", "400"]
        if extra:
            cmd += extra
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")

    def drain_bridge(p, label):
        try:
            p.terminate()
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
        print(f"[编排] {label} 桥接器退出：")
        for line in (out or "").splitlines()[-20:]:
            print("  " + line, flush=True)

    # ---- AB 阶段 ----
    bridge = start_bridge()
    time.sleep(5)                       # 隧道 + HELLO 就绪
    ab = run_cloud_phase(cli, sftp, "AB")
    drain_bridge(bridge, "AB")
    time.sleep(2)

    # ---- C 阶段（故障注入）----
    bridge = start_bridge(["--kill-after", "6"])
    time.sleep(5)
    c = run_cloud_phase(cli, sftp, "C")
    drain_bridge(bridge, "C")

    if mock_proc is not None:
        mock_proc.terminate()
        try:
            mock_proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            mock_proc.kill()

    # ---- 合并 ----
    results = {
        "mode": f"{args.mode}" + ("" if args.mode == "mock" else "-usb-serial"),
        "chip": ("mock (Python behavioral twin)" if args.mode == "mock"
                 else "ESP32-S3 (QFN56 rev v0.2, 240MHz dual-core, 8MB PSRAM, "
                      "USB-Serial/JTAG)"),
        "module_transport": ("bridge TCP → mock" if args.mode == "mock" else
                             "cloud SSH tunnel → bridge → USB-CDC serial → ESP32-S3, "
                             "streaming elementwise fp32 transform, 2048B chunks"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "AB": ab,
        "C": c,
        "checks": {
            "B_all_served": ab.get("checks", {}).get("all_tokens_served", False),
            "B_bit_exact": ab.get("checks", {}).get("bit_exact", False),
            "B_zero_fallbacks": ab.get("checks", {}).get("zero_fallbacks", False),
            "C_degraded_not_died": c.get("checks", {}).get("degraded_not_died", False),
        },
    }
    out_path = os.path.join(HERE, "s11_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[编排] 合并结果 → {out_path}", flush=True)
    print("[S11 总断言]", flush=True)
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)

    sftp.close()
    cli.close()


if __name__ == "__main__":
    main()

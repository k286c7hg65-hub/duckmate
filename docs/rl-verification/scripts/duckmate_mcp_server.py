#!/usr/bin/env python3
"""DuckMate Control Hub — P0 训练域 MCP Server (T1-T5)
运行: python3 duckmate_mcp_server.py (stdio transport)
客户端: 任意 MCP 客户端 (Claude Desktop / 群 bot / Codex / mcp cli)

工具:
  T1 train.smoke         — brax env 冒烟 (构造 + 单步)
  T2 gate.eval           — M1 门槛评估 (门槛 1-4)
  T3 train.export_cfg    — 导出实际安装版超参 config
  T4 spec.patch_preview  — Spec patch 5 处预览 (只读)
  T5 train.status        — AutoDL GPU/任务/日志状态

安全边界: 全部只读/预览/冒烟; 无训练启动/无代码写操作。
"""
import os, sys, json, time
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ── AutoDL 连接配置 (克隆实例) ──────────────────────────────
# 优先级: 环境变量 > ~/.config/duckmate-mcp.env (本地凭据文件, 600, 禁入 GitHub)
def _load_env_file():
    path = os.path.expanduser("~/.config/duckmate-mcp.env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                # 空 env 值视为未设置, 用文件值补齐
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()

_load_env_file()

AUTODL_HOST = os.environ.get("AUTODL_HOST", "connect.bjb2.seetacloud.com")
AUTODL_PORT = int(os.environ.get("AUTODL_PORT", "47617"))
AUTODL_USER = os.environ.get("AUTODL_USER", "root")
AUTODL_PASS = os.environ.get("AUTODL_PASS", "")

# 训练栈位置 (克隆实例 /root 系统盘 — 2026-09-06 适配 odm-playground)
RL_FORK = "/root/odm-playground"
VENV_PY = "/root/odm-playground/.venv/bin/python"

mcp = FastMCP("duckmate-control-hub")


def ssh_run(command: str, timeout: int = 120) -> dict:
    """SSH 到 AutoDL 执行命令, 返回结构化结果."""
    import paramiko
    if not AUTODL_PASS:
        return {"ok": False, "error": "AUTODL_PASS 未配置 (env)", "stdout": "", "stderr": ""}
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(AUTODL_HOST, port=AUTODL_PORT, username=AUTODL_USER,
                       password=AUTODL_PASS, timeout=20, banner_timeout=20)
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode(errors="ignore")
        err = stderr.read().decode(errors="ignore")
        rc = stdout.channel.recv_exit_status()
        return {"ok": rc == 0, "rc": rc, "stdout": out[-4000:], "stderr": err[-2000:]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "stdout": "", "stderr": ""}
    finally:
        client.close()


# ── T1: brax 冒烟 ──────────────────────────────────────────
@mcp.tool()
def train_smoke(env: str = "joystick", steps: int = 5) -> dict:
    """T1 brax 冒烟测试: 验证训练 env 可构造并完成单步。

    Args:
        env: 环境名 (joystick/standing 等)
        steps: 步数 (默认 5)
    """
    # odm-playground 自定义 env (joystick/standing) — 非 brax 内置 envs
    cmd = (
        f"cd {RL_FORK} && export WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false && "
        f"timeout 240 {VENV_PY} -c \"\n"
        f"from playground.open_duck_mini_v2 import {env}\n"
        f"e = {env}.Joystick(task='flat_terrain') if '{env}'=='joystick' else {env}.Standing(task='flat_terrain')\n"
        f"s = e.reset(__import__('jax').random.PRNGKey(0))\n"
        f"act = __import__('jax').numpy.zeros((1, e.action_size))\n"
        f"s = e.step(s, act)\n"
        f"print('SMOKE_OK env={env} action_size=' + str(e.action_size))\" 2>&1 | tail -8"
    )
    res = ssh_run(cmd, timeout=180)
    ok_marker = "SMOKE_OK" in res.get("stdout", "")
    return {
        "env": env, "smoke_pass": ok_marker,
        "stdout_tail": res.get("stdout", "")[-800:],
        "stderr_tail": res.get("stderr", "")[-500:],
        "ssh_ok": res.get("ok", False),
    }


# ── T2: M1 门槛评估 ────────────────────────────────────────
@mcp.tool()
def gate_eval(gate: str = "1", patch_status: str = "applied") -> dict:
    """T2 M1 门槛评估 (门槛 1-4, 对应 Spec #2)。

    Args:
        gate: 门槛号 1|2|3|4 (或 'all')
        patch_status: applied|pending — patch 是否已应用
    """
    # M1 gate 评估脚本: /root/m1_gate_eval.py (stand ONNX 本地验收, 纯本地无 GPU 亦可跑)
    # 注: joystick 门槛 2 evaluator (joystick_eval.py) 尚未实现 — 实现后在此挂接
    cmd = (
        f"ls /root/m1_gate_eval.py 2>/dev/null && python3 /root/m1_gate_eval.py --gate {gate} 2>&1 | tail -30 "
        f"|| echo 'M1_GATE_SCRIPT_NOT_FOUND — joystick_eval 待实现'"
    )
    res = ssh_run(cmd, timeout=300)
    return {
        "gate": gate, "patch_status": patch_status,
        "output": res.get("stdout", "")[-2500:],
        "stderr": res.get("stderr", "")[-800:],
        "ssh_ok": res.get("ok", False),
    }


# ── T3: 导出超参 config ────────────────────────────────────
@mcp.tool()
def train_export_cfg(env: str = "joystick") -> dict:
    """T3 从实际安装版导出训练超参 config (结构化 JSON)。

    Args:
        env: 环境名 joystick|standing
    """
    # 导出实际安装版超参: preset 从 mujoco_playground locomotion_params + env default_config
    cmd = (
        f"cd {RL_FORK} && export WANDB_MODE=disabled && "
        f"timeout 120 {VENV_PY} -c \"\n"
        f"import json\n"
        f"from mujoco_playground.config import locomotion_params\n"
        f"ppo = dict(locomotion_params.brax_ppo_config('BerkeleyHumanoidJoystickFlatTerrain'))\n"
        f"ppo = {{k: (list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else str(v)) for k, v in ppo.items()}}\n"
        f"print('PRESET_JSON ' + json.dumps(ppo))\n"
        f"\" 2>&1 | grep -E 'PRESET_JSON|Error' | tail -3"
    )
    res = ssh_run(cmd, timeout=120)
    out = res.get("stdout", "")
    # 解析 PRESET_JSON 前缀行
    cfg_json = None
    for line in out.split("\n"):
        line = line.strip()
        if line.startswith("PRESET_JSON "):
            try:
                cfg_json = json.loads(line[len("PRESET_JSON "):])
                break
            except Exception:
                continue
    return {
        "env": env, "config": cfg_json,
        "raw_tail": out[-1500:],
        "stderr": res.get("stderr", "")[-500:],
        "ssh_ok": res.get("ok", False),
    }


# ── T4: Spec patch 预览 (只读) ─────────────────────────────
SPEC_PATCHES = [
    {"id": 1, "item": "lin_vel_x", "default": "[-0.15, 0.15]", "spec": "[0.05, 0.15] (90% 采样域)"},
    {"id": 2, "item": "lin_vel_y", "default": "[-0.2, 0.2]", "spec": "[0.0, 0.0]"},
    {"id": 3, "item": "ang_vel_yaw", "default": "[-1.0, 1.0]", "spec": "[0.0, 0.0]"},
    {"id": 4, "item": "head_range_factor", "default": "1.0", "spec": "0.0 (头颈命令恒 0)"},
    {"id": 5, "item": "push_config.enable", "default": "True", "spec": "False (首轮)"},
]

@mcp.tool()
def spec_patch_preview(env: str = "joystick") -> dict:
    """T4 Spec #2 patch 5 处清单预览 (只读, 不应用)。

    Args:
        env: 目标环境
    """
    return {
        "env": env, "patches": SPEC_PATCHES,
        "note": "只读预览 — 应用需人工审核后另行操作",
        "source": "joystick-gate-analysis-20260906.md §2",
    }


# ── T5: 训练状态 ───────────────────────────────────────────
@mcp.tool()
def train_status() -> dict:
    """T5 AutoDL GPU / 训练进程 / 磁盘状态."""
    cmd = (
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null; "
        "echo '---PROC---'; "
        "ps aux | grep -E 'python.*(train|brax|joystick)' | grep -v grep | head -5; "
        "echo '---DISK---'; df -h /root /root/autodl-tmp 2>/dev/null | tail -2"
    )
    res = ssh_run(cmd, timeout=60)
    return {
        "gpu": res.get("stdout", "").split("---PROC---")[0].strip(),
        "processes": (res.get("stdout", "").split("---PROC---")[1].split("---DISK---")[0]
                      if "---PROC---" in res.get("stdout", "") else "").strip(),
        "disk": (res.get("stdout", "").split("---DISK---")[1]
                 if "---DISK---" in res.get("stdout", "") else "").strip(),
        "ssh_ok": res.get("ok", False),
        "error": res.get("error", ""),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

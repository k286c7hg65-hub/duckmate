#!/usr/bin/env python3
"""m1_gate_eval.py — M1 gate headless 验收：ONNX 策略在 mujoco 模拟中站 20s

对齐 standing 训练 obs 布局（85 维），无头服务器（AutoDL）可用，不需要图形界面。

用法：
    uv run python tools/m1_gate_eval.py <policy.onnx> [duration_s]

通过标准：duration_s 内零跌倒（body z > 0.05m 全程）。
2026-09-03 实测：standing 全量 150M ONNX → 20s PASS（body z mean 0.1515 / min 0.1489）。

依赖复用 playground.open_duck_mini_v2.mujoco_infer.MjInfer（其观测辅助方法），
但 obs 拼接精确对齐训练 standing._get_obs（85 维），不走官方 get_obs（101 维 joystick 布局）。
"""
import sys
import time

import mujoco
import numpy as np

# 轻量兼容：mujoco_infer 链中 common/utils.py 的 LowPassActionFilter 只用 jp.array+标量运算。
# 在无 jax 环境（如本地纯 mujoco 重验）自动注入 numpy stub；真实 venv 中 jax 存在则不受影响。
try:
    import jax  # noqa: F401
except ImportError:
    import types as _types
    import pathlib as _pl

    # --- jax stub：LowPassActionFilter 只用 jp.array + 标量运算 ---
    _jax = _types.ModuleType("jax")
    _jax.numpy = np
    _jax.Array = np.ndarray
    sys.modules["jax"] = _jax
    sys.modules["jax.numpy"] = np

    # --- mujoco.mjx stub（base.py: from mujoco import mjx）---
    import mujoco as _mujoco

    if not hasattr(_mujoco, "mjx"):
        _mjx = _types.ModuleType("mujoco.mjx")
        _mjx.Data = object
        _mjx.Model = object
        _mjx.put_model = lambda *a, **k: None
        _mujoco.mjx = _mjx

    # --- mujoco_playground._src.mjx_env stub（base.py: update_assets / MjxEnv）---
    def _update_assets(assets, path, pattern="*"):
        p = _pl.Path(path)
        if not p.exists():
            return
        if pattern == "*":
            files = [f for f in p.rglob("*") if f.is_file()]
        else:
            files = [f for f in p.glob(pattern) if f.is_file()]
        for f in files:
            assets[f.name] = f.read_bytes()

    class _MjxEnv:  # noqa: D401
        pass

    _pkg_mjxenv = _types.ModuleType("mujoco_playground._src.mjx_env")
    _pkg_mjxenv.update_assets = _update_assets
    _pkg_mjxenv.MjxEnv = _MjxEnv
    _pkg_src = _types.ModuleType("mujoco_playground._src")
    _pkg_src.mjx_env = _pkg_mjxenv
    _pkg = _types.ModuleType("mujoco_playground")
    _pkg._src = _pkg_src
    sys.modules["mujoco_playground"] = _pkg
    sys.modules["mujoco_playground._src"] = _pkg_src
    sys.modules["mujoco_playground._src.mjx_env"] = _pkg_mjxenv

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

ONNX = sys.argv[1] if len(sys.argv) > 1 else None
DURATION_S = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
if ONNX is None:
    print("用法: uv run python tools/m1_gate_eval.py <policy.onnx> [duration_s]")
    sys.exit(2)

mjinfer = MjInfer(
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    ONNX,
    standing=True,
)


def build_obs():
    """精确对齐训练 standing._get_obs: gyro3+acc3+cmd7+joint14x5+contact2 = 85（无噪声/ref/phase/targets）

    注（2026-09-05 Codex 审计修正）：
    - 无 acc[0]+=1.3 偏置——该偏置是 mujoco_infer.py get_obs（joystick 布局）的行为，
      standing.py 训练侧 _get_obs 直接使用传感器加速度（无 bias）
    """
    gyro = mjinfer.get_gyro(mjinfer.data)
    acc = mjinfer.get_accelerometer(mjinfer.data)  # 无 bias，对齐训练
    cmd = np.array(mjinfer.commands, dtype=np.float32)  # 7: 3 vel + 4 head
    joint_angles = mjinfer.get_actuator_joints_qpos(mjinfer.data.qpos)
    joint_vel = mjinfer.get_actuator_joints_qvel(mjinfer.data.qvel)
    contacts = mjinfer.get_feet_contacts(mjinfer.data)
    obs = np.concatenate(
        [
            gyro,
            acc,
            cmd,
            joint_angles - mjinfer.default_actuator,
            joint_vel * mjinfer.dof_vel_scale,
            mjinfer.last_action,
            mjinfer.last_last_action,
            mjinfer.last_last_last_action,
            contacts,
        ]
    ).astype(np.float32)
    return obs


fb_addr = mjinfer._floating_base_qpos_addr
print(f"initial body z: {mjinfer.data.qpos[fb_addr + 2]:.4f} m | nu={mjinfer.model.nu}")

dec = mjinfer.decimation
n_policy_steps = int(DURATION_S * 50)  # 50Hz policy
body_z = []
fall_t = None
t0 = time.time()
for i in range(n_policy_steps):
    for _ in range(dec):
        mujoco.mj_step(mjinfer.model, mjinfer.data)
    obs = build_obs()
    assert obs.shape[0] == 85, f"obs {obs.shape[0]} != 85"
    action = mjinfer.policy.infer(obs)
    mjinfer.last_last_last_action = mjinfer.last_last_action.copy()
    mjinfer.last_last_action = mjinfer.last_action.copy()
    mjinfer.last_action = action.copy()
    mjinfer.motor_targets = mjinfer.default_actuator + action * mjinfer.action_scale
    # 限速对齐 mujoco_infer 原版: max_motor_velocity * (sim_dt * decimation) = 5.24*0.02 = 0.1048 rad/周期
    # （2026-09-05 Codex 审计修正：原版漏乘 sim_dt*decimation，宽松 50 倍）
    _vel_clip = mjinfer.max_motor_velocity * (mjinfer.sim_dt * mjinfer.decimation)
    mjinfer.motor_targets = np.clip(
        mjinfer.motor_targets,
        mjinfer.prev_motor_targets - _vel_clip,
        mjinfer.prev_motor_targets + _vel_clip,
    )
    mjinfer.prev_motor_targets = mjinfer.motor_targets.copy()
    mjinfer.data.ctrl = mjinfer.motor_targets.copy()
    z = mjinfer.data.qpos[fb_addr + 2]
    body_z.append(z)
    if fall_t is None and z < 0.05:
        fall_t = i / 50.0
    if i % 250 == 0:
        print(f"  t={i / 50:.1f}s z={z:.4f}", flush=True)

body_z = np.array(body_z)
print("=" * 50)
print(f"Rollout {DURATION_S}s done in {time.time() - t0:.1f}s real time")
print(f"body z: mean={body_z.mean():.4f} min={body_z.min():.4f} final={body_z[-1]:.4f}")
if fall_t is not None:
    print(f"FALL at t={fall_t:.2f}s -> FAIL")
    sys.exit(1)
else:
    print(f"No fall in {DURATION_S}s -> PASS (body z min {body_z.min():.3f}m)")
    sys.exit(0)

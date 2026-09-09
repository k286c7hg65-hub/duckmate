#!/usr/bin/env python3
"""apply_brax_fix.py — brax 0.14.2 × jax >=0.11 单卡兼容修复（幂等，可重复运行）

背景：jax 0.11.1 移除了 jax.device_put_replicated（pmap 新语义），brax 0.14.2 仍有 4 处调用。
单 GPU 等价替代 = 给 pytree 每个 leaf 加 leading device 维 ([1, ...]) 再 device_put。

修复点：
1. brax/training/pmap.py            bcast_local_devices
2. brax/training/agents/ppo/train.py  training_state 复制 + _unpmap
3. brax/training/agents/sac/train.py  device_put_replicated 调用
4. brax/training/agents/apg/train.py  training_state 复制

用法：
    uv run python tools/apply_brax_fix.py
    # 或直接 python tools/apply_brax_fix.py（需在 venv 内）
"""
import pathlib
import shutil
import sys

try:
    import jax.numpy as jnp  # noqa: F401  确保在 venv 里跑（jax 存在）
except ImportError:
    print("⚠️ 请用 venv 运行：uv run python tools/apply_brax_fix.py")
    sys.exit(1)

BRAX_ROOT = None
import brax  # noqa: E402
try:
    import brax.training as _bt
    BRAX_ROOT = pathlib.Path(_bt.__file__).parent
    print(f"brax training 定位: {BRAX_ROOT}")
except Exception as e:
    print(f"⚠️ 无法定位 brax.training: {e}")
    sys.exit(1)
if not BRAX_ROOT.exists():
    print(f"❌ brax training 目录不存在: {BRAX_ROOT}")
    sys.exit(1)

# ---- 备份 ----
bak = BRAX_ROOT.parent / "training_bak_pre_fix"
if not bak.exists():
    shutil.copytree(BRAX_ROOT, bak)
    print(f"📦 备份: {bak}")

# ---- 1. pmap.py bcast_local_devices ----
old_pmap = """def bcast_local_devices(value, local_devices_to_use=1):
  \"\"\"Broadcasts an object to all local devices.\"\"\"
  devices = jax.local_devices()[:local_devices_to_use]
  return jax.device_put_replicated(value, devices)"""
new_pmap = """def bcast_local_devices(value, local_devices_to_use=1):
  \"\"\"Broadcasts an object to all local devices.\"\"\"
  devices = jax.local_devices()[:local_devices_to_use]
  if len(devices) == 1:
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.expand_dims(jnp.asarray(x), 0), devices[0]), value)
  return jax.device_put_replicated(value, devices)"""

# ---- 2a. ppo train.py training_state ----
old_repl = """  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )"""
new_repl = """  _local_devs = jax.local_devices()[:local_devices_to_use]
  if len(_local_devs) == 1:
    training_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.expand_dims(jnp.asarray(x), 0), _local_devs[0]), training_state)
  else:
    training_state = jax.device_put_replicated(training_state, _local_devs)"""

# ---- 2b. ppo train.py _unpmap ----
old_unpmap = """def _unpmap(v):
  # Avoid degraded performance under the new jax.pmap.
  return jax.tree_util.tree_map(
      lambda x: x.addressable_shards[0].data.squeeze(0), v
  )"""
new_unpmap = """def _unpmap(v):
  # Avoid degraded performance under the new jax.pmap.
  def _f(x):
    if hasattr(x, \"addressable_shards\") and len(x.addressable_shards) > 0:
      d = x.addressable_shards[0].data
      if d.ndim > 0 and d.shape[0] == 1:
        return d.squeeze(0)
      return d
    return x
  return jax.tree_util.tree_map(_f, v)"""

# ---- 3. sac / apg 单行调用 ----
old_single = "  return jax.device_put_replicated("
new_single = "  return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.expand_dims(jnp.asarray(x), 0)), "

jobs = [
    ("pmap.py", old_pmap, new_pmap),
    ("agents/ppo/train.py", old_repl, new_repl),
    ("agents/ppo/train.py", old_unpmap, new_unpmap),
    ("agents/sac/train.py", old_single, new_single),
    ("agents/apg/train.py", old_repl, new_repl),
]

changed = 0
for rel, old, new in jobs:
    f = BRAX_ROOT / rel
    t = f.read_text()
    if old in t:
        t = t.replace(old, new)
        f.write_text(t)
        print(f"✅ patched {rel}")
        changed += 1
    else:
        # 已修复或版本不同
        print(f"⏭️  skip {rel} (pattern not found — maybe already fixed or different brax version)")

print(f"\n完成：{changed} 处修复。")
print("验证：uv run python -c \"from mujoco_playground._src.collision import geoms_colliding; print('collision OK')\"")

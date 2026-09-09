# SETUP.md — DuckMate 训练环境搭建手册（AutoDL 4090 实测 2026-09-03）

> 全部步骤在 AutoDL RTX 4090 24GB + PyTorch 2.5.1 镜像（Python 3.12 + CUDA 12.4）实测通过。
> 总耗时 ~40min（主要 uv sync 下载 8.2G venv），踩坑全部记录。

## 0. 前置

- AutoDL 等 GPU 实例（30G 系统盘 + 数据盘）。**系统盘 30G 是硬约束**（见 §6 磁盘坑）。
- 镜像选 PyTorch 2.5.1 + Python 3.12 + CUDA 12.4（Python 在 `/root/miniconda3/bin/python3`，不在默认 PATH）。

## 1. Clone（国内网络必须镜像）

**clone 代码库**（DuckMate 维护版，含 tools/apply_brax_fix.py 等增量修复/验收脚本；本仓 patches/ 目录同步存放）：

```bash
# DuckMate 维护版（含增量修复/验收脚本）：
git clone https://ghproxy.net/https://github.com/apirrone/Open_Duck_Playground.git ~/odm-playground
```

## 2. 安装 uv + 配置国内镜像

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
# 阿里云镜像实测 3.7MB/s（清华 tuna 只有 2MB/s）
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
```

## 3. uv sync（venv 8.2G / 246 包）

```bash
cd ~/odm-playground
uv sync 2>&1 | tail -5
```

⚠️ **卡死处理**：uv sync 可能卡 futex_wait（并发死锁）。症状 = 长时间 0 进度。
解法：`pkill -f "uv sync"` 后加并发限制重跑：

```bash
UV_CONCURRENT_DOWNLOADS=8 UV_CONCURRENT_BUILDS=4 uv sync 2>&1 | tail -5
```

实测最终能跑通（阿里云镜像下 ~15-20min）。venv 落在 `.venv/`（8.2G）。

## 4. mujoco_playground 版本锁定（关键！）

**必须 0.0.5**。PyPI 最新 0.2.0 移除了 `_src/collision.py`（2025-08-15 提交改为 contact sensor），
ODM 代码 `standing.py:27` / `joystick.py:27` 的
`from mujoco_playground._src.collision import geoms_colliding` 会 `ModuleNotFoundError`。

```bash
uv pip install "playground==0.0.5"
# 验证：
.venv/bin/python -c "from mujoco_playground._src.collision import geoms_colliding; print('OK')"
```

（uv sync 默认装 0.2.0，因为 pyproject 写的是 `playground>=0.0.3` 无上限。）

## 5. brax × jax 单卡兼容修复（关键！）

brax 0.14.2 用了 `jax.device_put_replicated`，jax 0.11.1 已移除（AttributeError）。
单 GPU 下等价替代 = 给每个 leaf 加 leading device 维 `[1, ...]`（jax 0.11 新 pmap 语义）。

```bash
uv run python tools/apply_brax_fix.py
# 修复 4 处：pmap.py bcast_local_devices / ppo train.py / sac train.py / apg train.py
# 外加 ppo train.py _unpmap 兼容非 pmap 输出
```

修复后验证（~2min 首次 jax 编译）：

```bash
cd ~/odm-playground && WANDB_MODE=disabled .venv/bin/python playground/open_duck_mini_v2/runner.py \
  --env standing --task flat_terrain --num_timesteps 1000000 --output_dir /tmp/smoke_test
# 看到 "STEP: ... reward: ..." 即通
```

## 6. 磁盘管理（30G 系统盘是硬约束）

**坑**：uv 缓存 `/root/.cache/uv` 会涨到 21G！加上 venv 8.2G = 30G 爆满 →
JAX 编译 PTX 写不进 `/tmp` → `RESOURCE_EXHAUSTED: No space left on device`。

```bash
# venv 装好后立刻清 uv 缓存（venv 不依赖缓存）
rm -rf /root/.cache/uv /root/.cache/pip
df -h /   # 应回到 ~21G 可用
```

## 7. 训练（standing）

```bash
cd ~/odm-playground
export WANDB_MODE=disabled
# 小验证 10M（4090 实测 ~6min）
.venv/bin/python playground/open_duck_mini_v2/runner.py \
  --env standing --task flat_terrain --num_timesteps 10000000 --output_dir checkpoints_verify
# 全量 150M（4090 实测 ~15min，reward 196.8）
.venv/bin/python playground/open_duck_mini_v2/runner.py \
  --env standing --task flat_terrain --num_timesteps 150000000 --output_dir checkpoints_full
```

后台跑（nohup + 日志），ONNX 自动导出到 output_dir。

## 8. 验收（headless，AutoDL 无图形界面可用）

```bash
.venv/bin/python tools/m1_gate_eval.py checkpoints_full/<最新>.onnx 20
# PASS = 20s 站立零跌倒（body z min > 0.05）
```

## 9. 关机前

⚠️ AutoDL 非持久盘，实例释放 = 数据全丢。关机前：
1. ONNX/checkpoint 下载回本地（scp / paramiko）
2. 或传 HuggingFace
3. 不需要实例时控制台关机（只收存储费 ¥0.1/h 级）

## 10. 已知限制（诚实记录）

- mujoco_infer.py（官方 viewer 推理）的 get_obs 是 joystick 布局 101 维，
  standing 训练 obs 是 85 维（gyro3+acc3+cmd7+joint14×5+contact2，无 motor_targets/phase）。
  官方脚本跑 standing ONNX 会 INVALID_ARGUMENT——headless 验收请用 tools/m1_gate_eval.py。
- standing 训练 termination 判据 = 身体翻转（gravity z < 0），不是高度阈值。
- 训练必须 use_pmap_on_reset=True（brax 0.14.2 设计），单卡会包 pmap——apply_brax_fix.py 已兼容。

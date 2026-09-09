#!/bin/bash
# ============================================================
# DuckMate M1 gate · ODM Playground standing 最小重训一键脚本（大鸭 42cm）
# 用法（AutoDL GPU 实例 SSH 后）：
#   bash odm_m1_gate.sh            # 全流程：部署 + 小验证 + 全量
#   bash odm_m1_gate.sh --verify   # 只跑小验证（10M，~20min 看收敛）
# ============================================================
set -e
export PATH="$HOME/.local/bin:$PATH"
MODE="${1:-full}"

echo "=== [1/5] 系统依赖 ==="
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git curl build-essential python3 python3-pip 2>/dev/null || true

echo "=== [2/5] 安装 uv ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

echo "=== [3/5] 检查 GPU ==="
nvidia-smi | head -12 || echo "⚠️ nvidia-smi 不可用，请确认选了 GPU 实例"

echo "=== [4/5] clone Open Duck Playground（公开仓库，大鸭官方训练栈）==="
if [ ! -d "$HOME/odm-playground/.git" ]; then
  cd "$HOME"
  git clone https://github.com/apirrone/Open_Duck_Playground.git odm-playground
fi
cd "$HOME/odm-playground"
git pull --ff-only 2>/dev/null || true
echo "仓库就绪: $(pwd) | 分支: $(git branch --show-current)"

echo "=== [5/5] uv sync（依赖 ~5-10min，jax-cuda + mujoco_playground）==="
uv sync 2>&1 | tail -3 || echo "⚠️ uv sync 失败，请查看输出"

if [ "$MODE" = "--verify" ]; then
  echo ""
  echo "=== 小验证（standing 10M timesteps，~20min，看 reward 收敛）==="
  cd "$HOME/odm-playground"
  WANDB_MODE=disabled uv run playground/open_duck_mini_v2/runner.py \
    --env standing --task flat_terrain \
    --num_timesteps 10000000 \
    --output_dir checkpoints_m1_verify
  echo "✅ 验证完成。看曲线：checkpoints_m1_verify 下训练日志"
  echo "   收敛 OK → 跑全量：bash odm_m1_gate.sh（full）"
else
  echo ""
  echo "=== 全量训练（standing 150M timesteps，~1-2h on 4090）==="
  cd "$HOME/odm-playground"
  WANDB_MODE=disabled uv run playground/open_duck_mini_v2/runner.py \
    --env standing --task flat_terrain \
    --num_timesteps 150000000 \
    --output_dir checkpoints_m1_full
  echo "✅ 全量完成。checkpoint + onnx 在 checkpoints_m1_full/（runner 自动导出）"
fi

echo ""
echo "=== 完成。用完 AutoDL 控制台关机（只收存储费）==="
echo "⚠️ 非持久盘：重要 checkpoint 先下载回本地/传 HF 再关机"
echo ""
echo "=== ONNX 产物说明（已自动导出，无需手动步骤）==="
echo "runner.py 训练中每个 checkpoint 自动 export_onnx → checkpoints_m1_*/<date>_<step>.onnx"
echo ""
echo "=== 真机/仿真验证（下一步，本地或 AutoDL 均可）==="
echo "cd ~/odm-playground && uv run playground/open_duck_mini_v2/mujoco_infer.py -o <策略.onnx>"
echo ""
echo "=== M1 gate 通过标准 ==="
echo "① 仿真站立 20s 不倒（mujoco_infer 目视/录屏）"
echo "② 全程 GPU 成本 < \$3（4090 ≈ \$0.5/h × ~2h）"
echo "③ checkpoint + onnx 已下载回本地（archive/microduck-opportunity/）"

#!/usr/bin/env python3
"""原生/ONNX 动作对拍 — Codex 最小交付规格 (TRACE4V3_FULL_VERIFIED 下一步)

规格:
  1. 现有 obs_before 选 >=100 组, 覆盖 0/0.10 命令、参考相位、首次动作、reset 后首动作
  2. 原生 checkpoint 归一化状态 + 确定性 actor, 同一输入 vs ONNX 对拍, max 误差 <=1e-4
  3. 原生参数加载路径独立于 ONNX 导出路径 (orbax restore + 手动 numpy 前向, 不经 tf2onnx)
  4. 附 checkpoint/ONNX/exporter/配置/网络结构哈希
"""
import os, sys, time, hashlib, subprocess
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import onnxruntime as ort

CKPT_DIR = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160"
ONNX_PATH = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
NPZ_PATH = "/root/trace4_0.1_seed0.npz"
EXPORTER = "/root/odm-playground/playground/common/export_onnx.py"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def sha256_dir(d):
    files = []
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            p = os.path.join(root, f)
            files.append(f"{sha256_file(p)}  {p}")
    return files

# ===== 1. 哈希记录 (规格 4) =====
print("=== 哈希清单 ===", flush=True)
print(f"ckpt dir: {CKPT_DIR}")
for line in sha256_dir(CKPT_DIR):
    print("  " + line)
print(f"onnx: {sha256_file(ONNX_PATH)}  {ONNX_PATH}")
print(f"exporter: {sha256_file(EXPORTER)}  {EXPORTER}")

# ===== 2. 原生 params restore (orbax, 训练原产物) =====
print("\n=== orbax restore ===", flush=True)
sys.path.insert(0, "/root/odm-playground")
from orbax import checkpoint as ocp
ckptr = ocp.PyTreeCheckpointer()
t0 = time.time()
params = ckptr.restore(CKPT_DIR)
print(f"restore done in {time.time()-t0:.1f}s")
print(f"params 顶级类型: {type(params)}, len={len(params) if hasattr(params,'__len__') else 'n/a'}")
norm = params[0]
mean = np.asarray(norm["mean"]["state"], dtype=np.float32)   # (101,)
std = np.asarray(norm["std"]["state"], dtype=np.float32)     # (101,)
pol = params[1]["params"]
Ws, bs = [], []
for i in range(4):
    Ws.append(np.asarray(pol[f"hidden_{i}"]["kernel"], dtype=np.float32))
    bs.append(np.asarray(pol[f"hidden_{i}"]["bias"], dtype=np.float32))
print(f"mean/std: {mean.shape}/{std.shape}")
for i in range(4):
    print(f"hidden_{i}: W{Ws[i].shape} b{bs[i].shape}")

def native_policy(x):
    """手动 numpy 前向 — 与 export_onnx 同构 (swish + tanh(loc)) 但独立实现路径"""
    x = (x - mean) / std
    for i in range(3):
        x = x @ Ws[i] + bs[i]
        x = x * (1.0 / (1.0 + np.exp(-x)))  # swish
    logits = x @ Ws[3] + bs[3]
    loc = logits[..., :14]
    return np.tanh(loc).astype(np.float32)

# ===== 3. ONNX session =====
sess = ort.InferenceSession(ONNX_PATH)
in_name = sess.get_inputs()[0].name
print(f"\nonnx input: {in_name} shape hint: {sess.get_inputs()[0].shape}")
def onnx_policy(x):
    return sess.run(None, {in_name: x.astype(np.float32)})[0]

# ===== 4. 输入选择: 覆盖 0/0.10 命令、相位、首次动作、reset 后首动作 =====
d = np.load(NPZ_PATH)
obs = d["obs_before"]  # (300, 32, 101)
done = d["done"] > 0
cmd = d["cmd"]  # (300, 32, 7)
ep_step = d["episode_step"]
phase = d["phase"]
T, E, _ = obs.shape
flat_obs = obs.reshape(-1, 101)
flat_cmd = cmd.reshape(-1, 7)
walk_mask = (flat_cmd[:, 0] > 0.05)
stand_mask = (flat_cmd[:, 0] < 0.01)
first_act_mask = (ep_step.reshape(-1) == 0)
nxt = np.zeros_like(done); nxt[:-1] = done[1:]
reset_after_mask = (nxt & ~done).reshape(-1)
print(f"\n样本覆盖: walk={walk_mask.sum()} stand={stand_mask.sum()} "
      f"first_act={first_act_mask.sum()} reset_after={reset_after_mask.sum()} "
      f"total={len(flat_obs)}")
# 相位覆盖: 去重相位计数
uniq_phase = np.unique(phase[walk_mask.reshape(T, E)])
print(f"walk 覆盖相位数: {len(uniq_phase)} (of 27): {uniq_phase[:10]}...")

# ===== 5. 对拍 (全样本 9600 — 超 Codex >=100 规格) =====
def onnx_policy(x):
    # ONNX 静态 batch=1 — 逐行推理
    return np.stack([sess.run(None, {in_name: r.reshape(1, -1)})[0][0] for r in x])

def batched_compare(obs_sel, label):
    a_native = native_policy(obs_sel)
    a_onnx = onnx_policy(obs_sel)
    d_ = np.abs(a_native - a_onnx)
    return d_.max(), d_.mean(), d_.argmax()

print("\n=== 对拍 ===", flush=True)
results = {}
for label, m in [
    ("全部 9600", np.ones(len(flat_obs), bool)),
    ("walk cmd=0.10", walk_mask),
    ("stand cmd=0", stand_mask),
    ("首次动作 ep_step=0", first_act_mask),
    ("reset 后首动作", reset_after_mask),
    ("done 行(终止转换)", done.reshape(-1)),
]:
    sel = np.where(m)[0]
    mx, mn, am = batched_compare(flat_obs[sel], label)
    results[label] = (len(sel), float(mx), float(mn))
    print(f"{label:24s}: n={len(sel):6d} max|Δ|={mx:.8f} mean={mn:.8f}", flush=True)

# 最大误差样本 (ONNX 逐行, 只跑一次 — 9600 组)
print("计算最大误差样本...", flush=True)
a_native_all = native_policy(flat_obs)
mx_all = np.abs(a_native_all - onnx_policy(flat_obs)).max(axis=-1)
worst = np.argmax(mx_all)
te, ee = divmod(worst, E)
print(f"最大误差样本: t={te} env={ee} cmd={flat_cmd[worst]} max|Δ|={mx_all[worst]:.8f}")

# ===== 6. 阈值判定 (<=1e-4) =====
ok = all(v[1] <= 1e-4 for v in results.values())
print(f"\n{'PASS ✅' if ok else 'FAIL ❌'}: 全组 max|Δ| <= 1e-4")
for k, v in results.items():
    mark = "✅" if v[1] <= 1e-4 else "❌"
    print(f"  {mark} {k}: max={v[1]:.8f}")

# 网络结构摘要 (规格 4 配置哈希: brax ppo config)
try:
    from playground.common import locomotion_params
    cfg = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    nh = cfg.network_factory.policy_hidden_layer_sizes
    print(f"\npolicy_hidden_layer_sizes: {nh}")
except Exception as ex:
    print(f"\nconfig 读取失败: {ex}")

#!/usr/bin/env python3
"""原生/ONNX 动作对拍 v2 — 回应 ALIGN_REVIEW_AND_OBS_PD_SCOPE

修正项:
  1. reset_after_mask 方向写反 (nxt[:-1]=done[1:] 选的是下一步将 done 行)
     → mask[1:] = done[:-1] (上一行 done = reset 后首行)
  2. phase 覆盖改用 obs_before[99:101] (实际输入相位, 非 step 后 info)
  3. 补 Brax 训练原生 API 一路: network_factory + make_inference_fn deterministic=True
     (手写 numpy 保留为第三路)
  4. 补输入 NPZ sha256 + config (brax_ppo_config dump + sha)
  5. 逐样本动作差落盘 + 有限数检查 + 失败 exit 1
  6. 措辞: 不称 float32 极限
"""
import os, sys, time, hashlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import onnxruntime as ort
import jax
import jax.numpy as jp

CKPT_DIR = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160"
ONNX_PATH = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
NPZ_PATH = "/root/trace4_0.1_seed0.npz"
EXPORTER = "/root/odm-playground/playground/common/export_onnx.py"
CFG_PATH = "/root/odm-playground/.venv/lib/python3.12/site-packages/mujoco_playground/config/locomotion_params.py"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

print("=== 哈希清单 ===", flush=True)
print(f"input npz: {sha256_file(NPZ_PATH)}  {NPZ_PATH}")
print(f"onnx: {sha256_file(ONNX_PATH)}  {ONNX_PATH}")
print(f"exporter: {sha256_file(EXPORTER)}  {EXPORTER}")
print(f"config file: {sha256_file(CFG_PATH)}  {CFG_PATH}")
for root, _, fs in os.walk(CKPT_DIR):
    for f in sorted(fs):
        p = os.path.join(root, f)
        print(f"ckpt: {sha256_file(p)}  {p}")

# ===== config dump (训练实际配置) =====
print("\n=== config ===", flush=True)
sys.path.insert(0, "/root/odm-playground")
from mujoco_playground.config import locomotion_params
cfg = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
nf = cfg.network_factory
print(f"policy_hidden_layer_sizes: {nf.policy_hidden_layer_sizes}")
print(f"value_hidden_layer_sizes: {getattr(nf, 'value_hidden_layer_sizes', 'n/a')}")
print(f"activation: {getattr(nf, 'activation', 'n/a')}")
print(f"normalize_observations: {getattr(nf, 'normalize_observations', 'n/a')}")
print(f"num_timesteps 相关: {getattr(cfg, 'num_timesteps', 'n/a')}")

# ===== orbax restore =====
print("\n=== orbax restore ===", flush=True)
from orbax import checkpoint as ocp
from brax.training.acme import running_statistics
ckptr = ocp.PyTreeCheckpointer()
t0 = time.time()
params_raw = ckptr.restore(CKPT_DIR)
print(f"restore done in {time.time()-t0:.1f}s, top type={type(params_raw).__name__} len={len(params_raw)}")
# params[0] dict → RunningStatisticsState dataclass (brax Normalize 需属性访问)
n0 = params_raw[0]
cnt_hi = np.asarray(n0["count"]["hi"], dtype=np.uint64)
cnt_lo = np.asarray(n0["count"]["lo"], dtype=np.uint64)
cnt = cnt_hi * (1 << 32) + cnt_lo
rs = running_statistics.RunningStatisticsState(
    count=jax.numpy.asarray(cnt), mean=n0["mean"],
    summed_variance=n0["summed_variance"],
    std=n0["std"],
    std_eps=float(np.asarray(n0["std_eps"])))
params = (rs, params_raw[1], params_raw[2])
print(f"count={int(cnt)} std_eps={np.asarray(n0['std_eps'])}")
norm = params[0]
mean = np.asarray(norm.mean["state"], dtype=np.float32)
std = np.asarray(norm.std["state"], dtype=np.float32)
pol = params[1]["params"]
Ws, bs = [], []
for i in range(4):
    Ws.append(np.asarray(pol[f"hidden_{i}"]["kernel"], dtype=np.float32))
    bs.append(np.asarray(pol[f"hidden_{i}"]["bias"], dtype=np.float32))
print(f"mean/std: {mean.shape}/{std.shape}")

# ===== 路 1: 手写 numpy (参考 exporter 结构) =====
def native_policy_numpy(x):
    x = (x - mean) / std
    for i in range(3):
        x = x @ Ws[i] + bs[i]
        x = x * (1.0 / (1.0 + np.exp(-x)))  # swish
    logits = x @ Ws[3] + bs[3]
    return np.tanh(logits[..., :14]).astype(np.float32)

# ===== 路 2: Brax 训练原生 API (network_factory + make_inference_fn) =====
print("\n=== brax inference fn ===", flush=True)
from brax.training.agents.ppo import networks as ppo_networks

# 与训练同源: brax_ppo_config(BerkeleyHumanoidJoystickFlatTerrain).network_factory
# + env obs dict {state:101, privileged_state:212}
from brax.training.acme import running_statistics
ppo_nets = ppo_networks.make_ppo_networks(
    observation_size={"state": 101, "privileged_state": 212},
    action_size=14,
    preprocess_observations_fn=running_statistics.normalize,  # 训练同款 (normalize_observations=True)
    policy_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_hidden_layer_sizes=(512, 256, 128),
    value_obs_key="privileged_state",
)
make_policy = ppo_networks.make_inference_fn(ppo_nets)
policy = make_policy(params, deterministic=True)

def native_policy_brax(x):
    # 逐行 (vmap 会重排浮点引入 ~5e-4 差异 — 实测确认)
    return np.stack([np.asarray(policy({"state": jp.array(r)}, jax.random.PRNGKey(0))[0])
                     for r in x])
# 编译预热 (取 2 行)
_ = native_policy_brax(np.zeros((2, 101), np.float32))
print("brax inference fn 编译完成", flush=True)

# ===== 路 3: ONNX =====
sess = ort.InferenceSession(ONNX_PATH)
in_name = sess.get_inputs()[0].name
def onnx_policy(x):
    return np.stack([sess.run(None, {in_name: r.reshape(1, -1)})[0][0] for r in x])

# ===== 输入 + 分组 (mask 修正) =====
d = np.load(NPZ_PATH)
obs = d["obs_before"]
done = d["done"] > 0
cmd = d["cmd"]
ep_step = d["episode_step"]
T, E, _ = obs.shape
flat_obs = obs.reshape(-1, 101)
flat_cmd = cmd.reshape(-1, 7)
walk_mask = (flat_cmd[:, 0] > 0.05)
stand_mask = (flat_cmd[:, 0] < 0.01)
first_act_mask = (ep_step.reshape(-1) == 0)
# 修正: 上一行 done 且本行非 done = reset 后首行
reset_after_mask = np.zeros(len(flat_obs), bool)
rm = reset_after_mask.reshape(T, E)
rm[1:] = done[:-1]
rm &= ~done
# 输入相位 (obs_before[99:101])
in_phase = obs[:, :, 99:101].reshape(-1, 2)
uniq_in_phase = np.unique(in_phase[walk_mask], axis=0)
print(f"\n样本覆盖: walk={walk_mask.sum()} stand={stand_mask.sum()} "
      f"first_act={first_act_mask.sum()} reset_after={reset_after_mask.sum()} "
      f"total={len(flat_obs)}")
print(f"walk 输入相位对 (obs[99:101]) 唯一数: {len(uniq_in_phase)}")

# ===== 三路对拍 =====
print("\n=== 三路对拍 ===", flush=True)
a_np = native_policy_numpy(flat_obs)     # 路1 手写 numpy (全量)
a_brx = native_policy_brax(flat_obs)     # 路2 brax (全量)
a_onnx = onnx_policy(flat_obs)           # 路3 onnx (逐行)
# 有限数检查
for name, a in [("numpy", a_np), ("brax", a_brx), ("onnx", a_onnx)]:
    if not np.isfinite(a).all():
        print(f"FAIL: {name} 含非有限数"); sys.exit(1)
print("三路输出全部有限 ✅")

groups = [
    ("全部 9600", np.ones(len(flat_obs), bool)),
    ("walk cmd=0.10", walk_mask),
    ("stand cmd=0", stand_mask),
    ("首次动作 ep_step=0", first_act_mask),
    ("reset 后首动作(修正)", reset_after_mask),
    ("done 行(终止转换)", done.reshape(-1)),
]
results = {}
for label, m in groups:
    sel = np.where(m)[0]
    if len(sel) == 0:
        print(f"{label}: n=0 跳过"); continue
    dbn = np.abs(a_np[sel] - a_brx[sel]).max()
    dno = np.abs(a_np[sel] - a_onnx[sel]).max()
    dbo = np.abs(a_brx[sel] - a_onnx[sel]).max()
    results[label] = (len(sel), float(dbn), float(dno), float(dbo))
    print(f"{label:26s}: n={len(sel):6d} | numpy-vs-brax={dbn:.8f} "
          f"numpy-vs-onnx={dno:.8f} brax-vs-onnx={dbo:.8f}", flush=True)

# 逐样本动作差落盘 (brax-vs-onnx + numpy-vs-onnx)
per_sample = np.abs(a_brx - a_onnx).max(axis=-1)
np.savez("/root/align_per_sample.npz",
         brax_vs_onnx=per_sample,
         numpy_vs_onnx=np.abs(a_np - a_onnx).max(axis=-1),
         brax_vs_numpy=np.abs(a_brx - a_np).max(axis=-1))
worst = np.argmax(per_sample)
te, ee = divmod(worst, E)
print(f"\nbrax-vs-onnx 最大误差样本: t={te} env={ee} cmd={flat_cmd[worst]} max|Δ|={per_sample[worst]:.8f}")

ok = all(v[1] <= 1e-4 and v[2] <= 1e-4 and v[3] <= 1e-4 for v in results.values())
print(f"\n{'PASS ✅' if ok else 'FAIL ❌'}: 全组三路两两 max|Δ| <= 1e-4")
for k, v in results.items():
    mark = "✅" if all(x <= 1e-4 for x in v[1:]) else "❌"
    print(f"  {mark} {k}: n={v[0]} np-brx={v[1]:.2e} np-onnx={v[2]:.2e} brx-onnx={v[3]:.2e}")
if not ok:
    sys.exit(1)
print("逐样本差已存 /root/align_per_sample.npz")

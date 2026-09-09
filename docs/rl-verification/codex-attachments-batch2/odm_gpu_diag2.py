#!/usr/bin/env python3
"""GPU 诊断 v2: 训练同构批处理 rollout（VmapWrapper 64 env）+ 150M onnx actor
每步 env.step 一次 jit 编译后复用; metrics 分项 mean over batch
用法: python3 /root/odm_gpu_diag2.py <onnx> <cmd_vx> <seed> <steps>
"""
import sys, os, json, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/root/odm-playground")
from mujoco_playground import wrapper
from playground.open_duck_mini_v2.joystick import Joystick

NUM_ENVS = 64

def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
    cmd_vx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 480

    print(f"[diag2] onnx={os.path.basename(ckpt)} cmd_vx={cmd_vx} seed={seed} steps={steps} n_envs={NUM_ENVS}", flush=True)
    sess = ort.InferenceSession(ckpt)
    in_name = sess.get_inputs()[0].name

    env = wrapper.wrap_for_brax_training(Joystick(task="flat_terrain"))
    print(f"[diag2] wrapped env ok obs={env.observation_size} act={env.action_size}", flush=True)

    rng = jax.random.split(jax.random.PRNGKey(seed), NUM_ENVS)
    state = env.reset(rng)
    # 固定 cmd（前 500 步不重采样, 单 episode 480 步）
    cmd = jp.broadcast_to(jp.array([cmd_vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (NUM_ENVS, 7))
    state.info["command"] = cmd
    print(f"[diag2] reset done cmd={np.asarray(state.info['command'])[0,:2]}", flush=True)

    @jax.jit
    def step_fn(s, a):
        return env.step(s, a)

    # warmup compile
    obs0 = np.asarray(state.obs["state"], dtype=np.float32)
    act0 = np.stack([sess.run(None, {in_name: obs0[i:i+1]})[0][0] for i in range(NUM_ENVS)])
    t0 = time.time()
    state = step_fn(state, jp.array(act0))
    print(f"[diag2] compile+step0 done {time.time()-t0:.1f}s z={float(np.asarray(state.data.qpos[2]).mean()):.4f}", flush=True)

    keys = ["reward/tracking_lin_vel", "reward/tracking_ang_vel", "reward/alive",
            "reward/imitation", "cost/torques", "cost/action_rate", "cost/stand_still"]
    agg = {k: [] for k in keys}
    agg["reward_total"] = []
    agg["z"] = []
    t1 = time.time()
    for i in range(1, steps):
        obs = np.asarray(state.obs["state"], dtype=np.float32)
        action = np.stack([sess.run(None, {in_name: obs[j:j+1]})[0][0] for j in range(NUM_ENVS)])
        state = step_fn(state, jp.array(action))
        for k in keys:
            if k in state.metrics:
                agg[k].append(float(np.asarray(state.metrics[k]).mean()))
        agg["reward_total"].append(float(np.asarray(state.reward).mean()))
        agg["z"].append(float(np.asarray(state.data.qpos[2]).mean()))
        if (i + 1) % 120 == 0:
            print(f"[diag2] step {i+1}/{steps} {time.time()-t1:.1f}s z={agg['z'][-1]:.4f} r={agg['reward_total'][-1]:.4f}", flush=True)
    dt_env = time.time() - t1
    print(f"[diag2] rollout done {steps} steps in {dt_env:.1f}s ({steps/dt_env:.0f} step/s)", flush=True)

    def stats(v):
        v = np.array(v)
        return {"mean": round(float(v.mean()), 5), "p5": round(float(np.percentile(v, 5)), 5),
                "p50": round(float(np.percentile(v, 50)), 5), "p95": round(float(np.percentile(v, 95)), 5),
                "min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                "zero_frac": round(float((v == 0).mean()), 4)} if len(v) else None
    # 速度估计: qpos[0:2] 位移差分
    out = {"ckpt": os.path.basename(ckpt), "cmd_vx": cmd_vx, "seed": seed, "steps": steps, "n_envs": NUM_ENVS,
           "real_time_s": round(dt_env, 1),
           "metrics": {k: stats(v) for k, v in agg.items()},
           "reward_total_stats": stats(agg["reward_total"]),
           "z_mean": float(np.array(agg["z"]).mean()),
           "clip_zero_total_frac": float((np.array(agg["reward_total"]) == 0).mean())}
    print(json.dumps(out, indent=1))
    with open(f"/root/gpu_diag2_{cmd_vx}_seed{seed}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("[diag2] saved")

if __name__ == "__main__":
    main()

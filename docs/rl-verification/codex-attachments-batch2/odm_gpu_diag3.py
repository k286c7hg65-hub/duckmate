#!/usr/bin/env python3
"""GPU 诊断 v3: cmd sweep 0.0/0.05/0.10/0.15 x seeds 0-1 (单进程复用编译)
输出每个 (cmd, seed) 的分项 stats 到 json
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
CMDS = [0.0, 0.05, 0.10, 0.15]
SEEDS = [0, 1]
STEPS = 480

def main():
    ckpt = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
    print(f"[diag3] sweep cmds={CMDS} seeds={SEEDS} n_envs={NUM_ENVS} steps={STEPS}", flush=True)
    sess = ort.InferenceSession(ckpt)
    in_name = sess.get_inputs()[0].name

    env = wrapper.wrap_for_brax_training(Joystick(task="flat_terrain"))
    print(f"[diag3] env ok", flush=True)

    @jax.jit
    def step_fn(s, a):
        return env.step(s, a)

    keys = ["reward/tracking_lin_vel", "reward/tracking_ang_vel", "reward/alive",
            "reward/imitation", "cost/torques", "cost/action_rate", "cost/stand_still"]
    results = {}
    t_compile = None
    for cmd_vx in CMDS:
        for seed in SEEDS:
            rng = jax.random.split(jax.random.PRNGKey(seed), NUM_ENVS)
            state = env.reset(rng)
            cmd = jp.broadcast_to(jp.array([cmd_vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (NUM_ENVS, 7))
            state.info["command"] = cmd
            obs0 = np.asarray(state.obs["state"], dtype=np.float32)
            act0 = np.stack([sess.run(None, {in_name: obs0[j:j+1]})[0][0] for j in range(NUM_ENVS)])
            t0 = time.time()
            state = step_fn(state, jp.array(act0))
            if t_compile is None:
                t_compile = time.time() - t0
                print(f"[diag3] compile {t_compile:.1f}s", flush=True)
            agg = {k: [] for k in keys}
            agg["reward_total"] = []
            agg["vx_est"] = []
            prev_xy = np.asarray(state.data.qpos[:, :2]).copy()
            for i in range(1, STEPS):
                obs = np.asarray(state.obs["state"], dtype=np.float32)
                action = np.stack([sess.run(None, {in_name: obs[j:j+1]})[0][0] for j in range(NUM_ENVS)])
                state = step_fn(state, jp.array(action))
                for k in keys:
                    if k in state.metrics:
                        agg[k].append(float(np.asarray(state.metrics[k]).mean()))
                agg["reward_total"].append(float(np.asarray(state.reward).mean()))
                xy = np.asarray(state.data.qpos[:, :2])
                if i % 20 == 0:  # vx 估计每 20 步
                    agg["vx_est"].append(float(np.linalg.norm(xy - prev_xy, axis=1).mean() / (20 * 0.02)))
                    prev_xy = xy.copy()
            dt_env = time.time() - t0
            def stats(v):
                v = np.array(v)
                return {"mean": round(float(v.mean()), 5), "p5": round(float(np.percentile(v, 5)), 5),
                        "p50": round(float(np.percentile(v, 50)), 5), "p95": round(float(np.percentile(v, 95)), 5),
                        "zero_frac": round(float((v == 0).mean()), 4)} if len(v) else None
            tag = f"cmd{cmd_vx}_seed{seed}"
            results[tag] = {
                "metrics": {k: stats(v) for k, v in agg.items()},
                "reward_total": stats(agg["reward_total"]),
                "vx_est_mean": stats(agg["vx_est"]),
                "z_mean": float(np.asarray(state.data.qpos[2]).mean()),
                "time_s": round(dt_env, 1),
            }
            print(f"[diag3] {tag} done {dt_env:.1f}s r={results[tag]['reward_total']['mean']} vx={results[tag]['vx_est_mean']['mean']} z={results[tag]['z_mean']:.4f}", flush=True)
    out = {"ckpt": os.path.basename(ckpt), "n_envs": NUM_ENVS, "steps": STEPS, "compile_s": round(t_compile, 1), "results": results}
    with open("/root/gpu_diag3_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
    print("[diag3] saved /root/gpu_diag3_sweep.json")

if __name__ == "__main__":
    main()

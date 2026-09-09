#!/usr/bin/env python3
"""GPU trace: cmd 0.10 站姿 rollout 中 imitation 逐步分解 (B 各 exp 项 / E_q / E_dq / actual vs ref joint)
直接读训练 env state, 零重建误差"""
import sys, os, json, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/root/odm-playground")
from mujoco_playground import wrapper
from playground.open_duck_mini_v2.joystick import Joystick
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

NUM_ENVS = 32
STEPS = 240

def main():
    ckpt = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
    cmd_vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f"[trace] cmd={cmd_vx} seed={seed} n={NUM_ENVS} steps={STEPS}", flush=True)
    sess = ort.InferenceSession(ckpt)
    in_name = sess.get_inputs()[0].name

    env = wrapper.wrap_for_brax_training(Joystick(task="flat_terrain"))
    @jax.jit
    def step_fn(s, a):
        return env.step(s, a)

    rng = jax.random.split(jax.random.PRNGKey(seed), NUM_ENVS)
    state = env.reset(rng)
    cmd = jp.broadcast_to(jp.array([cmd_vx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (NUM_ENVS, 7))
    state.info["command"] = cmd
    obs0 = np.asarray(state.obs["state"], dtype=np.float32)
    act0 = np.stack([sess.run(None, {in_name: obs0[j:j+1]})[0][0] for j in range(NUM_ENVS)])
    state = step_fn(state, jp.array(act0))
    print("[trace] compiled", flush=True)

    # 分解 helper (纯 numpy, 从 state 提)
    def decompose(s, i_env):
        # actual
        aq = np.asarray(s.data.qpos[i_env])  # 7+14
        joint_actual = np.concatenate([aq[7:12], aq[16:21]])  # [:5]+[9:] of 14
        # ref
        ref = np.asarray(s.info["current_reference_motion"][i_env])  # 40
        ref_jp = np.concatenate([ref[:5], ref[11:16]])
        Eq = float(np.sum((joint_actual - ref_jp) ** 2))
        # base
        bq = aq[:7]; bq = bq / np.linalg.norm(bq)
        rq = ref[3:7]; rq = rq / np.linalg.norm(rq)
        # 实际 base vel 是 world? custom 里 base_qvel[:3] (body frame?)
        return {"joint_actual": np.round(joint_actual, 3).tolist(),
                "ref_jp": np.round(ref_jp, 3).tolist(),
                "Eq": Eq, "I_metric": float(np.asarray(s.metrics["reward/imitation"]).mean()),
                "z": float(aq[2])}

    rows = []
    for i in range(1, STEPS):
        obs = np.asarray(state.obs["state"], dtype=np.float32)
        action = np.stack([sess.run(None, {in_name: obs[j:j+1]})[0][0] for j in range(NUM_ENVS)])
        state = step_fn(state, jp.array(action))
        if i % 10 == 0:
            for e in range(min(3, NUM_ENVS)):
                rows.append(decompose(state, e))
    # 聚合
    Eq_all = [r["Eq"] for r in rows]
    I_all = [r["I_metric"] for r in rows]
    out = {
        "cmd": cmd_vx, "seed": seed, "steps": STEPS, "n_envs": NUM_ENVS,
        "Eq_stats": {"mean": float(np.mean(Eq_all)), "p10": float(np.percentile(Eq_all, 10)),
                     "p50": float(np.percentile(Eq_all, 50)), "p90": float(np.percentile(Eq_all, 90)),
                     "max": float(np.max(Eq_all))},
        "I_metric_mean": float(np.mean(I_all)),
        "sample_rows": rows[:6],
    }
    print(json.dumps(out, indent=1))
    with open(f"/root/gpu_trace_{cmd_vx}_seed{seed}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("[trace] saved")

if __name__ == "__main__":
    main()

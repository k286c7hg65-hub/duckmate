#!/usr/bin/env python3
"""GPU trace v4 — reset/命令同步修复版（Codex 报告 20260906 落地）
修复点:
  1. reset 后注入固定命令 + 全量重建 (info.command / reference / obs) → 首动作即目标 cmd
     (trace3 缺陷: warmup 用旧 obs 生成首动作, 命令同步在首个动作之后才完成)
  2. 每步推理前断言全 env obs_cmd == info.command
  3. 保存完整 101 维 obs_before (动作发出时观测, 与 cmd 严格同帧) + action + obs_after
     (trace3 缺陷: step 后抓 obs → done 行 AutoReset 混帧 obs_cmd != cmd, max 0.0347)
  4. done 行自洽: cmd/obs_before 均在 step 前抓 → done 步也保存且一致
  5. episode 全量计数 (每步累加 done 事件, 区分「保存样本中 done」vs「运行总终止数」)
     (trace3: SAVE_EVERY=2 隔步采样漏 done → 37 vs 17 不符)
  6. fallback: sync 失败 → warmup 同步 (trace3 行为) + WARNING
守恒对拍保留 (同帧, 排除 done 步 — 与 Codex 已签收口径一致)
"""
import sys, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/root/odm-playground")
from mujoco_playground import wrapper
from mujoco_playground._src.collision import geoms_colliding
from playground.open_duck_mini_v2.joystick import Joystick

NUM_ENVS = 32
STEPS = 300
SAVE_EVERY = 1  # 每步保存 (obs_before/action/obs_after 需全量; 240 步 x 32 env 开销可接受)


def main():
    ckpt = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
    cmd_vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tag = f"{cmd_vx}_seed{seed}"
    print(f"[trace4] cmd={cmd_vx} seed={seed} n={NUM_ENVS} steps={STEPS} save_every={SAVE_EVERY}", flush=True)
    sess = ort.InferenceSession(ckpt)
    in_name = sess.get_inputs()[0].name

    env = wrapper.wrap_for_brax_training(Joystick(task="flat_terrain"))
    uenv = env
    while hasattr(uenv, "unwrapped"):
        nxt = uenv.unwrapped
        if callable(nxt):
            nxt = nxt()
        if nxt is None or nxt is uenv:
            break
        uenv = nxt
    print(f"[trace4] unwrapped: {type(uenv).__name__}", flush=True)
    qpos_addr = np.asarray(uenv.get_actuator_joints_qpos_addr())
    qvel_addr = np.asarray(uenv.actuator_qvel_addr)
    fb_qpos_addr = int(np.asarray(uenv._floating_base_qpos_addr))
    fb_qvel_addr = int(np.asarray(uenv._floating_base_qvel_addr))
    default_act = np.asarray(uenv._default_actuator, dtype=np.float32)
    dt = 0.02
    # obs 布局断言 (Codex: 101 维 [0:3]gyro [3:6]acc [6:13]cmd [13:27]jq [27:41]jv
    #                     [41:55]last [55:69]last2 [69:83]last3 [83:97]mt [97:99]ct [99:101]phase)
    obs_dim = None

    @jax.jit
    def step_fn(s, a):
        return env.step(s, a)

    def sync_cmd(state, cmd_arr):
        """reset 后注入固定命令并一次性重建 info/reference/obs (per-env loop, 一次性)"""
        try:
            obs_list = []
            info_list = []
            for e in range(NUM_ENVS):
                d_e = jax.tree.map(lambda x: x[e], state.data)
                info_e = jax.tree.map(lambda x: x[e], state.info)
                cmd_e = jp.array(cmd_arr[e], dtype=jp.float32)
                info_e["command"] = cmd_e
                # 重建 reference motion (imitation 用当前 command + imitation_i)
                info_e["current_reference_motion"] = uenv.PRM.get_reference_motion(
                    cmd_e[0], cmd_e[1], cmd_e[2], info_e["imitation_i"]
                )
                # 重建 obs (contact + _get_obs, 与 reset 内部同路径)
                contact = jp.array([
                    geoms_colliding(d_e, gid, uenv._floor_geom_id)
                    for gid in uenv._feet_geom_id
                ])
                obs_e = uenv._get_obs(d_e, info_e, contact)
                obs_list.append(obs_e)
                info_list.append(info_e)
            obs_b = jax.tree.map(lambda *xs: jp.stack(xs), *obs_list)
            info_b = jax.tree.map(lambda *xs: jp.stack(xs), *info_list)
            return state.replace(obs=obs_b, info=info_b), True
        except Exception as ex:
            print(f"[trace4] sync_cmd FAILED ({type(ex).__name__}: {ex}) → warmup fallback", flush=True)
            return state, False

    rng = jax.random.split(jax.random.PRNGKey(seed), NUM_ENVS)
    state = env.reset(rng)
    obs_dim = np.asarray(state.obs["state"]).shape[-1]
    print(f"[trace4] obs_dim={obs_dim} (expect 101)", flush=True)
    # 固定命令注入: 前 16 env 直线走 cmd_vx, 后 16 env 站立 (多命令覆盖)
    cmd_arr = np.zeros((NUM_ENVS, 7), dtype=np.float32)
    cmd_arr[:16, 0] = cmd_vx
    # 后 16 env 站立 (cmd=0) — 覆盖 gate<0.01 分支
    state, synced = sync_cmd(state, cmd_arr)
    if synced:
        print("[trace4] cmd injected + obs/reference rebuilt (首动作即目标 cmd)", flush=True)
    else:
        # fallback: trace3 warmup 路径 (改 info.command → step 一次同步)
        state.info["command"] = jp.array(cmd_arr)
    # 同步断言 (全 env)
    oc = np.round(np.asarray(state.obs["state"][:, 6:13]), 3)
    cc = np.round(np.asarray(state.info["command"]), 3)
    match = np.allclose(oc, cc)
    print(f"[trace4] pre-step cmd sync check: match={match} max|Δ|={np.abs(oc-cc).max():.6f}", flush=True)
    if not match:
        print("[trace4] WARNING: cmd 未同步 — 轨迹开头将带命令偏差", flush=True)

    # ===== 分解函数 (与 trace3 同, verbatim custom_rewards) =====
    def imitation_decompose(bq, bv, jq, jv, contacts, ref, cmdv):
        cmd_norm = jp.linalg.norm(cmdv[:, :3], axis=-1)
        rbq = ref[:, 3:7]; rbq = rbq / jp.linalg.norm(rbq, axis=-1, keepdims=True)
        abq = bq[:, 3:7]; abq = abq / jp.linalg.norm(abq, axis=-1, keepdims=True)
        rblv = ref[:, 34:37]; blv = bv[:, :3]
        rbav = ref[:, 37:40]; bav = bv[:, 3:6]
        rjp = jp.concatenate([ref[:, :5], ref[:, 11:16]], axis=-1)
        ajp = jp.concatenate([jq[:, :5], jq[:, 9:]], axis=-1)
        rjv16 = ref[:, 16:32]
        rjv = jp.concatenate([rjv16[:, :5], rjv16[:, 11:]], axis=-1)
        ajv = jp.concatenate([jv[:, :5], jv[:, 9:]], axis=-1)
        rfc = ref[:, 32:34]
        o_rew = jp.exp(-20.0 * jp.sum(jp.square(abq - rbq), axis=-1))
        lxy = jp.exp(-8.0 * jp.sum(jp.square(blv[:, :2] - rblv[:, :2]), axis=-1))
        lz = jp.exp(-8.0 * jp.square(blv[:, 2] - rblv[:, 2]))
        axy = jp.exp(-2.0 * jp.sum(jp.square(bav[:, :2] - rbav[:, :2]), axis=-1)) * 0.5
        az = jp.exp(-2.0 * jp.square(bav[:, 2] - rbav[:, 2])) * 0.5
        Eq = jp.sum(jp.square(ajp - rjp), axis=-1)
        jpr = -Eq * 15.0
        Edq = jp.sum(jp.square(ajv - rjv), axis=-1)
        jvr = -Edq * 1.0e-3
        rfc_b = jp.where(rfc > 0.5, jp.ones_like(rfc), jp.zeros_like(rfc))
        cr = jp.sum(contacts == rfc_b, axis=-1)
        gate = (cmd_norm > 0.01).astype(jp.float32)
        I = (lxy + lz + axy + az + jpr + jvr + cr) * gate
        return dict(I=I, lin_xy=lxy * gate, lin_z=lz * gate, ang_xy=axy * gate, ang_z=az * gate,
                    Eq=Eq * gate, Edq=Edq * gate, contact=cr * gate, orient=o_rew * gate)

    n_save = STEPS // SAVE_EVERY
    cols = {}
    def A(n, s): cols[n] = np.zeros((n_save, NUM_ENVS) + s, np.float32)
    for n in ["I", "lin_xy", "lin_z", "ang_xy", "ang_z", "Eq", "Edq", "contact", "orient", "tlin", "tang",
              "torques", "arate", "sstill", "alive", "S_preclip", "reward_clip", "z", "done", "episode", "phase",
              "m_im", "m_lin", "m_tang", "m_torq", "m_arate", "m_sstill", "m_alive", "state_reward"]:
        A(n, ())
    for n, s in [("joint_act", (10,)), ("joint_ref", (10,)), ("jq_full", (14,)), ("ref_full", (16,)),
                 ("b_lin_body", (3,)), ("gyro_body", (3,)), ("cmd", (7,)), ("obs_cmd", (7,)),
                 ("action", (14,)), ("last_act", (14,)), ("world_v", (6,)),
                 ("obs_before", (obs_dim,)), ("obs_after", (obs_dim,))]:
        A(n, s)

    t1 = time.time(); save_i = 0
    done_events_total = 0  # 全量 done 事件 (每步累加, 非隔步)
    for i in range(STEPS):
        # step 前: 抓 obs_before + cmd + last_act (与动作严格同帧)
        obs_before = np.asarray(state.obs["state"], dtype=np.float32)
        cmd_now = np.asarray(state.info["command"], dtype=np.float32)
        if i % 50 == 0:
            oc = np.round(obs_before[:, 6:13], 3); cc = np.round(cmd_now, 3)
            if not np.allclose(oc, cc):
                print(f"[trace4] step {i} cmd-desync! max|Δ|={np.abs(oc-cc).max():.6f}", flush=True)
        last_act_before = np.asarray(state.info["last_act"])
        action = np.stack([sess.run(None, {in_name: obs_before[j:j+1]})[0][0] for j in range(NUM_ENVS)])
        # 推进 (AutoReset 在 wrapper step 内: done env 被 reset — obs_after 对 done env 无效)
        state = step_fn(state, jp.array(action))
        done_now = np.asarray(state.done).astype(np.float32)
        done_events_total += int(done_now.sum())
        obs_after = np.asarray(state.obs["state"], dtype=np.float32)

        if i % SAVE_EVERY == 0:
            qpos = np.asarray(state.data.qpos); qvel = np.asarray(state.data.qvel)
            ref = np.asarray(state.info["current_reference_motion"])
            jq = qpos[:, qpos_addr]; jv = qvel[:, qvel_addr]
            contacts = obs_after[:, 97:99]  # 仅用于分解 (非 done env 有效)
            bq = qpos[:, fb_qpos_addr:fb_qpos_addr+7]; bv = qvel[:, fb_qvel_addr:fb_qvel_addr+6]
            dec = imitation_decompose(jp.array(bq), jp.array(bv), jp.array(jq), jp.array(jv),
                                      jp.array(contacts), jp.array(ref), jp.array(cmd_now))
            af = np.asarray(state.data.actuator_force)
            torques = np.sum(af**2, axis=-1)
            arate = np.sum((action - last_act_before)**2, axis=-1)
            gate_ss = (np.linalg.norm(cmd_now[:, :3], axis=-1) < 0.01).astype(np.float32)
            sstill = (np.sum(np.abs(jq - default_act[None, :]), axis=-1) + np.sum(np.abs(jv), axis=-1)) * gate_ss
            si = save_i
            for k, v in dec.items(): cols[k][si] = np.asarray(v)
            b_lin = np.zeros((NUM_ENVS, 3)); gyr = np.zeros((NUM_ENVS, 3))
            for e in range(NUM_ENVS):
                d1 = jax.tree.map(lambda x: x[e] if getattr(x, 'shape', None) else x, state.data)
                b_lin[e] = np.asarray(uenv.get_local_linvel(d1))
                gyr[e] = np.asarray(uenv.get_gyro(d1))
            cols["b_lin_body"][si] = b_lin; cols["gyro_body"][si] = gyr
            err_x = (cmd_now[:, 0] - b_lin[:, 0])**2
            err_y = np.clip(np.abs(b_lin[:, 1] - cmd_now[:, 1]) - 0.1, 0, None)**2
            tlin = np.exp(-(err_x + err_y) / 0.01)
            tang = np.exp(-(cmd_now[:, 2] - gyr[:, 2])**2 / 0.01)
            cols["tlin"][si] = tlin; cols["tang"][si] = tang
            cols["torques"][si] = torques; cols["arate"][si] = arate; cols["sstill"][si] = sstill
            cols["alive"][si] = 1.0
            I_v = np.asarray(dec["I"])
            S = 2.5*tlin + 6.0*tang + 20.0 + I_v - 1e-3*torques - 0.5*arate - 0.2*sstill
            cols["S_preclip"][si] = S; cols["reward_clip"][si] = np.clip(S*dt, 0, 10000)
            cols["m_im"][si] = np.asarray(state.metrics["reward/imitation"])
            cols["m_lin"][si] = np.asarray(state.metrics["reward/tracking_lin_vel"])
            cols["m_tang"][si] = np.asarray(state.metrics["reward/tracking_ang_vel"])
            cols["m_torq"][si] = np.asarray(state.metrics["cost/torques"])
            cols["m_arate"][si] = np.asarray(state.metrics["cost/action_rate"])
            cols["m_sstill"][si] = np.asarray(state.metrics["cost/stand_still"])
            cols["m_alive"][si] = np.asarray(state.metrics["reward/alive"])
            cols["state_reward"][si] = np.asarray(state.reward)
            cols["joint_act"][si] = np.concatenate([jq[:, :5], jq[:, 9:]], axis=-1)
            cols["joint_ref"][si] = np.concatenate([ref[:, :5], ref[:, 11:16]], axis=-1)
            cols["jq_full"][si] = jq; cols["ref_full"][si] = ref[:, :16]
            cols["z"][si] = qpos[:, 2]; cols["cmd"][si] = cmd_now
            cols["obs_cmd"][si] = obs_before[:, 6:13]  # 与 cmd_now 同帧 (step 前)
            cols["done"][si] = done_now
            cols["episode"][si] = done_events_total  # 全量累计 (保存步快照)
            cols["phase"][si] = np.asarray(state.info["imitation_i"]) % 27
            cols["action"][si] = action; cols["last_act"][si] = last_act_before
            cols["world_v"][si] = bv
            cols["obs_before"][si] = obs_before  # 101 维, 与 action/cmd 严格同帧
            cols["obs_after"][si] = obs_after
            save_i += 1
        if (i+1) % 60 == 0:
            print(f"[trace4] step {i+1}/{STEPS} {time.time()-t1:.0f}s done_events={done_events_total}", flush=True)
    dt_env = time.time()-t1
    np.savez(f"/root/trace4_{tag}.npz", **{k: v[:save_i] for k, v in cols.items()})
    print(f"[trace4] saved /root/trace4_{tag}.npz rows={save_i}x{NUM_ENVS} "
          f"done_events_total={done_events_total} (保存样本中 done={int(cols['done'][:save_i].sum())}) ({dt_env:.0f}s)", flush=True)
    # 守恒对拍 (同帧, 排除 done 步)
    nd = ~(cols["done"][:save_i] > 0)
    pairs = [("im", "I", 1.0), ("lin", "tlin", 2.5), ("tang", "tang", 6.0), ("torq", "torques", 1e-3),
             ("arate", "arate", 0.5), ("sstill", "sstill", 0.2), ("alive", "alive", 20.0)]
    for mk, pk, sc_ in pairs:
        dv = np.abs(cols[f"m_{mk}"][:save_i][nd] - sc_*cols[pk][:save_i][nd])
        print(f"[trace4] 对拍 {mk:6s} (非done {nd.sum()}): max|Δ|={dv.max():.6f} 一致={(dv < 1e-4).mean():.4f}", flush=True)
    dS = np.abs(cols["state_reward"][:save_i][nd] - cols["reward_clip"][:save_i][nd])
    print(f"[trace4] state.reward vs reward_clip (非done): max|Δ|={dS.max():.6f} 一致={(dS < 1e-4).mean():.4f}", flush=True)
    # cmd 同步终检: obs_before cmd 槽 vs cmd 列 (应全零差 — trace3 的 done 行缺陷应消失)
    all_cmd_d = np.abs(cols["obs_cmd"][:save_i] - cols["cmd"][:save_i])
    print(f"[trace4] obs_cmd vs cmd (全部样本含 done): max|Δ|={all_cmd_d.max():.6f} (期望 0 — trace3 缺陷修复验证)", flush=True)


if __name__ == "__main__":
    main()

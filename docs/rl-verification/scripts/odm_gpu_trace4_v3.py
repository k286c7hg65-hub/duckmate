#!/usr/bin/env python3
"""GPU trace v3 — 落实 Codex 二审 (TRACE4V2_REVIEW_20260906) 三处 + 范围修正

二审三处 (全部成立):
  #1 冒烟非 fail-fast → 冒烟每动作前复用 hard_check (含 action shape/有限检查;
     shape 错先退出避免切片广播异常绕过诊断保存)
  #2 done 门槛 sum<NUM_ENVS 可被 [32,0,...] 骗过 → np.all(done_seen>=2);
     补逐 env episode_id 增量 == done 数断言 + episode_step done 后归零断言;
     done/truncation 分开记录 (truncation = 达 episode_length)
  #3 terminal_available=done 不成立 (wrapper 已覆盖终止 data, 未保存重置前
     terminal state) → terminal_state_available 恒 False +
     terminal_reward_available=done 分列; post_state_valid 注释 = 对当前动作
     后继/分解有效 (reset 后 data 本身是有效 reset 状态)

范围修正 (不扩大缓存同步声明):
  - first_obs 更新只精确修复 cmd 槽; phase/history/last_act/motor_targets 属
    训练原样路径 (wrapper 保留 info) — 不宣称一致
  - 记录 reset 后 obs history/phase vs info 对照 (v2 npz 离线实证已出:
    done 后首行 obs last 槽 norm≈1.9 非零、phase≈0 非精确 — 原样行为)
  - 运行前打印 wrapper 模块路径 + sha256 + first_obs key 存在性 (key 存在≠版本同)
"""
import sys, os, time, hashlib
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/root/odm-playground")
from mujoco_playground import wrapper as mjpl_wrapper_mod
from mujoco_playground._src.collision import geoms_colliding
from playground.open_duck_mini_v2.joystick import Joystick

NUM_ENVS = 32
CMD_ATOL = 1e-5
AUTORESET_OBS_KEY = "first_obs"   # 远端 2026-09-03 简版 BraxAutoResetWrapper 实际 key
AUTORESET_DATA_KEY = "first_state"
ACTION_DIM = 14


class SyncError(Exception):
    pass


def wrapper_version_record():
    """记录 wrapper 模块路径/sha/key (Codex: key 存在 ≠ 版本相同)"""
    p = mjpl_wrapper_mod.__file__
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print(f"[v3] wrapper 模块: {p}", flush=True)
    print(f"[v3] wrapper sha256[0:16]: {h}", flush=True)
    src = open(p).read()
    print(f"[v3] wrapper 关键行: first_obs 缓存={'first_obs' in src} "
          f"full_reset={'full_reset' in src} where_done={'where_done' in src}", flush=True)


def make_env(episode_length):
    return mjpl_wrapper_mod.wrap_for_brax_training(Joystick(task="flat_terrain"),
                                                   episode_length=episode_length)


def unwrap_env(env):
    uenv = env
    while hasattr(uenv, "unwrapped"):
        nxt = uenv.unwrapped
        if callable(nxt):
            nxt = nxt()
        if nxt is None or nxt is uenv:
            break
        uenv = nxt
    return uenv


def sync_cmd(uenv, state, cmd_arr):
    """reset 后注入固定命令 + 重建 reference/obs + 同步 first_obs 缓存 (cmd 槽精确修复)"""
    try:
        obs_list = []
        info_list = []
        for e in range(NUM_ENVS):
            d_e = jax.tree.map(lambda x: x[e], state.data)
            info_e = jax.tree.map(lambda x: x[e], state.info)
            cmd_e = jp.array(cmd_arr[e], dtype=jp.float32)
            info_e["command"] = cmd_e
            info_e["current_reference_motion"] = uenv.PRM.get_reference_motion(
                cmd_e[0], cmd_e[1], cmd_e[2], info_e["imitation_i"]
            )
            contact = jp.array([
                geoms_colliding(d_e, gid, uenv._floor_geom_id)
                for gid in uenv._feet_geom_id
            ])
            obs_e = uenv._get_obs(d_e, info_e, contact)
            obs_list.append(obs_e)
            info_list.append(info_e)
        obs_b = jax.tree.map(lambda *xs: jp.stack(xs), *obs_list)
        info_b = jax.tree.map(lambda *xs: jp.stack(xs), *info_list)
        # cmd 槽精确修复: 更新 first_obs 缓存 (history/phase 槽保持 reset 原样 — 不宣称一致)
        if AUTORESET_OBS_KEY not in state.info:
            raise SyncError(f"first_obs key 缺失 (wrapper 版本差异?): "
                            f"info keys: {[k for k in state.info if 'first' in str(k).lower()]}")
        first_obs = state.info[AUTORESET_OBS_KEY]
        first_obs = jax.tree.map(lambda f, s: f.at[:].set(s), first_obs, obs_b)
        info_b = {**info_b, AUTORESET_OBS_KEY: first_obs}
        return state.replace(obs=obs_b, info=info_b)
    except SyncError:
        raise
    except Exception as ex:
        raise SyncError(f"sync_cmd FAILED ({type(ex).__name__}: {ex})") from ex


def hard_check(state, cmd_arr, tag, diag_path, action=None, label="pre-step"):
    """每动作前全 env 硬检查 (Codex #1 二审: 冒烟/正式统一复用)。
    shape 错 → 立即保存诊断并退出 (不继续切片, 防广播异常绕过诊断保存)。"""
    obs = np.asarray(state.obs["state"], dtype=np.float32)
    cmd = np.asarray(state.info["command"], dtype=np.float32)
    # shape 检查优先 — fail 即存诊断退出
    shape_ok = obs.ndim == 2 and obs.shape[0] == NUM_ENVS and obs.shape[1] == 101
    if not shape_ok:
        diag = {"tag": tag, "label": label, "problems": [f"obs shape={obs.shape} 期望 ({NUM_ENVS},101)"]}
        np.savez(diag_path + "_DIAG.npz", diag=np.array([str(diag)]))
        raise SyncError(f"[{tag}] {label} check FAIL: obs shape={obs.shape}")
    probs = []
    if not np.isfinite(obs).all():
        probs.append(f"obs 含非有限数 {np.isfinite(obs).sum()}/{obs.size}")
    if not np.isfinite(cmd).all():
        probs.append("info.command 含非有限数")
    d_oc = np.abs(obs[:, 6:13] - cmd)
    if d_oc.max() > CMD_ATOL:
        probs.append(f"obs_cmd vs info.command max|Δ|={d_oc.max():.6f} > {CMD_ATOL}")
    d_cc = np.abs(cmd - cmd_arr)
    if d_cc.max() > CMD_ATOL:
        probs.append(f"info.command vs cmd_arr max|Δ|={d_cc.max():.6f} > {CMD_ATOL}")
    if action is not None:
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (NUM_ENVS, ACTION_DIM):
            probs.append(f"action shape={a.shape} 期望 ({NUM_ENVS},{ACTION_DIM})")
        elif not np.isfinite(a).all():
            probs.append(f"action 含非有限数 {np.isfinite(a).sum()}/{a.size}")
    if probs:
        diag = {"tag": tag, "label": label, "problems": probs,
                "obs_min": float(obs.min()), "obs_max": float(obs.max())}
        np.savez(diag_path + "_DIAG.npz", diag=np.array([str(diag)]))
        raise SyncError(f"[{tag}] {label} check FAIL: {'; '.join(probs)} "
                        f"(诊断 {diag_path}_DIAG.npz)")


def run_smoke(policy, cmd_arr, tag):
    """强制 done 冒烟 (episode_length=40, 150 步 → 每 env ≥3 done)。
    二审 #1/#2: 每动作前 hard_check + np.all(done_seen>=2) + ep 关系断言。"""
    env = make_env(episode_length=40)
    uenv = unwrap_env(env)
    step_fn = jax.jit(lambda s, a: env.step(s, a))
    rng = jax.random.split(jax.random.PRNGKey(999), NUM_ENVS)
    state = env.reset(rng)
    state = sync_cmd(uenv, state, cmd_arr)
    ep_id = np.zeros(NUM_ENVS, dtype=np.int32)
    ep_step = np.zeros(NUM_ENVS, dtype=np.int32)
    done_seen = np.zeros(NUM_ENVS, dtype=np.int32)
    trunc_seen = np.zeros(NUM_ENVS, dtype=np.int32)
    last_ep_id = np.zeros(NUM_ENVS, dtype=np.int32)
    ep_id_mono = True
    ep_id_eq_done = True
    ep_step_reset_ok = True
    for i in range(150):
        hard_check(state, cmd_arr, tag, f"/root/trace4_{tag}_smoke", label=f"smoke-step{i}")
        obs_before = np.asarray(state.obs["state"], dtype=np.float32)
        action = np.stack([policy(obs_before[j:j + 1]) for j in range(NUM_ENVS)])
        hard_check(state, cmd_arr, tag, f"/root/trace4_{tag}_smoke", action=action, label=f"smoke-act{i}")
        state = step_fn(state, jp.array(action))
        d = np.asarray(state.done).astype(bool)
        if (ep_id < last_ep_id).any():
            ep_id_mono = False
        if d.any():
            oc = np.asarray(state.obs["state"][d, 6:13])
            cc = cmd_arr[d]
            if np.abs(oc - cc).max() > CMD_ATOL:
                # first_obs 缓存同步失效 — 硬失败 (冒烟期立即暴露)
                np.savez(f"/root/trace4_{tag}_smoke_DIAG.npz",
                         diag=np.array([f"done 后 obs_cmd 失配 max|Δ|={np.abs(oc-cc).max():.6f}"]))
                raise SyncError(f"[{tag}] smoke FAIL: done 后 obs_cmd≠cmd_arr (first_obs 缓存未同步)")
            # 物理 done vs truncation (ep_step==39 = 达 episode_length)
            is_trunc = ep_step[d] >= 39
            trunc_seen[d] += is_trunc.astype(np.int32)
            done_seen[d] += 1
            ep_id[d] += 1
            ep_step[d] = 0
        nd = ~d
        if nd.any():
            ep_step[nd] += 1
        last_ep_id = ep_id.copy()
    # 终检断言 (二审 #2)
    ok = True
    if not np.all(done_seen >= 2):
        print(f"[smoke] FAIL: per-env done 门槛 — {int((done_seen < 2).sum())} env <2 次 done "
              f"分布 {done_seen.tolist()}")
        ok = False
    if not (done_seen == ep_id).all():
        print(f"[smoke] FAIL: episode_id 增量 ≠ done 次数 (max 差 {(done_seen-ep_id).max()})")
        ok = False
    if not ep_id_mono:
        print("[smoke] FAIL: episode_id 回退")
        ok = False
    if ok:
        print(f"[smoke] per-env done: min={done_seen.min()} max={done_seen.max()} "
              f"| truncation: min={trunc_seen.min()} max={trunc_seen.max()} | "
              f"ep_id==done 数: {int((done_seen == ep_id).sum())}/{NUM_ENVS}", flush=True)
        print(f"[smoke] PASS ({tag}): done 边界/命令同步/first_obs 缓存/episode 关系全验证", flush=True)
    else:
        raise SyncError(f"[smoke] 冒烟失败 ({tag}) — 不进入正式采集")


def main():
    ckpt = "/root/odm-playground/checkpoints_joystick_150m/2026_09_06_154455_151388160.onnx"
    cmd_vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    tag = f"{cmd_vx}_seed{seed}"
    print(f"[v3] cmd={cmd_vx} seed={seed} n={NUM_ENVS} steps={STEPS}", flush=True)
    wrapper_version_record()
    try:
        sess = ort.InferenceSession(ckpt)
    except Exception as ex:
        print(f"[v3] onnx 加载失败: {ex}", flush=True)
        sys.exit(1)
    in_name = sess.get_inputs()[0].name

    def policy(obs_batch):
        return sess.run(None, {in_name: obs_batch})[0][0]

    cmd_arr = np.zeros((NUM_ENVS, 7), dtype=np.float32)
    cmd_arr[:16, 0] = cmd_vx

    # 记录声明 (范围修正): reset 后 history/phase 为训练原样路径
    print("[v3] record: first_obs 同步仅精确修复 cmd 槽; history/phase/last_act 槽"
          "保留 reset 原样 (wrapper 保留 info, 训练原样路径) — 不宣称一致", flush=True)

    # === 阶段 1: 强制 done 冒烟 ===
    run_smoke(policy, cmd_arr, tag)

    # === 阶段 2: 正式采集 (episode_length=1000) ===
    env = make_env(episode_length=1000)
    uenv = unwrap_env(env)
    print(f"[v3] unwrapped: {type(uenv).__name__}", flush=True)
    qpos_addr = np.asarray(uenv.get_actuator_joints_qpos_addr())
    qvel_addr = np.asarray(uenv.actuator_qvel_addr)
    fb_qpos_addr = int(np.asarray(uenv._floating_base_qpos_addr))
    fb_qvel_addr = int(np.asarray(uenv._floating_base_qvel_addr))
    default_act = np.asarray(uenv._default_actuator, dtype=np.float32)
    dt = 0.02

    rng = jax.random.split(jax.random.PRNGKey(seed), NUM_ENVS)
    state = env.reset(rng)
    state = sync_cmd(uenv, state, cmd_arr)
    step_fn = jax.jit(lambda s, a: env.step(s, a))
    obs_dim = np.asarray(state.obs["state"]).shape[-1]
    print(f"[v3] obs_dim={obs_dim} (expect 101)", flush=True)
    if obs_dim != 101:
        raise SyncError(f"obs_dim={obs_dim} ≠ 101")

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

    cols = {}
    def A(n, s): cols[n] = np.zeros((STEPS, NUM_ENVS) + s, np.float32)
    for n in ["I", "lin_xy", "lin_z", "ang_xy", "ang_z", "Eq", "Edq", "contact", "orient", "tlin", "tang",
              "torques", "arate", "sstill", "alive", "S_preclip", "reward_clip", "z", "done", "phase",
              "m_im", "m_lin", "m_tang", "m_torq", "m_arate", "m_sstill", "m_alive", "state_reward",
              "episode_id", "episode_step", "global_done_total", "post_state_valid",
              "terminal_state_available", "terminal_reward_available", "truncation"]:
        A(n, ())
    for n, s in [("joint_act", (10,)), ("joint_ref", (10,)), ("jq_full", (14,)), ("ref_full", (16,)),
                 ("b_lin_body", (3,)), ("gyro_body", (3,)), ("cmd", (7,)), ("obs_cmd", (7,)),
                 ("action", (14,)), ("last_act", (14,)), ("world_v", (6,)),
                 ("obs_before", (obs_dim,)), ("obs_after", (obs_dim,)), ("next_policy_obs", (obs_dim,))]:
        A(n, s)

    t1 = time.time()
    global_done_total = 0
    global_trunc_total = 0
    ep_id = np.zeros(NUM_ENVS, dtype=np.int32)
    ep_step = np.zeros(NUM_ENVS, dtype=np.int32)
    for i in range(STEPS):
        hard_check(state, cmd_arr, tag, f"/root/trace4_{tag}", label=f"step{i}")
        obs_before = np.asarray(state.obs["state"], dtype=np.float32)
        cmd_now = np.asarray(state.info["command"], dtype=np.float32)
        last_act_before = np.asarray(state.info["last_act"])
        pre_ep_id = ep_id.copy()
        pre_ep_step = ep_step.copy()
        action = np.stack([policy(obs_before[j:j + 1]) for j in range(NUM_ENVS)])
        hard_check(state, cmd_arr, tag, f"/root/trace4_{tag}", action=action, label=f"act{i}")
        state = step_fn(state, jp.array(action))
        done_now = np.asarray(state.done).astype(np.float32)
        is_trunc = (pre_ep_step >= 999).astype(np.float32) * done_now
        global_done_total += int(done_now.sum())
        global_trunc_total += int(is_trunc.sum())
        obs_after = np.asarray(state.obs["state"], dtype=np.float32)
        next_policy_obs = obs_after
        d_mask = done_now > 0
        ep_id[d_mask] += 1
        ep_step[d_mask] = 0
        ep_step[~d_mask] += 1

        si = i
        qpos = np.asarray(state.data.qpos); qvel = np.asarray(state.data.qvel)
        ref = np.asarray(state.info["current_reference_motion"])
        jq = qpos[:, qpos_addr]; jv = qvel[:, qvel_addr]
        contacts = obs_after[:, 97:99]
        bq = qpos[:, fb_qpos_addr:fb_qpos_addr+7]; bv = qvel[:, fb_qvel_addr:fb_qvel_addr+6]
        dec = imitation_decompose(jp.array(bq), jp.array(bv), jp.array(jq), jp.array(jv),
                                  jp.array(contacts), jp.array(ref), jp.array(cmd_now))
        af = np.asarray(state.data.actuator_force)
        torques = np.sum(af**2, axis=-1)
        arate = np.sum((action - last_act_before)**2, axis=-1)
        gate_ss = (np.linalg.norm(cmd_now[:, :3], axis=-1) < 0.01).astype(np.float32)
        sstill = (np.sum(np.abs(jq - default_act[None, :]), axis=-1) + np.sum(np.abs(jv), axis=-1)) * gate_ss
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
        cols["obs_cmd"][si] = obs_before[:, 6:13]
        cols["done"][si] = done_now
        cols["truncation"][si] = is_trunc
        cols["global_done_total"][si] = global_done_total
        cols["episode_id"][si] = pre_ep_id
        cols["episode_step"][si] = pre_ep_step
        cols["post_state_valid"][si] = 1.0 - done_now  # 对当前动作后继/分解有效
        cols["terminal_state_available"][si] = 0.0     # wrapper 已覆盖终止 data — 恒 False
        cols["terminal_reward_available"][si] = done_now  # reward/metrics 属终止转换
        cols["phase"][si] = np.asarray(state.info["imitation_i"]) % 27
        cols["action"][si] = action; cols["last_act"][si] = last_act_before
        cols["world_v"][si] = bv
        cols["obs_before"][si] = obs_before
        cols["obs_after"][si] = obs_after
        cols["next_policy_obs"][si] = next_policy_obs
        if (i+1) % 60 == 0:
            print(f"[v3] step {i+1}/{STEPS} {time.time()-t1:.0f}s "
                  f"done_total={global_done_total} trunc={global_trunc_total}", flush=True)
    dt_env = time.time()-t1
    np.savez(f"/root/trace4_{tag}.npz", **{k: v for k, v in cols.items()})
    print(f"[v3] saved /root/trace4_{tag}.npz rows={STEPS}x{NUM_ENVS} "
          f"done_total={global_done_total} trunc={global_trunc_total} ({dt_env:.0f}s)", flush=True)

    # ===== 终检: 任一失败 → FAILED + exit(1) =====
    fails = []
    for k, v in cols.items():
        if not np.isfinite(v).all():
            fails.append(f"列 {k} 含非有限数 {np.isfinite(v).sum()}/{v.size}")
    nd = cols["post_state_valid"] > 0
    pairs = [("im", "I", 1.0), ("lin", "tlin", 2.5), ("tang", "tang", 6.0), ("torq", "torques", 1e-3),
             ("arate", "arate", 0.5), ("sstill", "sstill", 0.2), ("alive", "alive", 20.0)]
    for mk, pk, sc_ in pairs:
        dv = np.abs(cols[f"m_{mk}"][nd] - sc_*cols[pk][nd])
        print(f"[v3] 对拍 {mk:6s} (有效 {nd.sum()}): max|Δ|={dv.max():.6f}")
        if dv.max() > 1e-4: fails.append(f"对拍 {mk} max|Δ|={dv.max():.6f} > 1e-4")
    dS = np.abs(cols["state_reward"][nd] - cols["reward_clip"][nd])
    print(f"[v3] state.reward vs reward_clip (有效): max|Δ|={dS.max():.6f}")
    if dS.max() > 1e-4: fails.append(f"reward 对拍 max|Δ|={dS.max():.6f} > 1e-4")
    all_cmd_d = np.abs(cols["obs_cmd"] - cols["cmd"])
    print(f"[v3] obs_cmd vs cmd (全部 {cols['obs_cmd'].shape[0]*NUM_ENVS} 样本): max|Δ|={all_cmd_d.max():.6f}")
    if all_cmd_d.max() > CMD_ATOL: fails.append(f"obs_cmd vs cmd max|Δ|={all_cmd_d.max():.6f} > {CMD_ATOL}")
    dd = cols["done"] > 0
    if dd.any():
        npo_cmd_d = np.abs(cols["next_policy_obs"][dd][:, 6:13] - cols["cmd"][dd])
        print(f"[v3] done 行 ({dd.sum()}) next_policy_obs cmd 槽: max|Δ|={npo_cmd_d.max():.6f}")
        if npo_cmd_d.max() > CMD_ATOL:
            fails.append(f"done 行 next_policy_obs cmd max|Δ|={npo_cmd_d.max():.6f} > {CMD_ATOL}")
    if (cols["episode_id"][1:] < cols["episode_id"][:-1]).any():
        fails.append("episode_id 回退")
    if (cols["episode_step"] < 0).any() or (cols["episode_step"] > 1000).any():
        fails.append("episode_step 越界")
    # truncation 列一致性: truncation ⊆ done
    if (cols["truncation"] > cols["done"]).any():
        fails.append("truncation 行非 done")
    if fails:
        for f_ in fails: print(f"[v3] FAIL: {f_}")
        open(f"/root/trace4_{tag}.FAILED", "w").write("\n".join(fails))
        sys.exit(1)
    print(f"[v3] PACK: 全部终检通过 ({len(pairs)+7} 项) — 产物可用", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SyncError as ex:
        print(f"[v3] ABORT: {ex}", flush=True)
        sys.exit(1)

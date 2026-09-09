#!/usr/bin/env python3
"""Microduck 策略编排器：多策略 session 管理 + 行为 DSL 序列执行

支持：walking / stand / sitstand(sit↔stand) / kick_left / kick_right / roulade / ground_pick
场景：scene_walk.xml（基础）、scene_ball.xml（踢球）、scene_rollers.xml（轮滑）

行为 DSL:
  seq = [
    {"policy":"stand", "dur":2.0},
    {"policy":"walk", "dur":4.0, "vel":(0.5,0,0)},
    {"policy":"sit", "dur":2.0},        # sitstand flag=1
    {"policy":"standup", "dur":2.5},    # sitstand flag=0
    {"policy":"kick", "side":"left", "dur":3.0},
    {"policy":"roll", "dur":2.0},
  ]
"""
import math
import numpy as np
import mujoco
import onnxruntime as ort

SIM_DIR = "/tmp/duck_sim"
CTRL_EVERY = 10  # 50Hz @ dt=0.002

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)
JOINT_NAMES = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
               "neck_pitch","head_pitch","head_yaw","head_roll",
               "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qv_mult(q, v):
    qw,qx,qy,qz = q; x,y,z = v
    ix = qw*x + qy*z - qz*y; iy = qw*y + qz*x - qx*z; iz = qw*z + qx*y - qy*x
    iw = -qx*x - qy*y - qz*z
    return np.array([ix*qw+iw*-qx+iy*-qz-iz*-qy, iy*qw+iw*-qy+iz*-qx-ix*-qz,
                     iz*qw+iw*-qz+ix*-qy-iy*-qx], dtype=np.float32)

class DuckOrchestrator:
    def __init__(self, scene="walk", load_policies=None, render=False):
        """scene: walk | ball | rollers
        load_policies: list of policies to load, e.g. ['walk','stand','sitstand','kick_left','roulade']"""
        scene_files = {"walk":"scene.xml", "ball":"scene_ball.xml", "rollers":"scene_rollers.xml",
                      "full":"scene.xml", "light":"scene_walk.xml", "ball2":"scene_ball2.xml"}  # light=碰撞不全版(仅调试)
        self.xml = f"{SIM_DIR}/{scene_files[scene]}"
        self.m = mujoco.MjModel.from_xml_path(self.xml)
        self.d = mujoco.MjData(self.m)
        self.qidx = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.vidx = np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        key_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, key_id)
        mujoco.mj_forward(self.m, self.d)
        self.gyro_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.ball_qpos_adr = None
        self.ball_qpos_adrs = []
        self.ball_qvel_adrs = []
        for i in range(self.m.njnt):
            jname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jname and "free" in jname and (jname.startswith("ball") or "ball" in jname):
                self.ball_qpos_adrs.append(self.m.jnt_qposadr[i])
                self.ball_qvel_adrs.append(self.m.jnt_dofadr[i])
        if self.ball_qpos_adrs:
            self.ball_qpos_adr = self.ball_qpos_adrs[0]
            self.ball_qvel_adr = self.ball_qvel_adrs[0]
        self.world_g = np.array([0,0,-1], dtype=np.float32)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.cmd13 = np.zeros(13, dtype=np.float32)
        self.head_offset = np.zeros(4, dtype=np.float32)  # [neck, head_pitch, head_yaw, head_roll]
        self.head_clip = np.array([1.1, 1.1, 1.4, 0.31], dtype=np.float32)
        # policy sessions
        self.sessions = {}
        self.current_policy = None
        self.current_session = None
        # state
        self.sit_mode = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.phase = 0.0
        self._load(load_policies or ["walk"])

    def _load(self, policies):
        name_map = {
            "walk":"alpha_walking", "stand":"alpha_stand", "sitstand":"alpha_sitstand",
            "ground_pick":"alpha_ground_pick", "kick_left":"ball_kick_left",
            "kick_right":"ball_kick_right", "roll":"roller", "roll_crouch":"roller_crouch",
            "roulade":"roulade",
        }
        for p in policies:
            key = p
            fn = name_map[p]
            path = f"{SIM_DIR}/{fn}.onnx"
            try:
                self.sessions[key] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"  [load] {p} failed: {e}")
        # default policy: stand if available else walk
        if "stand" in self.sessions:
            self._switch("stand")
        elif "sitstand" in self.sessions:
            self._switch("sitstand")
        else:
            self._switch("walk")

    def _switch(self, policy):
        if policy not in self.sessions:
            print(f"  [switch] {policy} 未加载，忽略")
            return
        self.current_policy = policy
        self.current_session = self.sessions[policy]
        # 跨策略切换必须清零 last_action（14D 动作历史会污染新策略的 obs）
        self.last_action = np.zeros(14, dtype=np.float32)
        self._gp_start_t = None  # ground_pick phase 锚点（策略切换时重置）
        # reset command per policy semantics
        if policy in ("kick_left","kick_right","roulade","ground_pick"):
            # 行为策略训练时全零命令（含 head 槽）——清 head_offset 防 OOD
            self.head_offset[:] = 0.0
        if policy == "sitstand":
            self.cmd13[:] = 0.0
            self.cmd13[0] = 1.0 if self.sit_mode else 0.0
        elif policy in ("walk","roll"):
            self.cmd13[:] = 0.0
            self.cmd13[0:3] = self.vel_cmd
        else:  # stand / kick / roulade / ground_pick: all-zero cmd
            self.cmd13[:] = 0.0
        self.phase = 0.0

    # ---- 行为命令 ----
    def cmd_walk(self, vel=(0.5,0,0), dur=3.0):
        self.vel_cmd = np.array(vel, dtype=np.float32)
        self.sit_mode = False
        self._switch("walk")
        return dur

    def cmd_stand(self, dur=2.0):
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.sit_mode = False
        self._switch("stand")
        return dur

    def cmd_sit(self, dur=2.5):
        """坐下。sitstand 需要先 flag=0 站稳再翻转 flag=1（直接从 stand 策略切换会 OOD）"""
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.current_policy != "sitstand":
            # 先切到 sitstand 并以 flag=0 站稳（过渡）
            self.sit_mode = False
            self._switch("sitstand")
            dt = self.m.opt.timestep
            for _ in range(int(0.8 / (dt * 10))):
                self.control_step()
                self.step(10)
        self.sit_mode = True
        self._update_cmd()
        return dur

    def cmd_standup(self, dur=3.0):
        self.sit_mode = False
        self._switch("sitstand")  # same session, flag flip → policy glides up
        return dur

    def cmd_kick(self, side="left", dur=3.0, place_ball=True):
        self.sit_mode = False
        self._switch(f"kick_{side}")
        if self.ball_qpos_adr is not None and place_ball:
            self._place_ball(side)
        return dur

    def cmd_roll(self, dur=4.0):
        self.sit_mode = False
        self._switch("roll")
        return dur

    def cmd_roulade(self, dur=2.0):
        self.sit_mode = False
        self._switch("roulade")
        return dur

    def _place_ball(self, side):
        adr = self.trunk_id
        x, y = float(self.d.qpos[0]), float(self.d.qpos[1])
        qw, qx, qy, qz = self.d.qpos[3:7]
        yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        off_y = -BALL_OFFSET_ABS_Y if side == "right" else BALL_OFFSET_ABS_Y
        bx = x + math.cos(yaw)*BALL_OFFSET_X - math.sin(yaw)*off_y
        by = y + math.sin(yaw)*BALL_OFFSET_X + math.cos(yaw)*off_y
        self.d.qpos[self.ball_qpos_adr:self.ball_qpos_adr+7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        if self.ball_qvel_adr is not None:
            self.d.qvel[self.ball_qvel_adr:self.ball_qvel_adr+6] = 0.0

    # ---- 步进 ----
    def set_head(self, neck=0.0, pitch=0.0, yaw=0.0, roll=0.0, dur=None):
        """设置头部姿态命令（stand/sitstand 策略下驱动头部，无需重训）
        head 槽位只在非行为策略生效；返回建议持续时间或 None"""
        self.head_offset = np.clip(
            np.array([neck, pitch, yaw, roll], dtype=np.float32), -self.head_clip, self.head_clip)
        return dur

    def _update_cmd(self):
        p = self.current_policy
        if p == "sitstand":
            self.cmd13[:] = 0.0
            self.cmd13[0] = 1.0 if self.sit_mode else 0.0
            self.cmd13[3:7] = self.head_offset  # sitstand 也接受 head 命令
        elif p in ("walk","roll"):
            self.cmd13[:] = 0.0
            self.cmd13[0:3] = self.vel_cmd
        elif p == "ground_pick":
            # phase 编码命令（RL 源码 GroundPickPhaseCommand）：
            # phase∈[0,0.5] 下探、[0.5,1.0] 回升，周期 4s
            # 相位随物理时间推进（t/4 mod 1），不是静止 phase=0！
            if self._gp_start_t is None:
                self._gp_start_t = self._t()
            self.phase = ((self._t() - self._gp_start_t) / 4.0) % 1.0
            self.cmd13[0] = math.cos(2*math.pi*self.phase)
            self.cmd13[1] = math.sin(2*math.pi*self.phase)
            self.cmd13[2] = 0.0
        elif p == "stand":
            # stand 支持 head/body 槽位：保持 head_offset 持续生效
            self.cmd13[:] = 0.0
            self.cmd13[3:7] = self.head_offset
        else:
            self.cmd13[:] = 0.0

    def step(self, n_physics=1):
        for _ in range(n_physics):
            self.d.ctrl[:] = (DEFAULT_POSE + self.last_action).astype(np.float32)
            mujoco.mj_step(self.m, self.d)

    def control_step(self):
        """One 50Hz control tick: build obs → infer → apply action"""
        self._update_cmd()
        ang_vel = self.d.sensordata[self.m.sensor_adr[self.gyro_id]:self.m.sensor_adr[self.gyro_id]+3].copy().astype(np.float32)
        quat = self.d.xquat[self.trunk_id].copy().astype(np.float32)
        gravity = qv_mult(qconj(quat), self.world_g).astype(np.float32)
        jpos_rel = (self.d.qpos[self.qidx] - DEFAULT_POSE).astype(np.float32)
        jvel = self.d.qvel[self.vidx].copy().astype(np.float32)
        obs = np.concatenate([ang_vel, gravity, jpos_rel, jvel, self.last_action, self.cmd13]).astype(np.float32)
        out = self.current_session.run(None, {self.current_session.get_inputs()[0].name: obs.reshape(1,61)})[0]
        self.last_action = out[0].astype(np.float32)

    def run_sequence(self, seq, render_every=0, render_cb=None, verbose=True):
        """执行行为序列。seq: list of {policy/dur/...} dicts
        render_every>0 时每 N 物理步调 render_cb(d)"""
        dt = self.m.opt.timestep
        total_ctrl = 0
        events = []
        fallen_at = None
        for step_spec in seq:
            policy = step_spec.get("policy")
            dur = step_spec.get("dur", 2.0)
            # dispatch
            if policy == "walk": self.cmd_walk(step_spec.get("vel",(0.5,0,0)), dur)
            elif policy == "stand": self.cmd_stand(dur)
            elif policy == "sit": self.cmd_sit(dur)
            elif policy == "standup": self.cmd_standup(dur)
            elif policy == "kick": self.cmd_kick(step_spec.get("side","left"), dur)
            elif policy == "roll": self.cmd_roll(dur)
            elif policy == "roulade": self.cmd_roulade(dur)
            elif policy == "ground_pick": self.cmd_ground_pick(dur)
            elif policy == "head":
                # 头部动作：若当前是坐姿(sitstand+sit_mode)则留在 sitstand，否则切 stand
                if self.current_policy not in ("stand", "sitstand"):
                    if "stand" in self.sessions:
                        self.cmd_stand(dur)
                self.set_head(
                    neck=step_spec.get("neck", 0.0),
                    pitch=step_spec.get("pitch", 0.0),
                    yaw=step_spec.get("yaw", 0.0),
                    roll=step_spec.get("roll", 0.0))
            else:
                print(f"  [seq] 未知行为: {policy}")
                continue
            if verbose:
                print(f"  [{self._t():.1f}s] → {policy} {dur:.1f}s (vel={self.vel_cmd[:2]})")
            events.append((self._t(), policy))
            n_ctrl = int(dur / (dt * CTRL_EVERY))
            step_ctrl = 0
            for _ in range(n_ctrl):
                self.control_step()
                self.step(CTRL_EVERY)
                step_ctrl += 1
                total_ctrl += 1
                if render_every and total_ctrl % render_every == 0 and render_cb:
                    render_cb(self)
                if self.d.qpos[2] < 0.04:
                    fallen_at = self._t()
                    if verbose: print(f"  ❌ 摔倒 at t={fallen_at:.1f}s (policy={policy})")
                    break
            if fallen_at: break
            if verbose: print(f"      ✓ {policy} 完成 (z={self.d.qpos[2]:.3f})")
        return {"ok": fallen_at is None, "fallen_at": fallen_at, "events": events,
                "dist": self.d.qpos[0], "z": self.d.qpos[2]}

    def _t(self):
        return float(self.d.time)

    # ground_pick 需要完整 phase 周期（4s），简化处理
    def cmd_ground_pick(self, dur=4.0):
        self.sit_mode = False
        self._switch("ground_pick")
        return dur

    def _gp_phase_now(self):
        """当前 ground_pick phase（0-1），供外部验证"""
        if self.current_policy != "ground_pick":
            return None
        if self._gp_start_t is None:
            self._gp_start_t = self._t()
        return ((self._t() - self._gp_start_t) / 4.0) % 1.0

# ---- 便捷执行 ----
def run_sequence_demo():
    orch = DuckOrchestrator(scene="ball", load_policies=["walk","stand","sitstand","kick_left","roulade"])
    seq = [
        {"policy":"stand", "dur":1.5},
        {"policy":"walk", "vel":(0.5,0,0), "dur":3.0},
        {"policy":"sit", "dur":2.5},
        {"policy":"standup", "dur":3.0},
        {"policy":"walk", "vel":(0.4,0,0), "dur":2.0},
        {"policy":"kick", "side":"left", "dur":3.0},
        {"policy":"stand", "dur":1.0},
    ]
    return orch.run_sequence(seq)

if __name__ == "__main__":
    r = run_sequence_demo()
    print(f"\n编排结果: {'✅ 全程无摔倒' if r['ok'] else '❌ 摔倒 t=' + str(r['fallen_at'])} | 位移={r['dist']:.2f}m | 终z={r['z']:.3f}")

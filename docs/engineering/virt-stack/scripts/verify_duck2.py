#!/usr/bin/env python3
"""Microduck 虚拟鸭子验证 v2：50Hz 控制频率匹配 + 动作延迟缓冲"""
import numpy as np
import mujoco
import onnxruntime as ort

SIM_DIR = "/tmp/duck_sim"
XML = f"{SIM_DIR}/scene_walk.xml"
CTRL_DT = 0.02   # 50Hz policy control
SIM_DT = 0.002   # mujoco timestep
CTRL_EVERY = int(CTRL_DT / SIM_DT)  # 10 physics steps per policy step

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

JOINT_NAMES = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
               "neck_pitch","head_pitch","head_yaw","head_roll",
               "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])
def qv_mult(q, v):
    qw, qx, qy, qz = q
    x, y, z = v
    ix = qw*x + qy*z - qz*y
    iy = qw*y + qz*x - qx*z
    iz = qw*z + qx*y - qy*x
    iw = -qx*x - qy*y - qz*z
    return np.array([ix*qw + iw*-qx + iy*-qz - iz*-qy,
                     iy*qw + iw*-qy + iz*-qx - ix*-qz,
                     iz*qw + iw*-qz + ix*-qy - iy*-qx], dtype=np.float32)

def run_policy(onnx_path, seconds=6.0, vel_cmd=(0.2,0,0), delay=True):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    qidx = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    vidx = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)

    gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    last_action = np.zeros(14, dtype=np.float32)
    # action delay buffer (3-6 steps like training)
    buf = [np.zeros(14, dtype=np.float32) for _ in range(6)]
    bi = 0
    lag = 4
    world_g = np.array([0,0,-1], dtype=np.float32)
    vcmd = np.array(vel_cmd, dtype=np.float32)
    cmd13 = np.zeros(13, dtype=np.float32)
    cmd13[0:3] = vcmd

    start_x = d.qpos[0]
    fallen = False
    min_z = 9.9
    n_ctrl = 0
    total_steps = int(seconds / SIM_DT)
    xs = []

    for i in range(total_steps):
        if i % CTRL_EVERY == 0:
            ang_vel = d.sensordata[m.sensor_adr[gyro_id]:m.sensor_adr[gyro_id]+3].copy().astype(np.float32)
            quat = d.xquat[trunk_id].copy().astype(np.float32)
            gravity = qv_mult(qconj(quat), world_g).astype(np.float32)
            jpos_rel = (d.qpos[qidx] - DEFAULT_POSE).astype(np.float32)
            jvel = d.qvel[vidx].copy().astype(np.float32)
            obs = np.concatenate([ang_vel, gravity, jpos_rel, jvel, last_action, cmd13]).astype(np.float32)
            out = sess.run(None, {sess.get_inputs()[0].name: obs.reshape(1,61)})[0]
            action = out[0].astype(np.float32)
            # delayed action application
            if delay:
                buf[bi % len(buf)] = action
                applied = buf[(bi - lag) % len(buf)]
                bi += 1
            else:
                applied = action
            d.ctrl[:] = (DEFAULT_POSE + applied).astype(np.float32)
            last_action = action
            n_ctrl += 1
        mujoco.mj_step(m, d)
        if i % 500 == 0:
            z = d.qpos[2]
            min_z = min(min_z, z)
            if z < 0.03:
                fallen = True
                break
        xs.append(d.qpos[0])

    dist = d.qpos[0] - start_x
    return {"survived": not fallen, "min_z": min_z, "dist": dist,
            "final_z": d.qpos[2], "n_ctrl": n_ctrl, "final_x": d.qpos[0]}

if __name__ == "__main__":
    print("=== alpha_walking @50Hz, vel 0.2m/s, 6s ===")
    r = run_policy(f"{SIM_DIR}/alpha_walking.onnx", seconds=6.0, vel_cmd=(0.2,0,0))
    print(f"  存活:{r['survived']} min_z={r['min_z']:.3f} 位移={r['dist']:.3f}m ({r['n_ctrl']}次策略推理) 终z={r['final_z']:.3f}")
    print("=== alpha_stand @50Hz, zero cmd, 4s ===")
    r2 = run_policy(f"{SIM_DIR}/alpha_stand.onnx", seconds=4.0, vel_cmd=(0,0,0))
    print(f"  存活:{r2['survived']} min_z={r2['min_z']:.3f} 位移={r2['dist']:.3f}m")
    print("=== alpha_sitstand @50Hz ===")
    r3 = run_policy(f"{SIM_DIR}/alpha_sitstand.onnx", seconds=4.0, vel_cmd=(0,0,0))
    print(f"  存活:{r3['survived']} min_z={r3['min_z']:.3f} 位移={r3['dist']:.3f}m")
    ok = r['survived'] and r2['survived'] and r3['survived']
    print(f"\n结论: {'✅ 虚拟鸭子环境完全跑通' if ok else '⚠️ 部分通过'} walking位移={r['dist']:.2f}m/6s")

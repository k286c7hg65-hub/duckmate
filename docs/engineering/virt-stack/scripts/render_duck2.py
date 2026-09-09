#!/usr/bin/env python3
"""Microduck 行走视频渲染 v2 (EGL surfaceless)"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import numpy as np
import mujoco
import onnxruntime as ort
import subprocess, sys
from PIL import Image

SIM_DIR = "/tmp/duck_sim"
XML = f"{SIM_DIR}/scene_walk.xml"
CTRL_EVERY = 10
RENDER_EVERY = 5  # 每 5 物理步渲染 → 500Hz/5 = 100fps 模拟采样 → 25fps 播放(4x慢放更清晰)

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)
JOINT_NAMES = ["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
               "neck_pitch","head_pitch","head_yaw","head_roll",
               "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]

def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qv_mult(q, v):
    qw,qx,qy,qz = q; x,y,z = v
    ix = qw*x + qy*z - qz*y; iy = qw*y + qz*x - qx*z; iz = qw*z + qx*y - qy*x
    iw = -qx*x - qy*y - qz*z
    return np.array([ix*qw+iw*-qx+iy*-qz-iz*-qy, iy*qw+iw*-qy+iz*-qx-ix*-qz,
                     iz*qw+iw*-qz+ix*-qy-iy*-qx], dtype=np.float32)

vel = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
qidx = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
vidx = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
mujoco.mj_resetDataKeyframe(m, d, key_id)
mujoco.mj_forward(m, d)

gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
sess = ort.InferenceSession(f"{SIM_DIR}/alpha_walking.onnx", providers=["CPUExecutionProvider"])

last_action = np.zeros(14, dtype=np.float32)
world_g = np.array([0,0,-1], dtype=np.float32)
cmd13 = np.zeros(13, dtype=np.float32)
cmd13[0:3] = (vel, 0, 0)

renderer = mujoco.Renderer(m, 720, 1280)
import shutil
shutil.rmtree("/tmp/duck_frames", ignore_errors=True)
os.makedirs("/tmp/duck_frames")

total_steps = int(duration / m.opt.timestep)
start_x = d.qpos[0]
frame_n = 0

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = trunk_id
cam.distance = 1.0
cam.azimuth = 110
cam.elevation = -25

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
        d.ctrl[:] = (DEFAULT_POSE + action).astype(np.float32)
        last_action = action
    mujoco.mj_step(m, d)
    if i % RENDER_EVERY == 0:
        renderer.update_scene(d, camera=cam)
        pixels = renderer.render()
        Image.fromarray(pixels).save(f"/tmp/duck_frames/f_{frame_n:05d}.png")
        frame_n += 1
    if d.qpos[2] < 0.03:
        print(f"摔倒 at step {i}")
        break

dist = d.qpos[0] - start_x
elapsed = i * m.opt.timestep
print(f"帧数:{frame_n} 位移:{dist:.3f}m 用时:{elapsed:.1f}s 均速:{dist/elapsed:.3f}m/s")

out = "/tmp/microduck_walk.mp4"
cmd = ["ffmpeg", "-y", "-framerate", "25", "-i", "/tmp/duck_frames/f_%05d.png",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
r = subprocess.run(cmd, capture_output=True, text=True)
print("ffmpeg:", "OK" if r.returncode == 0 else r.stderr[-300:])

#!/usr/bin/env python3
"""渲染长程规划演示视频：greet + 双球连续踢飞（LLM 规划闭环全程）"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import sys, shutil, subprocess
sys.path.insert(0, '/tmp')
import numpy as np
import mujoco
from PIL import Image

FRAMES = "/tmp/duck_planner_frames"
shutil.rmtree(FRAMES, ignore_errors=True)
os.makedirs(FRAMES)

from duck_orchestrator import DuckOrchestrator
import duck_planner as dp

orch = DuckOrchestrator(scene="ball2", load_policies=["walk", "stand", "kick_left", "kick_right"])
m, d = orch.m, orch.d
dt = m.opt.timestep
a0, a1 = orch.ball_qpos_adrs
d.qpos[a0:a0+3] = [0.55, 0.18, 0.035]
d.qpos[a1:a1+3] = [0.85, -0.2, 0.035]
mujoco.mj_forward(m, d)
orch.cmd_stand(0.5)
for _ in range(int(0.5/(dt*10))): orch.control_step(); orch.step(10)
orch._ball0 = dp._ball_positions(orch)
orch._kicked_idx = set()

renderer = mujoco.Renderer(m, 360, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = orch.trunk_id
cam.distance = 1.1
cam.azimuth = 100
cam.elevation = -20
RENDER_EVERY = 10
frame_n = 0

def snap():
    global frame_n
    renderer.update_scene(d, camera=cam)
    Image.fromarray(renderer.render()).save(f"{FRAMES}/f_{frame_n:05d}.png")
    frame_n += 1

def run(dur):
    global frame_n
    n = int(dur / (dt * 10))
    for i in range(n):
        orch.control_step()
        for s in range(10):
            orch.step(1)
            if (i*10 + s) % RENDER_EVERY == 0:
                snap()
        if d.qpos[2] < 0.04:
            print(f"❌ 摔倒 t={orch._t():.1f}"); return False
    return True

print("=== 阶段1: greet 技能 ===")
from duck_skills import load_skill, skill_to_steps
skill = load_skill("greet")
for step in skill_to_steps(skill):
    p = step["policy"]
    if p == "head":
        orch.set_head(pitch=step.get("pitch",0), yaw=step.get("yaw",0))
        run(step["dur"])
orch.set_head()
print(f"    greet 完成 z={d.qpos[2]:.3f}")

print("=== 阶段2: 长程双球 ===")
# 逐球用 _kick_cycle（会自己逼近+冲撞）
for target in [0, 1]:
    if target in orch._kicked_idx: continue
    print(f"  目标球 #{target+1}...")
    hit, disp = dp._kick_cycle(orch, target, verbose=False)
    print(f"    结果: hit={hit} disp={disp:.2f}")
    orch._kicked_idx.add(target)
    orch.cmd_stand(0.8)
    run(0.8)

print(f"总帧: {frame_n}")
out = "/tmp/microduck_planner_demo.mp4"
r = subprocess.run(["ffmpeg","-y","-framerate","25","-i",f"{FRAMES}/f_%05d.png",
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","20",out],
                   capture_output=True, text=True)
print("ffmpeg:", "OK" if r.returncode==0 else r.stderr[-300:])

#!/usr/bin/env python3
"""Duck Vision: 视觉闭环——鸭子"看"画面判断球方位 → 决策转向/前进 → 追球

闭环：渲染场景 → DeepSeek vision 定位球 → 决策(左/右/前/近) → walk 斜向/直行/停踢 → 再看
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
import sys
import time
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, "/tmp")

def render_scene(orch, size=(300, 400), path="/tmp/duck_eye_view.png"):
    """渲染跟踪视角场景图（模拟鸭子感知）"""
    r = mujoco.Renderer(orch.m, size[0], size[1])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = orch.trunk_id
    cam.distance = 0.9
    cam.azimuth = 90  # 侧视角，能看出球在鸭子前/后、左/右
    cam.elevation = -12
    r.update_scene(orch.d, camera=cam)
    img = r.render()
    Image.fromarray(img).save(path)
    return path

def vision_locate_ball(img_path, key_holder=None):
    """DeepSeek vision 判断球相对鸭子方位"""
    sys.path.insert(0, "<openclaw-workspace>/tools")
    from vision_direct import analyze
    prompt = ("白色小机器鸭站在场地中，前方可能有橙色小球。判断球相对鸭子的方位："
              "1) 球在鸭子 左侧/右侧/正前方/看不到 哪个？"
              "2) 球离鸭子 很近(<0.2m)/中等(0.2-0.5m)/远(>0.5m)/无球 哪个？"
              "只回答格式：side=X,dist=Y，X∈{left,right,front,none}，Y∈{near,mid,far,none}，不要其他文字")
    try:
        out = analyze(img_path, prompt)
        out = out.strip().lower()
        side, dist = "front", "far"
        for token in out.split(","):
            t = token.strip()
            if t.startswith("side="):
                v = t.split("=")[1].strip()
                if v in ("left", "right", "front", "none"):
                    side = v
            elif t.startswith("dist"):
                v = t.split("=")[1].strip().rstrip(".")
                if v in ("near", "mid", "far", "none"):
                    dist = v
        # 兜底关键词
        if side == "front":
            for k, v in [("左侧", "left"), ("右边", "right"), ("右侧", "right"), ("左边", "left"), ("正前", "front")]:
                if k in out:
                    side = v
        return side, dist, out
    except Exception as e:
        return "front", "far", f"vision_error:{e}"

def ground_truth_ball_dir(orch):
    """仿真真值（对照用）：球相对鸭子的方位"""
    bx = orch.d.qpos[orch.ball_qpos_adr]
    by = orch.d.qpos[orch.ball_qpos_adr + 1]
    dx = bx - orch.d.qpos[0]
    dy = by - orch.d.qpos[1]
    # 鸭子朝 +x 方向（初始 yaw=0）
    if dx > 0.15:
        return "front", np.hypot(dx, dy)
    elif dx < -0.15:
        return "back", np.hypot(dx, dy)
    else:
        return ("right" if dy > 0 else "left"), np.hypot(dx, dy)

def decide_action(side, dist, verbose=True):
    """视觉决策 → walk 命令"""
    if side == "none":
        return ("walk", (0.3, 0, 0), 1.0), "没看到球，前进搜索"
    if side == "back":
        # 球在背后：大转弯掉头（左右弧线）
        return ("walk", (0.2, 0, 1.2), 1.5), "球在背后，掉头"
    if side == "left":
        if dist == "near":
            return ("walk", (0.15, 0, -0.4), 0.8), "球在左近，微调左转"
        return ("walk", (0.25, 0, -0.6), 1.2), "球在左，左弧逼近"
    if side == "right":
        if dist == "near":
            return ("walk", (0.15, 0, 0.4), 0.8), "球在右近，微调右转"
        return ("walk", (0.25, 0, 0.6), 1.2), "球在右，右弧逼近"
    # front
    if dist == "near":
        return ("kick", "left", 2.0), "球在前方很近，踢！"
    if dist == "mid":
        return ("walk", (0.25, 0, 0), 0.8), "球在前中距，慢步逼近"
    return ("walk", (0.3, 0, 0), 1.2), "球在前方，直行逼近"

def run_vision_chase(max_rounds=8, use_vision=True, ball_pos=None, seed=None):
    """视觉闭环追球主循环"""
    from duck_orchestrator import DuckOrchestrator
    orch = DuckOrchestrator(scene="ball", load_policies=["walk", "stand", "kick_left", "kick_right"])
    # 随机放球（右前/左前/正前）
    import random
    if seed is not None:
        random.seed(seed)
    placements = [(0.55, 0.20), (0.55, -0.20), (0.5, 0.0), (0.6, 0.15)]
    if ball_pos:
        px, py = ball_pos
    else:
        px, py = random.choice(placements)
    orch.d.qpos[orch.ball_qpos_adr:orch.ball_qpos_adr+3] = [px, py, 0.035]
    mujoco.mj_forward(orch.m, orch.d)
    # 站稳
    orch.cmd_stand(0.5)
    dt = orch.m.opt.timestep
    for _ in range(int(0.5 / (dt * 10))):
        orch.control_step(); orch.step(10)
    print(f"🎯 球初始位置: ({px}, {py}) | 鸭子: (0, 0) 朝 +x")

    history = []
    gt0 = ground_truth_ball_dir(orch)
    print(f"真值: 球在 {gt0[0]} 距离 {gt0[1]:.2f}m")

    for rnd in range(1, max_rounds + 1):
        # 1. 看
        img_path = render_scene(orch)
        if use_vision:
            side, dist, raw = vision_locate_ball(img_path)
        else:
            side, dist = ground_truth_ball_dir(orch)[0], "mid"
            if dist == "front" and gt0[1] < 0.2: dist = "near"
        print(f"\n[轮 {rnd}] 👁️ 视觉判断: side={side}, dist={dist}"
              + ("" if use_vision else " (真值)"))
        # 2. 决策
        action, why = decide_action(side, dist)
        print(f"   🧠 决策: {why} → {action}")
        # 3. 执行
        if action[0] == "walk":
            orch.cmd_walk(action[1], action[2])
        elif action[0] == "kick":
            orch.cmd_kick(action[1], 2.0)
        n_ctrl = int(action[2] / (dt * 10))
        for _ in range(n_ctrl):
            orch.control_step(); orch.step(10)
            if orch.d.qpos[2] < 0.04:
                print("   ❌ 摔倒"); return {"ok": False, "rounds": rnd}
        # 4. 检查球距离（放宽：侧向逼近到足够近也踢）
        bdx = orch.d.qpos[orch.ball_qpos_adr] - orch.d.qpos[0]
        bdy = orch.d.qpos[orch.ball_qpos_adr+1] - orch.d.qpos[1]
        bd = np.hypot(bdx, bdy)
        gts, gtd = ground_truth_ball_dir(orch)
        history.append((rnd, side, dist, gts, round(gtd, 3)))
        print(f"   真值: 球 {gts} 距离 {gtd:.2f}m | 鸭子位置 ({orch.d.qpos[0]:.2f},{orch.d.qpos[1]:.2f})")
        if bd < 0.28:
            print(f"\n🎉 鸭子接近球了！(距离 {bd:.2f}m)")
            # 踢侧按球相对方位：球在鸭子右侧(y+)用 left 脚？直接试 left 优先
            kick_side = "right" if bdy < -0.03 else "left"
            orch.cmd_kick(kick_side, 2.0)
            for _ in range(int(2.0 / (dt * 10))):
                orch.control_step(); orch.step(10)
            bd2 = np.hypot(orch.d.qpos[orch.ball_qpos_adr] - orch.d.qpos[0],
                           orch.d.qpos[orch.ball_qpos_adr+1] - orch.d.qpos[1])
            ball_disp = np.hypot(orch.d.qpos[orch.ball_qpos_adr] - px,
                                 orch.d.qpos[orch.ball_qpos_adr+1] - py)
            print(f"⚽ 踢出({kick_side})！球位移 {ball_disp:.2f}m")
            return {"ok": ball_disp > 0.1, "rounds": rnd, "history": history, "ball_moved": ball_disp}
    print("\n⏱️ 轮次用完，未接近球")
    return {"ok": False, "rounds": max_rounds, "history": history}

if __name__ == "__main__":
    use_v = "--real-vision" in sys.argv
    r = run_vision_chase(max_rounds=8, use_vision=use_v)
    print(f"\n结果: {'✅ 追到球' if r['ok'] else '❌ 未追到'} | 轮次 {r['rounds']}")
    if r.get("history"):
        print("过程:", r["history"])

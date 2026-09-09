#!/usr/bin/env python3
"""Duck Planner v2: LLM 长程规划层（认知层）——多球连续任务

任务文本 → LLM 分解子目标 → 逐目标执行（视觉/真值闭环踢球）→ 汇报
子目标词表: kick_ball / kick_all_balls / greet / look_around
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/tmp")
sys.path.insert(0, "<openclaw-workspace>/projects/microduck-virt/scripts")

try:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.path.insert(0, "<openclaw-workspace>/tools")
        from vision_ds import get_api_key as _ds_key
        _k = _ds_key()
        if _k:
            os.environ["DEEPSEEK_API_KEY"] = _k
except Exception:
    pass
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

GOAL_VOCAB = {
    "kick_ball": "找到场上一个球并踢飞它（视觉定位→逼近→踢）。",
    "kick_all_balls": "找到场上所有球，逐个踢飞（每踢完一个重新定位下一个）。",
    "greet": "打招呼（点头+张望）。",
    "look_around": "环顾四周。",
}

def llm_decompose(task_text, max_goals=5):
    """任务文本 → 子目标 JSON 列表"""
    import urllib.request
    url = "https://api.deepseek.com/chat/completions"
    vocab = "\n".join(f"- {k}: {v}" for k, v in GOAL_VOCAB.items())
    system = f"""你是机器人任务规划器。把用户的长任务分解为子目标序列。
可用子目标词表：
{vocab}

规则：
1. 只输出 JSON 数组，例如 ["kick_all_balls"] 或 ["greet","kick_ball"]
2. 子目标 ≤ {max_goals} 个，按执行顺序排列
3. 无法用现有词表表达的，用最接近的子目标近似并加 "note"
4. 无动作意图输出 []
5. 只输出 JSON，无解释无 markdown"""
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": task_text},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
    })
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode())
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        raise ValueError(f"LLM 未返回 JSON: {content[:200]}")
    goals = json.loads(m.group(0))
    out = []
    for g in goals:
        if isinstance(g, str):
            out.append({"goal": g, "note": None})
        elif isinstance(g, dict):
            out.append({"goal": g.get("goal", ""), "note": g.get("note")})
    return out

def _ball_positions(orch):
    """读取所有球的位置 [(x,y,z),...]"""
    out = []
    for adr in orch.ball_qpos_adrs:
        out.append((float(orch.d.qpos[adr]), float(orch.d.qpos[adr+1]), float(orch.d.qpos[adr+2])))
    return out

def _get_yaw(orch):
    """读取鸭子全局朝向 yaw (rad)"""
    import numpy as np
    w, x, y, z = orch.d.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _turn_to(orch, target_yaw, tol=0.10, max_steps=8, verbose=True):
    """用 wz 槽转向闭环：转到目标朝向 ±tol"""
    import numpy as np
    m, d = orch.m, orch.d
    dt = m.opt.timestep
    for s in range(max_steps):
        err = target_yaw - _get_yaw(orch)
        # 归一化到 [-pi, pi]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) < tol:
            return True
        wz = float(np.clip(err * 1.8, -0.8, 0.8))
        orch.cmd_walk((0.0, 0, wz), 0.5)
        for _ in range(int(0.5 / (dt * 10))):
            orch.control_step(); orch.step(10)
            if d.qpos[2] < 0.04:
                return False
    return abs(((target_yaw - _get_yaw(orch)) + np.pi) % (2 * np.pi) - np.pi) < 0.2


def _kick_cycle(orch, target_i, verbose=True):
    """转向对齐→连续短步逼近→站定→踢。返回 (踢中?, 球位移)

    【Preflight · 抄 EmbodiedSkills】执行前提检查：
    - 站姿稳定：kick 前若 z 低于站姿阈值，先 stand 归位恢复（防 greet 等前置扰动）
    - 摔倒即失败：任何阶段 z<0.04 → reason='fell'（姿态崩，需 Replan 而非盲目重试）
    失败原因记录在 orch._last_fail_reason（'fell'/'no_move'/'ball_moved'/'exhausted'）

    实测运动学（2026-09-02）：
    - vel0.3 walk 0.35s ≈ 2.4-5cm（低 vel 0.15/0.2 在近球区失效）
    - stand 会把鸭子拉回 HOME 姿态并前移 ~0.09m（改变球相对位置）
    - kick 踢窗：球心在鸭子局部前方 0.05-0.19m；left 万能脚全侧偏命中
    - kick 必须从站姿启动（动态中腿不到位）
    """
    import numpy as np
    m, d = orch.m, orch.d
    dt = m.opt.timestep
    adr = orch.ball_qpos_adrs[target_i]
    orch._last_fail_reason = None
    if not hasattr(orch, "_traj_events"):
        orch._traj_events = []

    # ---- Preflight：位姿异常先归位（EmbodiedSkills 最大缺口补丁）----
    if d.qpos[2] < 0.09:
        if verbose: print(f"    ⚠️ Preflight: z={d.qpos[2]:.3f} 低于站姿阈值，先 stand 归位")
        orch.cmd_stand(1.2)
        for _ in range(int(1.2 / (dt * 10))):
            orch.control_step(); orch.step(10)
            if d.qpos[2] < 0.04:
                orch._last_fail_reason = "fell"
                if verbose: print("    ❌ 归位时摔倒")
                return False, 0.0
        if verbose: print(f"    ✅ Preflight: 归位完成 z={d.qpos[2]:.3f}")

    for rnd in range(7):
        # 读球相对鸭子位置（局部坐标）
        bdx = d.qpos[adr] - d.qpos[0]
        bdy = d.qpos[adr + 1] - d.qpos[1]
        bd = np.hypot(bdx, bdy)
        yaw = _get_yaw(orch)
        cosy, siny = np.cos(yaw), np.sin(yaw)
        lx = cosy * bdx + siny * bdy
        ly = -siny * bdx + cosy * bdy
        side = "right" if ly > 0.05 else ("left" if ly < -0.05 else "front")
        if verbose:
            print(f"    [轮{rnd+1}] 球#{target_i+1} dist={bd:.2f}m local=({lx:.2f},{ly:.2f}) side={side} yaw={np.degrees(yaw):.0f}°")
        orch._traj_events.append({
            "t": round(float(d.time), 2), "round": rnd + 1, "ball": target_i + 1,
            "duck": [round(float(d.qpos[0]), 3), round(float(d.qpos[1]), 3)],
            "ball_local": [round(float(lx), 3), round(float(ly), 3)],
            "dist": round(float(bd), 3), "yaw_deg": round(float(np.degrees(yaw)), 1),
        })

        # 尝试踢击：若球够近（真实踢窗=球心 0.05-0.14m，站定可能前移 0-0.1m）
        if 0.06 < lx < 0.22:
            # 站定 1.5s（从 walk 动量完全衰减再踢；0.6s 不够——实测动量残留致踢空）
            orch.cmd_stand(1.5)
            for _ in range(int(1.5 / (dt * 10))):
                orch.control_step(); orch.step(10)
                if d.qpos[2] < 0.04:
                    orch._last_fail_reason = "fell"
                    if verbose: print("    ❌ 站定时摔倒")
                    return False, 0.0
            # 站定后重读
            bdx = d.qpos[adr] - d.qpos[0]
            bdy = d.qpos[adr + 1] - d.qpos[1]
            yaw = _get_yaw(orch)
            cosy, siny = np.cos(yaw), np.sin(yaw)
            lx2 = cosy * bdx + siny * bdy
            ly2 = -siny * bdx + cosy * bdy
            if verbose:
                print(f"    ↳ 站定后: 球局部({lx2:.3f},{ly2:.3f}) 球全局({d.qpos[adr]:.3f},{d.qpos[adr+1]:.3f}) 鸭({d.qpos[0]:.3f},{d.qpos[1]:.3f}) yaw={np.degrees(yaw):.0f}°")
            if 0.05 < lx2 < 0.22:
                # 站定后原地转向精确对齐（静止转，不走路；防 yaw 漂移导致踢偏）
                if abs(ly2) > 0.025:
                    tgt = yaw + np.arctan2(ly2, lx2)
                    _turn_to(orch, tgt, tol=0.02, max_steps=8, verbose=False)
                    # 转向后重读
                    bdx = d.qpos[adr] - d.qpos[0]
                    bdy = d.qpos[adr + 1] - d.qpos[1]
                    yaw = _get_yaw(orch)
                    cosy, siny = np.cos(yaw), np.sin(yaw)
                    lx2 = cosy * bdx + siny * bdy
                    ly2 = -siny * bdx + cosy * bdy
                    if verbose:
                        print(f"    ↳ 对齐后: 球局部({lx2:.3f},{ly2:.3f}) yaw={np.degrees(yaw):.0f}°")
                if 0.04 < lx2 < 0.20 and abs(ly2) < 0.12:
                    # 推球冲撞（实测比 kick 策略可靠：稳态站定后 walk 把
                    # 0.09m 前球碰飞 0.29m；kick 在远距逼近后命中不稳定）
                    # 行程：球距 0.04-0.20 用 1.2s（0.3m/s×1.2≈0.36m 加速空间）；
                    # 0.8s 在球距>0.12 时动量不足只蹭动（实测 0.14m 球只动 0.09m）
                    orch.cmd_walk((0.3, 0, 0), 1.2)
                    for _ in range(int(1.2 / (dt * 10))):
                        orch.control_step(); orch.step(10)
                        if d.qpos[2] < 0.04:
                            orch._last_fail_reason = "fell"
                            if verbose: print("    ❌ 冲撞时摔倒")
                            return False, 0.0
                    # 球若还在正前很近，补一次冲撞（0.8s）
                    bdx = d.qpos[adr] - d.qpos[0]
                    bdy = d.qpos[adr + 1] - d.qpos[1]
                    if np.hypot(bdx, bdy) < 0.18:
                        orch.cmd_walk((0.3, 0, 0), 0.8)
                        for _ in range(int(0.8 / (dt * 10))):
                            orch.control_step(); orch.step(10)
                            if d.qpos[2] < 0.04:
                                orch._last_fail_reason = "fell"
                                if verbose: print("    ❌ 二次冲撞时摔倒")
                                return False, 0.0
                    ball_disp = np.hypot(d.qpos[adr] - orch._ball0[target_i][0],
                                         d.qpos[adr + 1] - orch._ball0[target_i][1])
                    hit = ball_disp > 0.12
                    if not hit:
                        # 失败原因分叉：球动了但没够远 vs 球根本没动
                        orch._last_fail_reason = "ball_moved" if ball_disp > 0.02 else "no_move"
                    if verbose:
                        print(f"    ⚽ 推球冲撞！球位移 {ball_disp:.2f}m {'✅' if hit else '（未动）'}")
                    orch._traj_events[-1]["action"] = "push"
                    orch._traj_events[-1]["result"] = "hit" if hit else orch._last_fail_reason
                    orch._traj_events[-1]["ball_disp"] = round(float(ball_disp), 3)
                    return hit, ball_disp
            # 站定后球太远或太偏——继续逼近循环

        # 逼近：转向对齐球（阈值 0.05 + 侧偏强制）
        target_yaw = yaw + np.arctan2(ly, lx)
        yaw_err = ((target_yaw - yaw) + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_err) > 0.05 or abs(ly) > 0.06:
            _turn_to(orch, target_yaw, tol=0.04, max_steps=6, verbose=False)
        # 重读
        yaw = _get_yaw(orch)
        bdx = d.qpos[adr] - d.qpos[0]
        bdy = d.qpos[adr + 1] - d.qpos[1]
        bd = np.hypot(bdx, bdy)
        cosy, siny = np.cos(yaw), np.sin(yaw)
        lx = cosy * bdx + siny * bdy
        ly = -siny * bdx + cosy * bdy
        # 连续短步逼近 vel0.3（实测低 vel 近球失效；每段 0.35s≈2.4-5cm）
        if verbose:
            print(f"    ↳ 逼近: bd={bd:.2f} ly={ly:.2f} → walk vel0.3 连续短步 (yaw_err={np.degrees(yaw_err):.0f}°)")
        x_before, y_before = d.qpos[0], d.qpos[1]
        moved_total = 0.0
        for _seg in range(4):
            orch.cmd_walk((0.3, 0, 0), 0.35)
            for _ in range(int(0.35 / (dt * 10))):
                orch.control_step(); orch.step(10)
                if d.qpos[2] < 0.04:
                    orch._last_fail_reason = "fell"
                    if verbose: print("    ❌ 逼近时摔倒")
                    return False, 0.0
            moved_total += np.hypot(d.qpos[0] - x_before, d.qpos[1] - y_before)
            x_before, y_before = d.qpos[0], d.qpos[1]
            # 每段后重读并对齐（防走偏累积）
            yaw = _get_yaw(orch)
            bdx = d.qpos[adr] - d.qpos[0]
            bdy = d.qpos[adr + 1] - d.qpos[1]
            cosy, siny = np.cos(yaw), np.sin(yaw)
            lx = cosy * bdx + siny * bdy
            ly = -siny * bdx + cosy * bdy
            bd = np.hypot(bdx, bdy)
            tgt = yaw + np.arctan2(ly, lx)
            if abs(ly) > 0.04:
                _turn_to(orch, tgt, tol=0.02, max_steps=5, verbose=False)
            if bd < 0.13:
                break
            if moved_total < 0.01 and _seg > 0:
                break  # 卡死检测
        if verbose:
            print(f"    ↳ 位移: 共{moved_total:.3f}m 鸭子新位({d.qpos[0]:.3f},{d.qpos[1]:.3f})")
    if orch._last_fail_reason is None:
        orch._last_fail_reason = "exhausted"  # 7 轮逼近未达踢窗（球可能不可达/被卡）
    return False, 0.0

def execute_goal(orch, goal, verbose=True):
    """执行单个子目标"""
    name = goal.get("goal", "")
    if verbose:
        print(f"\n🎯 子目标: {name}" + (f"（note: {goal.get('note')}）" if goal.get("note") else ""))
    if name in ("kick_ball", "kick_all_balls"):
        import numpy as np
        m, d = orch.m, orch.d
        dt = m.opt.timestep
        if not hasattr(orch, "_ball0"):
            orch._ball0 = _ball_positions(orch)
        if not hasattr(orch, "_kicked_idx"):
            orch._kicked_idx = set()
        adrs = orch.ball_qpos_adrs
        # 找最近的未处理球（每次只处理一个目标）
        best_i, best_d = None, 1e9
        for i, adr in enumerate(adrs):
            if i in orch._kicked_idx:
                continue
            bd = np.hypot(orch.d.qpos[adr] - orch.d.qpos[0], orch.d.qpos[adr+1] - orch.d.qpos[1])
            if bd < best_d:
                best_d, best_i = bd, i
        if best_i is None:
            if verbose: print("    所有球都已处理 ✅")
            return True, {"status": "all_kicked"}
        hit, disp = _kick_cycle(orch, best_i, verbose)
        reason = getattr(orch, "_last_fail_reason", None)
        stop_disp, stop_reason = disp, reason
        if not hit:
            # 【抄 EmbodiedSkills】失败按原因分叉，不再无脑重试一次：
            # - fell: 姿态崩 → 先恢复站姿再重试（Replan 语义：恢复是重试前提）
            # - ball_moved: 球被推动(0.02-0.12m)=环境反馈 → 发散重试（位置变了可能更优角度）
            # - no_move: 球根本没动 → 可能是没碰到/撞偏 → 重读位置再试
            # - exhausted: 7 轮逼近失败 → 球可能不可达/被卡 → 止损换球
            if verbose:
                print(f"    ↻ 首次失败 reason={reason} disp={disp:.2f}m")
            retry_hit = False
            if reason in ("fell", "ball_moved", "no_move"):
                if reason == "fell":
                    # Replan：摔倒后先 stand 恢复（盲目重试会再摔）
                    if verbose: print("    🔧 Replan: 摔倒→先 stand 恢复 1.2s")
                    orch.cmd_stand(1.2)
                    for _ in range(int(1.2 / (dt * 10))):
                        orch.control_step(); orch.step(10)
                    if orch.d.qpos[2] < 0.09:
                        if verbose: print("    ❌ 恢复失败（仍站不稳），直接止损")
                    else:
                        retry_hit, stop_disp = _kick_cycle(orch, best_i, verbose)
                        stop_reason = getattr(orch, "_last_fail_reason", "unknown")
                else:
                    # 发散重试（球已被撞动或需重新逼近）——最多一次
                    if verbose:
                        print(f"    ↻ {'球被推动' if reason=='ball_moved' else '球未动'}，重试一次（环境已变）")
                    retry_hit, stop_disp = _kick_cycle(orch, best_i, verbose)
                    stop_reason = getattr(orch, "_last_fail_reason", "unknown")
                if retry_hit:
                    if verbose: print(f"    ✅ 重试成功！disp={stop_disp:.2f}")
                    orch._kicked_idx.add(best_i)
                    return True, {"kicked_idx": best_i, "disp": round(stop_disp, 2), "retried": True, "first_disp": round(disp, 2), "reason": reason}
            # 重试仍失败 / exhausted / 恢复失败 → 收敛止损：标记已处理继续下一球
            if verbose:
                print(f"    ⚠️ 止损(最后 disp={stop_disp:.2f} reason={stop_reason})，标记球#{best_i+1}已处理")
            orch._kicked_idx.add(best_i)
            return False, {"kicked_idx": best_i, "disp": round(stop_disp, 2), "retried": True, "first_disp": round(disp, 2), "reason": stop_reason}
        orch._kicked_idx.add(best_i)  # 无论踢中与否都标记已处理（避免死循环）
        if verbose:
            print(f"    标记球#{best_i+1}已处理，剩余未处理: {[i+1 for i in range(len(adrs)) if i not in orch._kicked_idx]}")
        return True, {"kicked_idx": best_i, "disp": round(disp, 2)}
    if name == "greet":
        from duck_skills import load_skill, skill_to_steps
        skill = load_skill("greet")
        r = orch.run_sequence(skill_to_steps(skill))
        return r["ok"], {"skill": "greet"}
    if name == "look_around":
        from duck_skills import load_skill, skill_to_steps
        skill = load_skill("look_around")
        r = orch.run_sequence(skill_to_steps(skill))
        return r["ok"], {"skill": "look_around"}
    return False, {"error": f"未知子目标 {name}"}

def run_task(task_text, verbose=True, seed=None):
    """主入口"""
    import numpy as np
    import mujoco
    from duck_orchestrator import DuckOrchestrator
    print(f"📋 任务: {task_text}")
    try:
        goals = llm_decompose(task_text)
        print(f"🧠 LLM 分解: {json.dumps(goals, ensure_ascii=False)}")
    except Exception as e:
        print(f"⚠️ LLM 失败({e})，默认分解")
        goals = [{"goal": "kick_all_balls", "note": None}]
    if not goals:
        print("❌ 无子目标"); return {"ok": False}
    # ball2 场景（双球）
    orch = DuckOrchestrator(scene="ball2", load_policies=["walk", "stand", "kick_left", "kick_right"])
    m, d = orch.m, orch.d
    dt = m.opt.timestep
    import random
    if seed is not None: random.seed(seed)
    # 双球初始位（沿用 XML 默认或随机）
    if len(orch.ball_qpos_adrs) >= 2:
        a0, a1 = orch.ball_qpos_adrs[0], orch.ball_qpos_adrs[1]
        d.qpos[a0:a0+3] = [0.55, 0.18, 0.035]
        d.qpos[a1:a1+3] = [0.85, -0.2, 0.035]
    mujoco.mj_forward(m, d)
    print(f"   球初始: {[tuple(round(v,2) for v in p[:2]) for p in _ball_positions(orch)]}")
    orch.cmd_stand(0.5)
    for _ in range(int(0.5 / (dt * 10))):
        orch.control_step(); orch.step(10)
    results = []
    ok_all = True
    for goal in goals:
        name = goal.get("goal", "")
        if name == "kick_all_balls":
            # 连续处理直到所有球踢完（失败球止损后继续下一球，不整体中断）
            guard = 0
            while True:
                guard += 1
                if guard > 12:
                    if verbose: print("    ⚠️ 循环保护触发（>12 轮）")
                    ok_all = False
                    break
                ok, detail = execute_goal(orch, goal, verbose)
                results.append({"goal": goal, "ok": ok, "detail": detail})
                if not ok:
                    ok_all = False
                    # 球已标记（止损），继续处理剩余球
                if detail.get("status") == "all_kicked" or detail.get("kicked_idx") is None:
                    break
                # 看是否还有未处理球
                remaining = [i+1 for i in range(len(orch.ball_qpos_adrs)) if i not in orch._kicked_idx]
                if not remaining:
                    break
                if verbose:
                    print(f"    ↻ 继续踢剩余球: #{remaining}")
                orch.cmd_stand(0.6)
                for _ in range(int(0.6 / (dt * 10))):
                    orch.control_step(); orch.step(10)
            continue
        ok, detail = execute_goal(orch, goal, verbose)
        results.append({"goal": goal, "ok": ok, "detail": detail})
        if not ok:
            ok_all = False
            break
        orch.cmd_stand(0.6)
        for _ in range(int(0.6 / (dt * 10))):
            orch.control_step(); orch.step(10)
    print(f"\n鸭子终点 ({d.qpos[0]:.2f}, {d.qpos[1]:.2f}) z={d.qpos[2]:.3f}")
    for r in results:
        g = r["goal"].get("goal")
        print(f"  {'✅' if r['ok'] else '❌'} {g}: {json.dumps(r['detail'], ensure_ascii=False)}")
    print(f"\n{'✅ 任务完成' if ok_all else '⚠️ 部分完成'}")
    # 【抄 EmbodiedSkills】任务轨迹结构化落盘（未来训 verifier/planner 的监督信号）
    try:
        traj = {
            "task": task_text,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ok": ok_all,
            "goals_llm": goals,
            "ball_end": [list(map(lambda v: round(float(v), 3), p)) for p in _ball_positions(orch)],
            "duck_end": {"pos": [round(float(d.qpos[0]), 3), round(float(d.qpos[1]), 3)], "z": round(float(d.qpos[2]), 3)},
            "results": results,
            "events": getattr(orch, "_traj_events", []),
        }
        import os as _os
        traj_dir = "<openclaw-workspace>/projects/microduck-virt/logs"
        _os.makedirs(traj_dir, exist_ok=True)
        traj_path = f"{traj_dir}/traj_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(traj_path, "w") as f:
            json.dump(traj, f, ensure_ascii=False, indent=1)
        print(f"    📜 轨迹落盘: {traj_path}")
    except Exception as e:
        print(f"    ⚠️ 轨迹落盘失败: {e}")
    return {"ok": ok_all, "results": results, "pos": (round(d.qpos[0],2), round(d.qpos[1],2))}

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "把场地里两个球都踢飞"
    run_task(task)

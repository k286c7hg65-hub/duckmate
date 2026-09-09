#!/usr/bin/env python3
"""Duck Commander v2: 自然语言指挥虚拟鸭子（集成技能库）

架构：LLM(DeepSeek) 意图解析 → DSL(含技能引用) → 技能展开 → DuckOrchestrator 执行
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/tmp")
sys.path.insert(0, "<openclaw-workspace>/projects/microduck-virt/scripts")

# DeepSeek API: 优先环境变量，fallback vision_ds 的 key 解析
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

def _skill_catalog():
    from duck_skills import list_skills
    try:
        skills = list_skills()
    except Exception:
        skills = []
    return "\n".join(f"- {s['name']} ({s['type']}): {s['description']}" for s in skills)

def llm_parse(prompt_text):
    """DeepSeek 意图解析 → DSL（支持原子行为 + 技能引用）"""
    import urllib.request
    url = "https://api.deepseek.com/chat/completions"
    skill_desc = _skill_catalog()
    system = f"""你是机器人行为编排器。把用户的自然语言指令转成 JSON 行为序列。
可用原子行为（步骤直接写 policy）：
- {{"policy":"walk","vel":[x,0,0],"dur":s}}  前进，x∈[0.2,0.8]
- {{"policy":"stand","dur":s}}  站立  |  {{"policy":"sit","dur":s}}  坐下
- {{"policy":"standup","dur":s}}  站起
- {{"policy":"head","pitch":p,"dur":s}}  点头(p负=低头)  |  {{"policy":"head","yaw":y,"dur":s}}  转头
- {{"policy":"kick","side":"left|right","dur":s}}  踢球  |  {{"policy":"roulade","dur":s}}  后空翻

组合技能（整体引用，格式 {{"skill":"名字"}}）：
{skill_desc}

规则：
1. 只输出 JSON 数组，无解释无 markdown
2. 动作间加 stand 缓冲≥0.5s；dur 1-5s；动作≤6
3. 语义映射："打招呼/欢迎"→[{{"skill":"greet"}}]；"环顾/看四周"→[{{"skill":"look_around"}}]；"追球射门"→[{{"skill":"walk_and_kick"}}]；"坐姿点头"→sit 后 head
4. 无动作意图输出 [{{"policy":"stand","dur":2}}]"""
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
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
    return json.loads(m.group(0))

def rule_fallback(prompt_text):
    """无 LLM 时的规则兜底（关键词→行为/技能）"""
    seq = []
    p = prompt_text.lower()
    if any(k in p for k in ["打招呼", "欢迎", "hello", "hi", "greet"]):
        seq.append({"skill": "greet"})
    if any(k in p for k in ["环顾", "看四周", "观察", "look"]):
        seq.append({"skill": "look_around"})
    if any(k in p for k in ["追球", "射门", "walk_and_kick"]):
        seq.append({"skill": "walk_and_kick"})
    if any(k in p for k in ["走", "前进", "过去", "walk"]):
        vel = 0.4 if "慢" in p else 0.6
        dur = 3.0 if not any(k in p for k in ["两步", "一会儿", "一下"]) else 1.5
        seq.append({"policy": "walk", "vel": [vel, 0, 0], "dur": dur})
    if any(k in p for k in ["坐", "蹲", "sit"]):
        seq.append({"policy": "sit", "dur": 2.0})
        if any(k in p for k in ["点头"]):
            seq.append({"policy": "head", "pitch": -0.4, "dur": 0.6})
            seq.append({"policy": "head", "pitch": 0.1, "dur": 0.5})
        seq.append({"policy": "standup", "dur": 2.5})
    if any(k in p for k in ["起来", "站起", "stand up"]):
        seq.append({"policy": "standup", "dur": 2.5})
    if any(k in p for k in ["踢", "kick"]):
        side = "right" if "右" in p else "left"
        seq.append({"policy": "kick", "side": side, "dur": 2.5})
    if any(k in p for k in ["翻", "空翻", "roulade"]):
        seq.append({"policy": "roulade", "dur": 2.0})
    if any(k in p for k in ["站", "stand"]):
        seq.append({"policy": "stand", "dur": 2.0})
    if not seq:
        seq.append({"policy": "stand", "dur": 2.0})
    return seq

def expand_skills(seq):
    """把技能引用展开为步骤序列"""
    from duck_skills import load_skill, skill_to_steps
    out = []
    for a in seq:
        if isinstance(a, dict) and "skill" in a:
            skill = load_skill(a["skill"])
            if skill:
                out.extend(skill_to_steps(skill, a.get("params")))
            else:
                print(f"  [skill] 未知技能: {a['skill']}")
        else:
            out.append(a)
    return out

def validate_seq(seq):
    """校验并规范化 DSL"""
    allowed = {"walk", "stand", "sit", "standup", "head", "kick", "roll", "roulade"}
    out = []
    for a in seq:
        if not isinstance(a, dict) or "policy" not in a:
            continue
        p = a["policy"]
        if p not in allowed:
            continue
        item = {"policy": p, "dur": max(0.5, min(5.0, float(a.get("dur", 2.0))))}
        if p == "walk":
            v = a.get("vel", [0.5, 0, 0])
            item["vel"] = (max(0.2, min(0.8, float(v[0]))), 0.0, 0.0)
        if p == "kick":
            item["side"] = "right" if a.get("side") == "right" else "left"
        if p == "head":
            item["pitch"] = float(a.get("pitch", 0.0))
            item["yaw"] = float(a.get("yaw", 0.0))
        out.append(item)
    return out

def run_command(text, scene=None, dry_run=False):
    """主入口：自然语言 → DSL(技能展开) → 执行（scene 自动推断）"""
    print(f"🎙️ 指令: {text}")
    seq = None
    if DEEPSEEK_KEY:
        try:
            seq = llm_parse(text)
            print(f"🤖 LLM 解析: {json.dumps(seq, ensure_ascii=False)}")
        except Exception as e:
            print(f"⚠️ LLM 失败({e})，规则兜底")
    if seq is None:
        seq = rule_fallback(text)
        print(f"📏 规则解析: {json.dumps(seq, ensure_ascii=False)}")
    seq = expand_skills(seq)
    seq = validate_seq(seq)
    print(f"✅ 展开后 DSL: {json.dumps(seq, ensure_ascii=False)}")
    if dry_run:
        return {"ok": True, "seq": seq, "dry": True}
    from duck_orchestrator import DuckOrchestrator
    policies = ["walk", "stand", "sitstand"]
    needs_ball = False
    needs_rollers = False
    for a in seq:
        if a["policy"] == "kick":
            policies += ["kick_left", "kick_right"]
            needs_ball = True
        if a["policy"] == "roll":
            policies.append("roll")
            needs_rollers = True
        if a["policy"] == "roulade": policies.append("roulade")
    if scene is None:
        scene = "rollers" if needs_rollers else ("ball" if needs_ball else "full")
    orch = DuckOrchestrator(scene=scene, load_policies=list(set(policies)))
    orch.cmd_stand(0.5)
    dt = orch.m.opt.timestep
    for _ in range(int(0.5 / (dt * 10))):
        orch.control_step(); orch.step(10)
    # sit 后跟 stand 缓冲 → standup；非 stand 间插缓冲（同策略连续不插）
    full = []
    for i, a in enumerate(seq):
        if i > 0:
            prev = seq[i-1]["policy"]
            cur = a["policy"]
            if cur == "stand" and prev == "sit":
                a = {"policy": "standup", "dur": a.get("dur", 2.5)}
                cur = "standup"
            elif prev == cur:
                pass  # 同策略连续（如 head→head 点头序列）不插缓冲
            elif prev not in ("stand",) and cur not in ("stand", "standup") and not (prev == "sit" and cur == "head"):
                full.append({"policy": "stand", "dur": 0.5})
        full.append(a)
    seq = full
    t0 = time.time()
    result = orch.run_sequence(seq)
    result["seq"] = seq
    result["wall_seconds"] = round(time.time() - t0, 1)
    return result

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "鸭子打个招呼"
    dry = "--dry" in sys.argv
    r = run_command(text, dry_run=dry)
    if not dry:
        status = "✅ 执行成功，全程无摔倒" if r["ok"] else f"❌ 摔倒于 {r['fallen_at']}s"
        print(f"\n{status} | 位移 {r['dist']:.2f}m | 用时 {r['wall_seconds']}s")
        print(f"事件: {[f'{t:.0f}s:{p}' for t, p in r['events']]}")

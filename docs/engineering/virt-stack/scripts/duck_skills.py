#!/usr/bin/env python3
"""Duck Skills: 技能库——原子 + 组合行为定义（quackd .duck 风格）

技能类型：
- atomic: 直接映射 orchestrator 行为（walk/stand/sit/...）
- composite: 组合序列（可含 head 命令、多策略切换、参数）

技能文件：JSON，每文件一技能。示例：
{
  "name": "greet",
  "description": "打招呼：点头两次 + 左右张望",
  "steps": [
    {"policy":"head","pitch":-0.5,"dur":0.6},
    {"policy":"head","pitch":0.2,"dur":0.6},
    {"policy":"head","pitch":-0.5,"dur":0.6},
    {"policy":"head","yaw":0.9,"dur":0.8},
    {"policy":"head","yaw":0.0,"dur":0.8}
  ]
}
"""
import json
import os

SKILL_DIR = os.path.join(os.path.dirname(__file__), "skills")

# 内置技能（原子，映射 orchestrator 行为）
ATOMIC_SKILLS = {
    "walk": {"description": "向前行走。参数 vel(0.2-0.8 m/s), dur(秒)"},
    "stand": {"description": "站立不动。参数 dur"},
    "sit": {"description": "坐下（蹲下）。参数 dur"},
    "standup": {"description": "从坐姿站起来。参数 dur"},
    "kick": {"description": "踢球。参数 side(left/right), dur"},
    "roll": {"description": "轮滑滑行。参数 dur"},
    "roulade": {"description": "翻滚/后空翻。参数 dur"},
}

# 内置组合技能（用 orchestrator 原语表达）
BUILTIN_COMPOSITE = {
    "greet": {
        "description": "打招呼：点头两下再左右张望（迎宾/回应呼唤）",
        "steps": [
            {"policy": "head", "pitch": -0.5, "dur": 0.5},
            {"policy": "head", "pitch": 0.15, "dur": 0.5},
            {"policy": "head", "pitch": -0.5, "dur": 0.5},
            {"policy": "head", "pitch": 0.15, "dur": 0.5},
            {"policy": "head", "yaw": 0.9, "dur": 0.7},
            {"policy": "head", "yaw": -0.9, "dur": 1.0},
            {"policy": "head", "yaw": 0.0, "dur": 0.7},
        ],
    },
    "look_around": {
        "description": "环顾四周：头左右扫视（观察环境/找球）",
        "steps": [
            {"policy": "head", "yaw": 1.2, "dur": 0.8},
            {"policy": "head", "yaw": -1.2, "dur": 1.4},
            {"policy": "head", "yaw": 0.0, "dur": 0.8},
        ],
    },
    "walk_and_kick": {
        "description": "走向前方并踢球（追球射门）",
        "steps": [
            {"policy": "walk", "vel": (0.45, 0, 0), "dur": 2.0},
            {"policy": "stand", "dur": 0.5},
            {"policy": "kick", "side": "left", "dur": 2.5},
            {"policy": "stand", "dur": 0.5},
        ],
    },
    "sit_and_wave": {
        "description": "坐下并用头示意（乖巧回应）",
        "steps": [
            {"policy": "sit", "dur": 1.5},
            {"policy": "head", "pitch": -0.4, "dur": 0.5},
            {"policy": "head", "pitch": 0.1, "dur": 0.5},
            {"policy": "head", "pitch": -0.4, "dur": 0.5},
            {"policy": "standup", "dur": 2.5},
        ],
    },
}

def load_skill(name):
    """加载技能定义（内置组合 > 文件 > 原子）"""
    if name in BUILTIN_COMPOSITE:
        return {"name": name, **BUILTIN_COMPOSITE[name]}
    if name in ATOMIC_SKILLS:
        return {"name": name, "description": ATOMIC_SKILLS[name]["description"], "steps": [{"policy": name}]}
    # 尝试文件
    path = os.path.join(SKILL_DIR, f"{name}.duck.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def list_skills():
    """返回技能清单（供 LLM 感知）"""
    items = []
    for name, meta in ATOMIC_SKILLS.items():
        items.append({"name": name, "type": "atomic", "description": meta["description"]})
    for name, meta in BUILTIN_COMPOSITE.items():
        items.append({"name": name, "type": "composite", "description": meta["description"]})
    # 文件技能
    if os.path.isdir(SKILL_DIR):
        for fn in sorted(os.listdir(SKILL_DIR)):
            if fn.endswith(".duck.json"):
                try:
                    with open(os.path.join(SKILL_DIR, fn), encoding="utf-8") as f:
                        s = json.load(f)
                    items.append({"name": s.get("name", fn), "type": "composite",
                                  "description": s.get("description", ""), "file": fn})
                except Exception:
                    pass
    return items

def skill_to_steps(skill, params=None):
    """技能 → 步骤序列（参数注入）"""
    steps = []
    for s in skill.get("steps", []):
        step = dict(s)
        p = step.get("policy")
        # 参数覆盖：vel/side
        if params:
            if p == "walk" and "vel" in params:
                step["vel"] = params["vel"]
            if p == "kick" and "side" in params:
                step["side"] = params["side"]
        # dur 默认 2s
        step.setdefault("dur", 2.0)
        steps.append(step)
    return steps

if __name__ == "__main__":
    print("可用技能:")
    for s in list_skills():
        print(f"  [{s['type']}] {s['name']}: {s['description']}")

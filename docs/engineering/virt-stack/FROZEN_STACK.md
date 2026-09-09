# MicroDuck 行为栈冻结清单（FROZEN v0.1）

> 2026-09-03 | 轨 A「可移植冻结」第一步 | git baseline: 940e2b7
> 目的：M2 (圣诞) sim2real 移植时，行为栈的每一层都有版本、依赖、验证命令可循。

## 冻结范围

| 层 | 脚本 | 行数 | 职责 | 依赖 |
|---|---|---|---|---|
| 指挥层 | duck_commander.py | 212 | 自然语言→意图→DSL→执行 | DeepSeek API（key 自 openclaw.json 解析）、orchestrator |
| 认知层 | duck_planner.py | 461 | 任务分解、多球连续、失败分叉(fell/ball_moved/no_move/exhausted)、Preflight、轨迹落盘 | DeepSeek API、orchestrator、logs/ |
| 编排层 | duck_orchestrator.py | 333 | 9×ONNX 策略会话、行为 DSL、ground_pick 相位、推球冲撞 | 9 个 ONNX、scene.xml(allcollisions)、50Hz 定时 |
| 技能库 | duck_skills.py | 133 | 7 原子 + 4 组合技能、head 槽位命令 | orchestrator 行为名 |
| 视觉层 | duck_vision.py | 176 | 渲染→vision 定位球→决策→逼近 | vision_ds/vision_direct、orchestrator |
| 渲染 | render_duck2.py | 101 | 单策略行走视频 | MuJoCo EGL surfaceless |
| demo | render_planner_demo.py | 88 | 长程任务渲染出片 | planner |
| 验证 | verify_duck2.py | 110 | 无渲染策略验证 | MuJoCo |

## 验证基线（冻结时刻实测）

- 双球连续踢飞：0.37m / 0.44m（greet + kick_all_balls，全程无摔倒）
- 失败重试闭环：单测 4 场景全过（fell→恢复重试 / ball_moved→发散重试 / no_move→重试 / exhausted→止损）
- Preflight：kick 前 z<0.09 先归位 1.2s
- 轨迹落盘：logs/traj_<ts>.json（含轮次事件 8 条）
- 9 策略全验证（含 ground_pick 相位修复后下探-回升完整动作谱）

## 环境依赖（冻结）

- Python 3.10 + mujoco + onnxruntime + numpy + pillow（CPU 零 GPU）
- 9×ONNX 策略来自 pollen-robotics/microduck（官方）
- scene.xml = allcollisions 底座（姿势策略必需，robot_walk.xml 碰撞不全）

## 冻结纪律

1. 此后任何行为栈改动 → 新 commit，版本号递增（v0.1 → v0.2）
2. 每次 commit 前跑验证基线（双球 demo 或 verify 脚本）确认不回归
3. sim2real 移植以本清单为索引——每个脚本移植真机时逐项核对依赖

## 待增强（冻结后轨 A 后续，按 Sky 决策）

- [ ] 多轮会话状态（duck_commander 会话上下文）
- [ ] 轨迹失败模式自动分析（logs/traj_*.json → 统计）
- [ ] 视觉连续追踪（当前单帧定位）

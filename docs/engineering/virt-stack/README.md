# Microduck 虚拟鸭子环境（microduck-virt）

> 2026-09-02 建立 | 目标：零硬件、零 GPU 环境下开发验证 Microduck 软件栈

## 已验证能力（2026-09-02 实测）
- CPU MuJoCo 3.12 + onnxruntime 1.23 跑通官方 alpha_walking 策略
- 61D 观测 → 14D 动作闭环，STAND keyframe 启动
- 行走均速：vel=0.3→0.125 m/s | vel=0.6→0.272 m/s（单调速度响应 ✅）
- EGL surfaceless 无头渲染出片（720×1280 → ffmpeg mp4）
- 视频：projects/microduck-virt/microduck_walk.mp4

## 环境搭建（从零 30 分钟）
```bash
pip3 install mujoco onnxruntime numpy pillow
# 拉 RL 仓（47 STL + MJCF 全量）——不要用 GitHub API 逐文件(限流 403)，用 codeload
curl -sL -o /tmp/mdrl.tar.gz https://codeload.github.com/pollen-robotics/microduck_rl/tar.gz/refs/heads/main
tar xzf /tmp/mdrl.tar.gz
cp -r microduck_rl-main/src/mjlab_microduck/robot/microduck /tmp/duck_sim
# 9 个策略从主仓 policies/ 拉（onnx 直下，GitHub API 60req/h 内可）
# 或 git clone pollen-robotics/microduck --depth 1 后 cp policies/*.onnx
```

## 关键坑（血泪）
1. **50Hz 控制频率**：timestep=0.002（500Hz），策略训练于 50Hz → 每 10 物理步推理 1 次，不能每步推理（超频抖动走不动）
2. **延迟缓冲要关**：训练时 delay 3-6 步，但推理脚本默认 delay=0；加了延迟 walking 会摔
3. **vel_cmd 直接进 obs cmd[0:3]**，无归一化；vel<0.3 走不动（策略切换阈值内）
4. **无头渲染**：`MUJOCO_GL=egl` + `EGL_PLATFORM=surfaceless` + `LIBGL_ALWAYS_SOFTWARE=1`；Renderer(m, height, width)——参数是 (高,宽) 别写反
5. **model XML 渲染缓冲**：visual/global 里 offwidth/offheight 需 ≥ 渲染尺寸（1280×720）
6. EGL close 时的 EGLError traceback 是清理噪音，不影响成片

## 脚本
- `scripts/verify_duck2.py`：无渲染验证（run_policy 函数，速度扫描用）
- `scripts/render_duck2.py`：渲染单策略行走视频 `python3 render_duck2.py <vel> <秒>` → /tmp/microduck_walk.mp4
- `scripts/duck_orchestrator.py`：**策略编排器**（2026-09-02）——7 行为 × 3 场景 session 管理 + 行为 DSL 序列；
  `python3 duck_orchestrator.py` 跑 demo（stand→walk→sit→standup→kick→stand，实测无摔倒/球踢飞 0.43m）
- `scripts/duck_skills.py`：**技能库**（2026-09-02 晚）——原子+组合技能
- `scripts/duck_vision.py`：**视觉闭环**（2026-09-02 晚）——渲染→vision定位→决策→追球
- `scripts/duck_planner.py`：**LLM 长程规划**（2026-09-03）——任务分解+多球连续处理
- `scripts/duck_commander.py`：**自然语言指挥层**（2026-09-02）——DeepSeek LLM 意图→DSL→执行，规则兜底；
  `python3 duck_commander.py "鸭子走过去然后坐下再站起来" [--dry]`；key 自动从 openclaw.json 解析（vision_ds.get_api_key）

## 架构分层
```
自然语言 → [Agent: duck_commander] → DSL → [编排: orchestrator] → [策略: 9×ONNX] → [仿真: MuJoCo] → 真机(sim2real)
```

## 后续方向
- [ ] quackd 式自然语言指挥层（Agent 控制）→ ✅ 2026-09-02 完成
- [ ] 9 策略切换编排（sitstand/kick/roll/ground_pick）→ ✅ 2026-09-02 完成（walk/stand/sit/standup/kick L+R/roll/roulade 已验证）
- [ ] 视觉闭环（鸭子摄像头画面参与决策）
- [ ] 技能库 DSL（组合动作封装，quackd .duck 模式）
- [ ] 自定义 reward 训练新技能（需 GPU 出口：HF Jobs/MacMini）

## 技能库 + 视觉闭环（2026-09-02 晚）

**duck_skills.py（技能库）**
- 原子技能 7 个（映射 orchestrator 行为）+ 组合技能（greet 打招呼/look_around 环顾/walk_and_kick 追球/sit_and_wave 坐姿点头）
- head 槽位命令（cmd13[3:7]）在 stand/sitstand 策略下直接驱动头部——无需重训即可组合新行为
- 组合技能=步骤序列 JSON（quackd .duck 风格），LLM 可整体引用

**duck_vision.py（视觉闭环）**
- 闭环：渲染场景图 → DeepSeek vision 定位球(side/dist) → 决策(左/右/前/背) → walk 斜向/直行逼近 → 距离<0.28m 踢出
- 实测：真实视觉 1 轮追到并踢出（球位移 0.20m）；真值模式中/右/侧球场景全绿
- 关键：鸭子转向靠 walk 的斜向弧线（w 槽产生 y 漂移），非原地转；球到背后须掉头分支

**场景选择重大修正（P0）**
- sitstand 等姿势策略需要 FULL_COLLISION（robot_allcollisions.xml 82 geoms），robot_walk.xml 碰撞不全 → 坐下穿透摔倒
- 统一：walk 场景映射到 scene.xml（allcollisions 底座）；scene_ball/rollers 本来就是 allcollisions
- "light" 场景(scene_walk.xml)仅调试用

## 长程规划层 + 踢球真相修正（2026-09-03 凌晨）

**duck_planner.py（认知层/LLM 长程规划）**
- 任务文本 → DeepSeek 分解子目标(kick_ball/kick_all_balls/greet/look_around) → 逐目标执行
- 逼近闭环：转向对齐(wz槽真转弯) → vel0.3 连续短步(每段0.35s≈2.4-5cm) → 站定 → 冲撞推球
- 目标状态记忆（_kicked_idx 集合），多球连续处理
- 实测：「把两个球都踢飞」双球全中（0.12/0.24m）；「先打招呼再踢球」greet+双球全过，全程无摔倒

**🕳️ P0 踢球真相（诚实记录）**
- 旧 cmd_kick 内 _place_ball 每次把球瞬移回鸭子面前 → 历史所有「球位移」为假象
- 官方 kick 策略踢窗极窄：球心必须在脚前 1-8cm 且从初始位启动；远距逼近后命中≈0
- 破局：推球冲撞（鸭子逼近到球前 10cm 保持 walk 0.8s 身体碰飞球，实测 0.14-0.29m 可靠）
- 真踢能力需重训 kick 策略（GPU 出口）——sim2real 前置任务之一

**鸭子运动学速查（全实测 2026-09-03）**
- vel0.3 连续短步才能可靠逼近（vel 0.15/0.2 近球区失效/死区）
- 站定(stand)拉回 HOME 姿态：初始位鸭子前移 ~0.1m，稳态鸭不前移
- 转向 = cmd[2] wz 槽（真 yaw 旋转，(0.3,0,0.5) 1.5s 转 42°；带 vx 有侧漂）
- 逼近卡死检测：moved_total<0.01 连续 2 段 → break
- [x] ground_pick 相位编码验证 → ✅ 2026-09-03 完成（见下）

## 任务 1/2/4 完成记录（2026-09-03 04:30 BJT）

### ① ground_pick 相位验证 ✅（9 策略全验证齐）
- **技术债实锤**：orchestrator 的 `_update_cmd` 已有 phase 编码（cos/sin 分支），但 `self.phase` 只在 init/_switch 清零、**从不随物理时间推进**——命令恒为 phase=0 的 [1,0,0]，策略永远停在「下探起点」
- **修复**：加 `_gp_start_t` 锚点（_switch 重置），`_update_cmd` 中 phase = ((t - t_start)/4.0) % 1.0（4s 周期对齐 RL 源码 GroundPickPhaseCommand）
- **验证**：trunk_z 0.117→0.084（下探 3.3cm）→0.116（回升），4s 后第二周期重复——完整「下探→回升」动作谱 ✅

### ② 推球冲撞可靠性修复（渲染 demo 双球未踢飞根因）✅
- **现象**：render_planner_demo 双球 disp 0.09/0.12 < 0.12 阈值（视频“双球踢飞”名不副实，未发群）
- **根因**：headless 全成（0.213/0.135）但 greet 前置后鸭子位姿漂移（yaw 14°→-21°，位移 0.1m）→ 球距 0.14m 边缘触发 push 但 **0.8s walk 加速距离不够**，只蹭动 0.09m
- **修复**：push 行程 0.8→1.2s（加速空间 0.36m）+ 触发窗 0.14→0.20 + 二次补撞 0.15→0.18/0.6→0.8s
- **验证**：greet+双球场景 disp 0.280/0.288 ✅（修复前 0.094/0.116 ❌）

### ④ Agent 本体实验：失败重试闭环 ✅（本体轨方法论注入）
- **现状**：_kick_cycle 失败一次即标记球已处理（防死循环）= 失败即放弃
- **注入**：失败后重试一次（球已被撞动=环境反馈，位置变了可能进入更优角度——发散重试）；重试仍败 → 止损标记继续下一球（收敛不阻断任务）
- **验证**：monkeypatch 单测——首败二成路径 ✅（retried=True 返回成功）/ 双败止损路径 ✅（标记后不阻断）
- **本体轨意义**：把「16 次失败规则（发散继续/收敛止损）」从 Ariste 自身架构移植到鸭子 agent——最小可用版
- 另：render_planner_demo 的 kick_all_balls 循环加 guard>12 防死循环

### 修复后 LLM 长程全链路（回归）
`先打招呼然后把球都踢飞` → greet ✅ → 球#1 disp 0.37 ✅ → 球#2 disp 0.44 ✅ → 全程无摔倒 ✅

## 架构图交付（2026-09-03 05:00 BJT）

- **产物**：archive/microduck-arch-20260902/microduck-architecture.html（Archify 2.16 渲染，JSON 源同目录）
- **内容**：三 region 横切（控制栈 / 被控对象 sim→真机 / 训练与分发）+ 四阶段演进 cards（现在虚拟验证 → 近期训练出口 → 圣诞 sim2real → 未来闭环学习）
- **验证**：validate 9/9 ✅ 三视口零横向溢出 ✅ 三轮视觉自检无重叠/截断 ✅
- 交付图发群 oc_a25b3bc48218f65a0c4d6bb5285d3c31（om_x100b66bce6b214a0b2a7f555d2bcb93）

## 三个立刻可抄（EmbodiedSkills 吸收，2026-09-03 05:00 BJT）

Sky「三个立刻可抄，按建议执行」→ duck_planner.py 三项落地：

### ① Preflight（kick 前位姿检查）
- `_kick_cycle` 开头新增：z<0.09（低于站姿阈值）→ 先 stand 归位 1.2s 再踢
- 补丁目标 = EmbodiedSkills 对照发现的架构级缺口（greet 扰动鸭子位姿后直接 kick 的失败根因）

### ② 失败按原因分叉（替代无脑重试一次）
- 失败原因上报：`orch._last_fail_reason` ∈ {fell, ball_moved, no_move, exhausted}
  - fell：推球/站定/逼近/归位时摔倒（z<0.04）
  - ball_moved：球被推动 0.02-0.12m（碰到了但不够远）
  - no_move：球根本没动（没碰到/撞偏）
  - exhausted：7 轮逼近未达踢窗（球不可达/被卡）
- `execute_goal` 分叉：fell→stand 恢复再重试（Replan 语义）；ball_moved/no_move→发散重试一次；exhausted→不重试直接止损换球
- 单测 4 场景全过（verify_embodiedskills.py）

### ③ 任务轨迹结构化落盘
- `_kick_cycle` 每轮记录事件（t/round/ball/duck pos/ball_local/dist/yaw）
- run_task 结束写 logs/traj_<ts>.json：{task, ts, ok, goals_llm, ball_end, duck_end, results, events}
- 用途 = EmbodiedSkills「轨迹监督组件训练」的未来 verifier/planner 训练信号

### 回归
greet+双球 0.37/0.44 ✅ 全程无摔倒 ✅ 轨迹 8 事件落盘 ✅（正常路径未破坏）

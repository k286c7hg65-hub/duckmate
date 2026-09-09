# orchestrator 迁移映射表（A2.1 交付 · MicroDuck → ODM v2）

> 结论（一手实证）：**两机器人动作空间 1:1 同构（14→14），命名全一致**——迁移无维度映射，只有执行器参数层替换
> 依据：microduck-rl-fork _LEG_JOINTS/_NECK_JOINTS（src 实测）vs ODM v2 XML 关节清单（Playground 实测）

## 一、动作空间映射（14 = 10 腿 + 4 颈头）

| MicroDuck 索引 | MicroDuck 关节 | ODM v2 关节 | ODM 索引 | 迁移动作 |
|---|---|---|---|---|
| 0 | left_hip_yaw | left_hip_yaw | 0 | 直接映射 |
| 1 | left_hip_roll | left_hip_roll | 1 | 直接映射 |
| 2 | left_hip_pitch | left_hip_pitch | 2 | 直接映射 |
| 3 | left_knee | left_knee | 3 | 直接映射 |
| 4 | left_ankle | left_ankle | 4 | 直接映射 |
| 5 | neck（颈 pitch）| neck_pitch | 5 | 直接映射 |
| 6 | head_pitch | head_pitch | 6 | 直接映射 |
| 7 | head_yaw | head_yaw | 7 | 直接映射 |
| 8 | head_roll | head_roll | 8 | 直接映射 |
| 9 | right_hip_yaw | right_hip_yaw | 9 | 直接映射 |
| 10 | right_hip_roll | right_hip_roll | 10 | 直接映射 |
| 11 | right_hip_pitch | right_hip_pitch | 11 | 直接映射 |
| 12 | right_knee | right_knee | 12 | 直接映射 |
| 13 | right_ankle | right_ankle | 13 | 直接映射 |

**备注**：MicroDuck 物理上 15 舵机（14 受控 + 1 被动喙/天线？），ODM v2 = 14 受控全同构。若 MicroDuck 有第 15 个非受控自由度，编排层忽略即可。

## 二、需在迁移时实测/校准的参数（执行器差异层）

| 参数 | MicroDuck（XL330）| ODM v2（STS3215）| 动作 |
|---|---|---|---|
| 关节行程（qpos 限位）| soft_joint_pos_limit_factor 0.95 | ODM XML 定义（以 XML 为准）| 读 ODM MJCF 限位 → 替换 |
| 正方向 | XL330 安装方向 | STS3215 安装方向（可能反）| **真机标定：逐一验证 ± 方向** |
| 关节零位 | 官方模型 | ODM XML home | 以 ODM 装配体 home 为准 |
| 力矩上限 | XL330 0.42-0.60 N·m | STS3215 1.91 N·m | 策略 torque clip 放宽（3 倍余量）|
| 速度上限 | XL330 规格 | max_motor_velocity 5.24 rad/s（实测 joystick.py）| 用 ODM 值 |
| 通信 | Dynamixel 协议 | feetech 协议 | rustypot.feetech 模块（ODM 同款）|

## 三、编排层（orchestrator 7 原子/4 组合）迁移判定

- **原子技能定义**（走/站/坐/踢/捡/头动/表情）→ 引用的是关节语义（「左腿前踢」「头 yaw 左转」）非索引 → **语义层零修改** ✅
- **技能实现**（DSL → 关节目标序列）→ 关节名一致 → **DSL 脚本零修改** ✅
- **底层策略**（踢球 ONNX 等）→ MicroDuck 训练产物，**不能直接跑**（动力学不同）→ ODM Playground 重训（standing/joystick 现成 + 踢球自定义）⚠️
- **观测契约**：MicroDuck actor obs（重力投影/角速度/关节 pos+vel/无球盲视）→ ODM Playground observation 定义需对齐（base.py 的 observation_size["state"]，实测动作前比对维度）⚠️ 小改

## 四、迁移工作量结论

| 层 | 工作量 | 说明 |
|---|---|---|
| 动作空间 | ≈0 | 1:1 同构（本表）|
| 关节名/DSL | 0 | 语义一致 |
| 观测契约 | 小（≤1 天）| obs 维度对齐 + IMU 噪声参数 |
| 策略层 | 中（M1 gate 起）| standing/joystick 现成；踢球重训 |
| 执行器校准 | 真机必做 | 方向/零位/限位逐关节标定（M2 原型期）|
| **编排层整体** | **≈2-3 天** | 主要耗在 obs 适配 + 测试 |

## 五、对 BP v3.0 的意义

- 「行为编排层硬件无关可迁移」从**推论升级为实证**（动作空间同构 = 编排层核心接口零改动）
- 虚拟鸭栈（duck_orchestrator/duck_skills）可原样作为 ODM 行为层——M1 gate 后接入
- 主要软件工作 = 踢球重训（拳头差异化技能）+ obs 适配，非架构改造

---
_生成：Ariste 2026-09-03 | 对应 ariste-tasklist A2.1 | 依据：microduck-rl-fork src + ODM Playground XML 实测_

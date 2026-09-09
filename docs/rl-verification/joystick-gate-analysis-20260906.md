# Spec #2 门槛 2/3 源码级分析 — joystick obs/时序核对

日期：2026-09-06 | 作者：Ariste | 基准：本地训练迁移库 a0d82ad（≈Codex 审计 SHA 3be79a42 的 fork 版）
状态：分析完成，evaluator 实现与 patch 待提交

---

## 1. 101 维 obs 布局 — 源码级确认（门槛 2 依据）

joystick.py `_get_obs` state 段（非特权 actor 观测），逐字段：

| 段 | 元素 | 维度 |
|---|---|---|
| noisy_gyro | 3 | 3 |
| noisy_accelerometer | 3 | 3 |
| info["command"] | sample_command 返回（vx,vy,yaw_rate,neck_pitch,head_pitch,head_yaw,head_roll） | **7**（源码注释「#3」过时误导） |
| noisy_joint_angles − default_actuator | 10 leg + 4 head 关节 | nu=14 |
| noisy_joint_vel × dof_vel_scale(0.05) | 同上 | 14 |
| last_act | 三连之一 | 14 |
| last_last_act | 三连之二 | 14 |
| last_last_last_act | 三连之三 | 14 |
| motor_targets | 本步计算目标 | 14 |
| contact | 双足触地 | 2 |
| imitation_phase | cos/sin | 2 |
| **合计** | | **3+3+7+84+2+2 = 101 ✅** |

**结论**：Codex 的 101 维描述（gyro3+acc3+cmd7+6×joint14+contact2+phase2）与源码**精确一致**。6 组 × 14 = 角度、速度、三档 action history、motor_targets（注释「#10」为 25cm 旧版残留，实际 nu=14 via `self._mj_model.nu`）。

**噪声关闭时的对照构造**（门槛 4 用）：noise level=0 时 gyro/acc/joint_angles/joint_vel 均无 ±uniform 扰动；accelerometer 有 **+1.3 重力 hack**（`accelerometer.at[0].set(accelerometer[0]+1.3)` 活代码，非注释）——**joystick 训练侧存在 acc bias**（standing 没有——两任务不同，评估器按 joystick 实现）；imu_history 延迟采样（min0/max3）在 noise level=0 时仍走 roll 但 uniform 恒 0 档？**非也**——noise level=0 只关加性噪声，imu_idx 的 random 采样仍在（uniform min0 max3 → 随机档，但 history 内容全相同[无噪声→gravity 恒定] → 无实际影响 ✅）；action delay（min0 max3）**不受 noise level 控制**（见下 §3）——延迟采样始终随机。

**phase**：USE_IMITATION_REWARD=True 时 imitation_i 每 env step +1，phase = [cos(i/N×2π), sin(...)]，N = PRM.nb_steps_in_period（来自 polynomial_coefficients.pkl）。

## 2. Spec 首轮 patch 清单（相对源码 default_config，5 处）

| # | 项 | 源码默认 | Spec v1 要求 |
|---|---|---|---|
| 1 | lin_vel_x | [-0.15, 0.15] | [0.05, 0.15]（90% 采样域；10% 零已有 bernoulli 机制 ✅） |
| 2 | lin_vel_y | [-0.2, 0.2] | [0.0, 0.0] |
| 3 | ang_vel_yaw | [-1.0, 1.0] | [0.0, 0.0] |
| 4 | head_range_factor | 1.0 | 0.0（头颈 4 命令恒 0，但 14 维动作输出保留） |
| 5 | push_config.enable | True | False（首轮；扰动另立一轮） |

reward 数值（2.5/6.0/-0.001/-0.5/-0.2/20/1.0）与 tracking_sigma=0.01、action_scale=0.25、max_motor_velocity=5.24（→±0.1048 rad/周期）与 Spec **已一致**，无需改。命令重采样每 500 env steps = 10s ✅（Spec「保留 10 秒重采样」）。

⚠️ **runner 无这些 config 的 CLI**（Spec 已指出）→ patch 需直改 joystick.default_config 或加 env_config 注入；**env 与 eval_env 必须接收同一 config 对象**（只改 runner.env_config 不生效——runner 创建 env 后 config 已固化，需确认 runner 的 env 构造路径）。

## 3. 门槛 3：动作历史时序差异 — 精确定位

### 训练侧（joystick.py step，输入 action_n）
```
① imitation_i/phase 更新（step 开头）
② action_history = roll + set(action_n)（延迟缓冲，物理 step 前）
③ action_w_delay = action_history[delay_idx]  (delay_idx ~ U{0,1,2,3})
④ motor_targets = default + action_w_delay×0.25（clip ±0.1048）
⑤ 物理 step（执行 motor_targets）
⑥ contact/air_time/swing 更新
⑦ obs 构造 ← last_act 三连 = action_{n-1},action_{n-2},action_{n-3}（⑨ 尚未执行）
   motor_targets = 本步 ④ 的新值（delay 版目标）
⑧ done/reward
⑨ last_last_last←last_last; last_last←last_act; last_act = action_n（原始未延迟）
```

### 推理侧（mujoco_infer.py run，每 ctrl 周期）
```
a. 物理子步 ×10
b. phase 更新（imitation_i += phase_frequency_factor×1.0）
c. obs 构造 ← last_action 三连 = action_{n-1} 系列；motor_targets = 上周期执行值（旧）
d. action_n = policy.infer(obs)
e. last_last_last←last_last; last_last←last_action; last_action = action_n
f. motor_targets = default + action_n×0.25（clip ±0.1048）→ data.ctrl
```

### 两处实质差异（Codex「一档差异」的精确定位）

**差异 A — motor_targets 时相**：训练 obs_n.motor_targets = **本步将执行的 delay 版目标**（step 中段可见）；推理 obs_n.motor_targets = **上周期已执行目标**（周期开头构造，f 未跑）。观测分布差一档。

**差异 B — 动作延迟缺失**：训练物理执行 action_w_delay（delay_idx ∈ {0,1,2,3} 随机，action_min/max_delay 不受 noise level 控制 → **始终随机**）；推理执行当步 action_n（恒 delay=0，无 action_history 缓冲——49-51 行仅 last 三连）。训练中仅 ~25% 步与推理同分布 → **obs 的 last_act 三连在训练侧 = 原始 action 序列（与执行序列差 0-3 档），推理侧 = 实际执行序列（差 0 档）**。

### 统一语义候选方案（待 Codex 复核）

- **方案 A（推荐，保 DR 基线）**：推理侧补 action_history 延迟缓冲（f 前从缓冲取 delay 档执行；last 三连仍记原始 infer action）→ 两侧「obs 三连 = 原始 action 序列、执行 = delay 档」语义一致；motor_targets 差异 A 另行处理。
- **方案 B（最简单，但动训练 DR）**：训练 action_min_delay=action_max_delay=0（关延迟）→ 两侧恒 delay=0，obs/执行完全同轨；代价 = 与 Spec「动作延迟保留基线」冲突，需 Codex 豁免。
- 差异 A 处理（两案共通）：统一为「obs 的 motor_targets = 最近一次已执行决策目标」→ 训练侧 info 需存 prev_motor_targets 供 obs 读取（或 obs 用 ④ 前旧值），推理侧不变（已是旧值语义）。

**schema 版本**：本 fork 定义 `OBS_SCHEMA = "joystick_v1_101d"`（字段序见 §1 表），写入 evaluator 与回传 JSON。standing 85 维模型不套用（85 布局 command=7+joint14×5+contact2 无 motor_targets/phase——已有文档，不动）。

## 4. 下一步（本地零成本）

1. joystick_eval.py：独立 101 维 obs 构造（§1 表逐字段，噪声关闭路径）+ 14 维动作 + 起停命令调度 + 判据（z>0.05/倾角 60°×0.2s/横向 ≤0.20m/航向 ≤15°/每脚 5 次摆动 ≥5mm×0.04s/vx MAE/净位移/末段停）→ CPU rollout 冒烟
2. 时序 patch 设计稿（方案 A/B 差异表）→ 交 Codex 复核后再改训练侧
3. 逐字段对齐工具（对照训练 jax 路径 vs evaluator numpy 路径，≤1e-5 门槛）→ 需 joystick checkpoint 产出后执行（门槛 4 实际运行在训练后）

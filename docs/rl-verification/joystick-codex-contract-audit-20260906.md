# 契约审计批 3 — custom_rewards / PRM / 命令 / obs 全链路核对（Codex 复核逐项回应）

日期: 2026-09-06 | 对应 EVIDENCE_REVIEW_20260906_BATCH2 §1/§3/§4 全部缺口 | 全部本地源码 + 远端训练环境实测

## 1. custom_rewards.py 完整函数核对（Codex 对, 修正两处描述）

reward_imitation 实际结构（joystick 调用的已加载版本）:
- **orientation 计算但未加入总和**（`# + torso_orientation_rew` 注释）——Codex 正确; 且切片 `reference_frame[3:7]` 取的是 joint_pos dim3-6（L_knee/L_ankle/neck/head）非 root quat——**reference 40 维无 root quat**（见 §3）。双重无害（未用）。
- **contact_rew = Σ(contacts == ref_contacts>0.5) × w_contact** —— 匹配计数 0/1/2, **非 exp 项** —— Codex 正确
- 实际求和: I = lin_vel_xy_rew(1) + lin_vel_z_rew(1) + ang_vel_xy_rew(0.5) + ang_vel_z_rew(0.5) + joint_pos_rew(−15·E_q) + joint_vel_rew(−0.001·E_dq) + contact_rew(2 max)
- **B≤5 修正**: B_max = 1+1+0.5+0.5+2 = 5, 由 lin_xy/lin_z/ang_xy/ang_z(exp) + contact(计数) 构成 —— 无 orientation
- 尾部 `reward *= cmd_norm > 0.01`（零命令 gate）→ cmd=0 时 I=0（实测吻合）
- joint 切片: actual = joints_qpos[:5]+[9:]（10 腿）; ref = ref[:5]+ref[11:]（10 腿）—— **对齐正确**（见 §2）

## 2. 关节序/切片核对（通过, 无错位）

- xml joint 序（open_duck_mini_v2.xml）: floating_base(0), L_hip_yaw..L_ankle(1-5), neck_pitch(6), head_pitch(7), head_yaw(8), head_roll(9), R_hip_yaw..R_ankle(10-14); actuator = joint 1-14（14 个, nu=14）
- PRM 参考 16 维（pkl 注释）: L腿5 + neck/head×3 + **L_ant/R_ant(9,10)** + R腿5 —— 模型无 antenna, 参考含 2 antenna 占位
- actual [:5]+[9:] = L腿5 + R腿5 ✅; ref [:5]+[11:] = L腿5 + R腿5 ✅（跳 5-10 = 头颈4+ant2）
- **切片逻辑正确**; joints_qpos 由 qpos[actuator_joint_ids qposadr] 取（get_actuator_joints_qpos, base.py:193）

## 3. reference_frame 结构（59 维原始 → 40 维拟合）

- pkl frame_offsets: root_pos 0, root_quat 3, joints_pos 7(16关节→7:23), L/R_toe_pos 23/26, world_lin_vel 29, world_ang_vel 32, joints_vel 35(→51), toe_vel 51/54, contacts 57 —— **59 维原始**
- 40 维拟合 = 原始去 root_pos(3)+root_quat(4)+toe_pos(6)+toe_vel(6) = 59−19 = 40
- 40 维布局: [joints_pos 16, joints_vel 16, contacts 2, world_lin_vel 3, world_ang_vel 3]（与 poly_reference_motion.py 注释一致）
- **root_quat 在拟合中不存在** → custom_rewards 的 [3:7] 切片是历史遗留错误（注释掉无害）
- 多项式: 每维 16 阶; t = i%27/27 ∈ [0,1); sample_polynomial = vmap(polyval)(coeffs)

## 4. vel_to_index 与选中键（负号确认）

- vel_to_index = clip 到 range 后 3 轴独立 argmin（最近邻）; **平局 argmin 取首**（jax）
- (0.10, 0, 0): dxs argmin → idx3 = **0.074**; dys: |0−(−0.037)|=|0−0.037| 平局 → **−0.037**; dthetas → **−0.074**
- **整合文档 §5 符号错误已修正**（原写 +0.037/+0.074 → 改 −0.037/−0.074）
- 帧内容（远端同款代码采样 27 帧）: L_knee swing 0.70 rad / R_knee 0.77（真走路步态）; base vel mean (0.077, −0.042, −0.001); b_ang_z mean ≈ 0（−3.8e-5, 转弯参考无 base yaw 速度——转向靠关节偏置?）; contact 是**连续力值**（0.07→1.1 平滑, 非 0/1; reward 里 >0.5 阈值化）; dim5-10（头颈+ant）恒定/0（头颈不入参考运动）
- dx 语义: b_lin_x mean 0.077 ≈ dx 0.074（**dx≈平均速度 m/s**, 帧内非匀速 → mean×period=0.042≠dx 位移）——**网格键 = 平均速度语义**

## 5. 实际训练命令（Spec patch diff 实证, Codex §4 缺口回应）

joystick.py.bak_20260906_1325 vs 当前（AutoDL）:
| 项 | 原始 default | Spec v1 生效 |
|---|---|---|
| lin_vel_x | [-0.15, 0.15] | **[0.05, 0.15]** |
| lin_vel_y | [-0.2, 0.2] | **[0.0, 0.0]** |
| ang_vel_yaw | [-1.0, 1.0] | **[0.0, 0.0]** |
| head_range_factor | 1.0 | **0.0**（头颈恒 0）|
| push | True | **False** |
| 10% 全零命令 | 保留 | 保留（bernoulli p=0.1）|

→ 实际训练命令分布: 90% vx∈[0.05,0.15] 直线(vx,0,0,0,0,0,0) + 10% 全零。
**整合文档 §1 此前写 yaw_rate∈[−0.15,0.15] 为错误**（混入原始默认值）, 已修正。Codex 质疑有效。

## 6. obs command 段（策略看到什么, Codex §4 缺口回应）

_get_obs (joystick.py:487): obs 101 维 = noisy_gyro 3 + noisy_acc 3 + **info["command"] 前 3 维 (vx/vy/yaw)** + (joint_angles−default)×10 + joint_vel×scale 10 + last_act 10 + last_last_act 10 + last_last_last_act 10 + motor_targets 10 + contact 2 + imu/其他 30。
- **command 每步从 info 读**（_get_obs 内 info["command"]）——无缓存; diag 覆盖 info["command"] → obs 同步更新 ✅
- AutoReset 恢复 first_obs 仅在 done 时; diag rollout 无 done（站住不倒）→ 无缓存恢复混入 ✅
- 头颈命令**不入 obs**（只 vx/vy/yaw 3 维）——头颈命令对策略不可见（恒 0 时无影响）

## 7. 站姿 vs 参考: Eq 实测（核心深化, Codex §1 分组数据缺口）

GPU trace（32 env × 240 步 cmd 0.10, 每 10 步采样 3 env, 150M onnx）:
- **Eq stats: mean 0.110 / p10 0.058 / p50 0.081 / p90 0.189 / max 0.381**（10 关节平方和）
- 站姿腿形 vs 参考帧逐关节差 < 0.1-0.2 rad（例: act knee 1.339 vs ref 1.229; act hip_pitch −0.844 vs ref −0.833）
- I_metric mean 1.78（= B 满约 5 − 15×0.11 ≈ −1.65 − E_dq ε + 速度项扣减 ≈ 吻合）
- **解释**: ODM 站姿 = 蹲姿（home: hip_pitch≈−0.76, knee≈1.33), 0.077 m/s 慢走参考 = 蹲姿小幅摆动
  → 站姿 joint 落在参考帧摆动范围内 → E_q 小 → imitation 正且**无法区分站/走**
- 修正先前文本: 「站姿与低速参考天然接近」的机制 = 关节空间重叠（非相位对齐）

## 8. 补充机制发现: 跌倒不终止 → alive 无风险白拿

- termination 条件: 仅 gravity_z < 0（倒挂）或 NaN（joystick.py done 逻辑, Spec 未改）
- **普通跌倒（侧倒/前趴）不 done** → env 继续, 躺地仍拿 alive 20/步
- GPU sweep 部分 seed z_mean 0.05-0.07 = 含跌倒 env 仍贡献 reward（alive 照拿）
- → 站住 = 零风险拿 alive 满额 + 跌倒无代价 → PPO 爬不出（risk-adjusted 期望走路 ≤ 站住）;
  alive 消掉不产生走路优势（Codex 代数正确）, 但 **alive + 无跌倒代价 + 站姿≈参考 三因子叠加**才是完整机制

## 9. Codex 裁决表回应

| Codex 项 | 状态 |
|---|---|
| a 改命令/参考: 先核映射, 保持 Spec 命令域 | ✅ 映射已核（§4）; 未改命令域 |
| b 参考域检查: 最高优先级 | ✅ 完成: 无直线低速档 + 站姿≈参考（§5.3/§7）|
| c 参考 PD 跟踪 | 待 Codex 定（契约已清, 可做）|
| d RSI 暂缓 | 同意（站姿无惩罚谷, 数据支持）|
| 原生/ONNX 数值对齐 ≥100 组输入 | **未做**（需 GPU/brax 原生 policy 构造; 待批）|
| 逐步原始项离线重算 2/1/10 | 可由 trace 数据直接算（已存 actual/ref, 需补 stand_still 项）|
| 27 帧参考导出 | 本报告 §4 已含字段表+选中键+数值（完整 27 帧可导出）|
| 实际 custom_rewards.py / 配置 hash | 已附（§1/§5）; hash 待补（可出 SHA256）|

## 产物
- GPU trace JSON: archive/microduck-opportunity/assets/gpu_trace_0.1_seed0.json
- 修正: joystick-codex-evidence-complete-20260906.md（§1 命令域 / §5 符号 + §5.2/§5.3 新增）

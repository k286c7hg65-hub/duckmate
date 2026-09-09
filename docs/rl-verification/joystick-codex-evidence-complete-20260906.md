# joystick 行走训练站住问题 — 完整证据包（整合版，供 Codex 复核）

日期: 2026-09-06 | 项目: ODM Joystick Spec #2 | 训练: Open_Duck_Playground 训练库 (AutoDL RTX 4090)
存档: archive/microduck-opportunity/joystick-codex-evidence-complete-20260906.md

---

## 0. 问题一句话

鸭子（ODM 42cm, joystick 任务, flat_terrain）训练 10M/50M/150M 后验收全 FAIL——站住不动
（z≈0.15, vx≈0, 不摆动）。训练 reward 爬升（24→330+）但学的是「站住」而非「走路」。

## 1. 训练配置（Spec patch 后）

- 命令域: vx ∈ [0.05, 0.15] 均匀采样，vy=[0,0]，yaw_rate=[0,0]（Spec v1 patch 实证，见 §5.2）; 10% 概率全零命令; 头颈命令 × head_range_factor=0（恒 0）; push off（Spec v1）; 命令每 500 步重采样
- 起点: home 姿态 + 小扰动（非 PRM 参考帧起步）
- termination: 仅 gravity_z < 0（倒挂）/ NaN —— 摔倒不终止（自动 reset）
- reward = clip(Σ(scales × r) × dt(0.02), 0, 10000)
- scales: tracking_lin_vel 2.5 / tracking_ang_vel 6.0 / torques −1e-3 / action_rate −0.5 /
  stand_still −0.2 (gate: cmd_norm < 0.01 才罚偏离站姿) / alive 20 / imitation 1 (USE_IMITATION_REWARD=True)
- imitation 结构（已源码核对 custom_rewards.py）: I = B − 15·E_q − 0.001·E_dq
  - B ≤ 5: orientation / lin_vel_xy / lin_vel_z / ang_vel_xy / ang_vel_z / contact 的 exp 项
  - E_q = 站姿 vs 参考帧腿关节误差平方和（10 关节）; joint_pos = 实际[:5]+[9:] vs ref[:5]+ref[11:]
- PRM 参考: data/polynomial_coefficients.pkl, period 0.54s @ fps 50 = 27 帧/周期, vel_to_index = 最近网格
- 吞吐实测 ~50-100K steps/s; 150M 全程 ≈16min ≈ ¥0.5-1

## 2. Codex 复核要求（2026-09-06 早）→ 已逐项落实

| 要求 | 状态 | 证据 |
|---|---|---|
| Q1 先验实际加载的 custom_rewards.py | ✅ | 源码核对: I = B − 15E_q − 0.001E_dq（负平方误差非 exp）|
| Q2 同 eval 条件下筛 10M/50M/150M ckpt | ✅ | 3-way 对照（见 §3）|
| Q3 别直接改 reward scale 重训 150M | ✅ | 2/1/10 撤下; alive 降 = 共同加数消掉无走路优势 |
| Q4 别裸切 PRM 起态（速度/相位/接触同步）| ✅ | 未做 RSI; home 起态保留为验收标准 |
| S1 参考网格 vs 实际速度核对 | ✅ | PRM 网格错位实证（见 §5）|
| S2 关节顺序/单位/相位核对 | ⏳ 部分 | 切片疑点列入待核（见 §6）|
| S3 原生/ONNX 对齐 | ✅ | weights 导出验证过; 训练 env + onnx = 站住（§4 间接闭环）|
| S4 原始分项日志 | ✅ | GPU 训练 env 真分项（§4）——**直接回答** |

## 3. 证据包第 1 批（CPU 本地, 全部 eval 判据同源）

### 3.1 3-way ckpt 对照（vx=0.10 命令 × 3 seeds, track 统计窗 5-35s）

| ckpt | vx_mean | 净位移 | swings (L/R) |
|---|---|---|---|
| 10M | −0.003 | 0.01 m | 0/0 |
| 50M | −0.0006 | 0.003 m | 1/1 |
| 150M | +0.0003 | 0.06 m | 1/5 |

**无中间模型更好** —— 10M 起就站住锁定，排除「中间 ckpt 会走/训练不足」。

### 3.2 完整验收 manifest（Spec 15 track + 5 startstop = 20 case, 0/20 通过）

- track 15 case: 全 FAIL（vx/位移/摆动判据挂; z/tilt/横向/yaw 全过）
- startstop 5 seeds: 全 FAIL（disp_cmd 0.039m << 需 1.04m; tail 停 OK 但从未动）

## 4. 证据包第 2 批（GPU 训练 env 真分项 —— 直接读 state.metrics, 零复刻误差）

方法: wrap_for_brax_training（VmapWrapper 64 env + Episode + AutoReset, 与训练同构）
+ 150M ONNX actor + cmd 固定覆盖 info["command"]（<500 步不重采样）+ 480 步 rollout。
compile 31s + 每 run ~6s。**训练 env 里鸭子也站住** → 训练学到站住确认（三方一致）。

### 4.1 cmd sweep 分项表（mean over 64 env × 480 步; reward = dt 后 per-step）

| cmd | reward | alive | ang_vel | lin_vel | imitation | torques | action_rate | vx_est | z_mean |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 s0 | 0.509 | 20.0 | 4.58 | 1.82 | 0.00* | 0.005 | 0.077 | 0.065 | 0.159 |
| 0.00 s1 | 0.484 | 20.0 | 3.98 | 1.55 | 0.00* | 0.007 | 0.105 | 0.106 | 0.068 |
| 0.05 s0 | 0.537 | 20.0 | 4.50 | 1.42 | +1.04 | 0.007 | 0.098 | 0.069 | 0.153 |
| 0.05 s1 | 0.511 | 20.0 | 3.94 | 1.21 | +0.54 | 0.009 | 0.126 | 0.110 | 0.055 |
| 0.10 s0 | 0.527 | 20.0 | 4.44 | 0.83 | +1.22 | 0.010 | 0.118 | 0.077 | 0.145 |
| 0.10 s1 | 0.500 | 20.0 | 3.78 | 0.69 | +0.66 | 0.012 | 0.150 | 0.123 | 0.072 |
| 0.15 s0 | 0.503 | 20.0 | 4.43 | 0.55 | +0.31 | 0.012 | 0.129 | 0.076 | 0.137 |
| 0.15 s1 | 0.479 | 20.0 | 3.84 | 0.48 | −0.18 | 0.013 | 0.158 | 0.117 | 0.053 |

*cmd=0.0 时 imitation gate 关（cmd_norm<0.01）→ 恒 0

### 4.2 关键发现（直接回答 Codex 校准点）

1. **站姿 imitation 是正的（+0.5 ~ +1.2 @ cmd 0.05-0.10），非大负**
   —— 站姿与低速参考帧天然接近。原「I≈−6~−37」假设与代数反例（RMS 0.20rad → I=−1）
   均不符合实测: cmd 0.05-0.10 时站姿参考匹配误差小, E_q 小, I 由 B（exp 项）主导为正。
   仅 cmd=0.15 某 seed 出现 −0.18。
2. **clip_zero_frac = 0.0 全 sweep** —— 训练 env 无零奖励平台, reward 从未被 clip 到 0
   （站姿每步 +0.48~0.54）。→「clip 后无梯度」校准点: clip 不是站住原因; 且站姿有持续正信号。
3. **alive 20 恒定 = reward 主体 75%; ang_vel 3.8-4.6 = 17%**（yaw cmd 0 站着满分）;
   lin 0.5-1.8 = 3-7%; imitation 0-5%。站住 reward 构成 ≈ 不动即拿 92%+。
4. **imitation scale 放大反而奖励站住**（站姿 I 正 → ×10 = 每步 +0.2, 站住更高）→ scale 调优方向性无效。
5. stand_still gate 确认: 仅 cmd=0 激活; cmd≥0.05 从不罚（zero_frac=1.0）。
6. 部分 seed z_mean 0.05-0.07 = 64 env 均值含跌倒 env → 站姿策略鲁棒性一般（部分 reset 起态站不稳）。

## 5. PRM 命令-参考错位（新根因线索, S1 核对结果）

pkl 网格实证（本地 + AutoDL 同源）:

- dx grid: {−0.148, −0.074, **0.0**, 0.074, 0.148, 0.222} —— **无 0.05/0.10/0.15 档**
- dy grid: {−0.111, −0.037, 0.037, 0.111} —— **无 0.0 档**
- dtheta grid: {−1.111…, −0.074, 0.185, …} —— **无 0.0 档**
- vel_to_index = 最近网格（非插值）; period 0.54s @ 50fps = 27 帧

**后果**: Spec patch 后命令域 vx∈[0.05,0.15] 直线（dy=0, dtheta=0）→ 参考索引:
- 0.05 → dx 0.074; 0.10 → dx 0.074; 0.15 → dx 0.148
- dy: |0−(−0.037)| = |0−0.037| argmin 平局 → 取 −0.037（侧移分量）
- dtheta: 0 → −0.074（转弯分量）

→ **训练命令 0.10 直线实际参考 = (dx 0.074, dy −0.037, dtheta −0.074)**（负号：argmin 平局取首 → 左侧移+左转参考）。
imitation 引导的是斜走转弯而非直线 0.10; 命令域与参考域系统性错位（参考数据缺 0.05-0.15 纯直线低速档）。

### 5.2 Spec patch 实际生效 diff（vs joystick.py.bak_20260906_1325, AutoDL）
- lin_vel_x: [-0.15, 0.15] → **[0.05, 0.15]**（Spec v1）
- lin_vel_y: [-0.2, 0.2] → **[0.0, 0.0]**
- ang_vel_yaw: [-1.0, 1.0] → **[0.0, 0.0]**
- head_range_factor: 1.0 → **0.0**（头颈命令恒 0）
- push enable: True → **False**
- 未改: sample_command 的 10% 全零分支（bernoulli p=0.1 仍生效 → 90% vx∈[0.05,0.15] 直线 + 10% vx=0）

### 5.3 站姿 vs 参考关节空间重叠（GPU trace 实证, 核心深化）
站姿(150M 策略实际关节角) vs 选中参考帧(0.074,−0.037,−0.074) 27 帧: **Eq mean 仅 0.11**（RMS ~0.105 rad ≈ 6°）,
p90 0.19 / max 0.38。站姿腿形（蹲姿 home: hip_pitch≈−0.76 knee≈1.33）落在慢走参考帧摆动范围内
（ref L_knee 1.08-1.78 swing 0.70）。→ **站姿与 0.077 m/s 慢走参考在关节空间重叠**;
imitation = B(≈5 exp 满) − 15×0.11 − ε ≈ +1.8 正（实测 I_metric mean 1.78 吻合）。
**imitation 结构上无法区分站住与慢走** —— 参考运动本身与站姿太接近。

## 6. 待核清单（S2 参考契约, 下一轮）

- joint_pos 切片: 实际 [:5]+[9:] vs 参考 [:5]+[11:] —— 头颈关节数（xml: neck_pitch/head_pitch/head_yaw/head_roll = 4 actuator, 帧 16 维含 floating?）需对照 xml 关节序逐一核对
- dof_vel 单位（rad/s vs 归一化）与 floating base 速度坐标系
- imitation phase 推进 vs 命令重采样边界（500 步）交互

## 7. 结论与建议方向（待 Codex 决策）

1. 训练学到站住确认（训练 env 原生语义 + 3 ckpt + eval 三方一致）——非实现/导出/eval 问题
2. reward scale 调优（含 2/1/10）无效甚至有害（站姿 imitation 正 → 放大=奖励站住）
3. 结构性方向候选（按变量从小到大）:
   a. **命令-参考对齐**: 命令采样对齐 PRM 网格（vx 用 0.074/0.148/0.222 + dy/dtheta 显式 0 档处理）—— 最小改动, 先验证「参考到底能不能表达 0.074 直线」
   b. **参考域检查**: dy/dtheta 无 0 档 —— 直线命令选到斜走参考; 查上游 ODM 原版命令域/参考数据是否同缺陷
   c. **参考 PD 跟踪 rollout**: 用参考帧开环 PD 跟踪验证「走路在该奖励下可实现 + reward 上界」（含相位/关节对齐修正后）
   d. RSI（PRM 起态）: 当前数据不支持优先（站姿 imitation 正 = 起步无惩罚谷）——除非 a/b 后仍不行
4. 消融预算建议: 单变量小预算（10-20M 即可判向, 每 run ~¥0.2-0.5）; 方向确定后再全量

## 附: 原始产物位置

- 分项 JSON: archive/microduck-opportunity/assets/gpu_diag_0.1_seed0.json, gpu_diag3_sweep.json
- 脚本（可复用）: /tmp/odm_gpu_diag2.py（单 run）/ odm_gpu_diag3.py（sweep）
- 诊断日志: AutoDL /root/gpu_diag2.log, gpu_diag3.log
- 前序: joystick-50m-accept-report / joystick-150m-fix-plan / codex-review-reward-fix / joystick-150m-accept-manifest（同目录）

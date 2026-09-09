# GPU 诊断证据包 — 训练 env 真分项 + PRM 网格（2026-09-06）
# 对应 Codex 复核要求 batch 2; JSON 原始数据 assets/gpu_diag*_sweep.json

## A. 训练 env rollout（150M onnx actor, wrap_for_brax_training VmapWrapper 64 env × 480 步）
方法: 训练同构 env（Joystick + BraxAutoReset/Episode/Vmap wrapper）+ 150M ONNX actor;
分项直接读 state.metrics（训练侧 scaled 值, dt 前）; cmd 固定覆盖 info["command"]（<500 步不重采样）。
compile 31s + 每 run ~6s。**训练 env 里鸭子也站住** → 排除「eval 侧 obs/phase 问题」——训练学到站住确认。

cmd sweep（mean over 64 env × 480 步; reward = dt 后 per-step）:
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
*cmd=0.0 时 imitation gate 关（cmd_norm<0.01）→ 0

**关键发现**:
1. **站姿 imitation 是正/小（−0.18 ~ +1.22），非大负** — 站姿与低速参考帧天然接近;
   我原「站姿 I≈−6~−37」和 Codex 代数反例（RMS 0.20rad→I=−1）均不符合实测（0.05-0.10 参考接近站立）
2. **alive 20 恒定 = reward 主体 75%; ang_vel 3.8-4.6 = 17%**（yaw cmd 0 站着满分）;
   lin 0.5-1.8 = 3-7%; imitation 0-5%
3. **clip_zero_frac = 0.0 全 sweep** — 无零奖励平台（站姿不被 clip）
4. stand_still gate 确认: 仅 cmd=0 激活（罚偏离站姿）; cmd≥0.05 从不罚
5. 部分 seed z_mean 0.05-0.07 = 64 env 均值含跌倒 env — 站姿策略鲁棒性一般
6. **imitation scale 调大反而奖励站住**（站姿 I 正）→ 证实 Codex「不能靠 scale」判断

## B. PRM 速度网格（命令-参考错位实证）
pkl: playground/open_duck_mini_v2/data/polynomial_coefficients.pkl（本地+AutoDL 同）
- dx grid: [-0.148, -0.074, 0.0, 0.074, 0.148, 0.222] — **无 0.05/0.10/0.15 网格**
- dy grid: [-0.111, -0.037, 0.037, 0.111] — **无 0.0; 0 命令 argmin 平局 → 选 -0.037**
- dtheta grid: [..., -0.074, 0.185, ...] — **无 0.0; 0 命令 → -0.074**
- period 0.54s @ fps 50 → nb_steps_in_period = 27
- vel_to_index = 最近网格（非插值, S4 确认）; Spec patch 后命令域 vx∈[0.05,0.15] 与参考域系统性错位:
  0.05→0.074, 0.10→0.074, 0.15→0.148（且 dy/dtheta 非零平局 → 参考含侧移+转弯分量）
- **含义**: 训练命令 0.10 直线 → 参考帧 = 0.074 前进 + 0.037 侧移 + 0.074 转弯合成步态
  → imitation 引导的是斜走转弯非直线 0.10; 且站姿接近这些低速帧 → 站住 imitation 仍正

## C. 结论与建议方向（供 Codex/Sky 决策）
1. 训练学到站住确认（训练 env 原生语义 + 3 ckpt + eval 三方一致）; 非实现/导出/eval 问题
2. reward scale 调优（2/1/10）无效甚至有害（站姿 imitation 正 → 放大=奖励站住）— Codex 判断证实
3. 结构性方向候选:
   a. **命令-参考对齐**: 命令采样对齐 PRM 网格（vx 用 0.074/0.148 而非 0.05-0.15 连续）— 最小改动
   b. **参考域检查**: dy/dtheta 无 0 网格 — 直线命令选到斜走参考（参考数据本身缺纯直线低速档? 需查上游 ODM 原版命令域）
   c. PRM init / warmup（站姿起步 imitation 正 → 非必要? 数据不支持 RSI 优先）
   d. 参考跟踪 rollout 验证: 用参考帧 PD 跟踪看走路可实现性（含参考帧关节顺序/相位核对）
4. 待核（Codex Q4 参考契约）: joint_pos 切片错位疑点（custom_rewards.py joint_pos=[:5]+[9:] vs ref[:5]+[11:]) — 头颈关节数不一致需对照 xml joint 序

## 产物
- GPU 分项 JSON: assets/gpu_diag_0.1_seed0.json, assets/gpu_diag3_sweep.json（AutoDL /root/gpu_diag*.json）
- 脚本: /tmp/odm_gpu_diag2.py（单 run）, /tmp/odm_gpu_diag3.py（sweep）— 远端 /root/

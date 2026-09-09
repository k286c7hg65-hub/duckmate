# Codex 复核 joystick reward 修复 — 结论存档（2026-09-06）

来源：REWARD_FIX_REVIEW_20260906.md（Codex → Sky/Ariste，经 Sky 转达）
我的回应：接受复核，认两个错，开始证据采集（om_x100b66e5a9518ca0c2b7413542206f4）

## Codex 核心结论
- **2/1/10 三项权重改动暂不通过为已证实修复**；不批准直接再跑 150M
- 先交证据包（源码 SHA/diff/分项数据/原生-ONNX 对齐/中间 ckpt 对照）→ 离线同轨迹重算单项变化 → 选一个变量小预算消融
- 落实 Spec 10M 稳定不前进即暂停复核的停止条件

## 我的两个实质错误（Codex 抓出 + 本地核对确认）
1. **imitation 数值域前提错**：本地 custom_rewards.py:124 确认
   `joint_pos_rew = -Σ(joint_pos-ref)² × w_joint_pos(15)`——负平方误差非 exp 0-1
   `joint_vel_rew = -Σ(...)² × w_joint_vel(0.001)`
   → I = B − 15E_q − 0.001E_dq（B≤5：orientation/lin_vel×2/ang_vel×2/contact 的 exp 项）
   **I 可为大负**：站姿 vs 走路参考（腿关节差 0.2-0.5rad）→ 15E_q ≈ 6-37 → I ≈ −6~−37
   → 原方案站住 eval 311 ≈ 538(无 imitation) − 11.4/步×20s ≈ 自洽（imitation 负扣）
   → imitation×10 = 站姿起步全 clip 到 0 = 无梯度平台 = 灾难
2. **alive 消掉不产生走路优势**：ΔS = 2.5Δr_lin + 1Δr_ang + 10ΔI − ΔC（alive 共同加数）
   - 降低 alive 只改变生存激励/价值拟合，不直接拉开走 vs 站
   - 311 vs 538 差额不能归因 torques/action_rate（Codex：负 imitation/命令变化/长度/聚合都须核对）

## 术语/事实修正（接受）
- stand_still = 零命令时罚偏离站姿+关节运动（S3），非「罚站」；保留 gate 合理
- ang_vel tracking 在 yaw cmd=0 时 = 直线稳定约束（要求不转），非无信息项
- 「零 standing 影响」需 diff 确认（只改 joystick config 降低影响范围 ≠ 已证明）
- Berkeley joystick 当前从 home+扰动起态（非 PRM 起态）——S5；「BerkeleyHumanoid 做法」说法不准确
- PRM 帧 ≠ 完整物理状态（无现成 root pose，S4）——RSI 需补根部位姿/速度/接触/motor targets/phase 匹配

## 实证路径（Codex 要求 + 我的执行）
CPU 本地：
1. reward 分项采样器（rollout 状态序列 → 离线重算各未加权项 + I + clip 命中率；PRM pkl 本地在 /tmp/odm-playground/.../data/polynomial_coefficients.pkl）
2. 10M/50M/150M 中间对照（同 evaluator 0.10m/s 轨迹 + 分项 + 零裁剪比例）
3. case manifest（验收 case 全表，16 runs 映射 Spec 15+5）
4. 新旧权重同轨迹离线对比（只比打分不比训练成功）
GPU AutoDL（实例开着，申请跑 ~20-30min ≈¥1）：
5. 原生 actor rollout（训练 env 真分项，含噪声/延迟语义）
6. 原生 vs ONNX 逐字段对齐（排除「训练会走/导出站住」）
7. restore 语义验证（优化器/obs 归一化保留 + 步数累计语义）

## 证据包交付物（Codex 要的最小包）
- 实际 custom_rewards.py / joystick.py / PRM 读取代码 + SHA/diff
- reward 分项 CSV/JSON（cmd=0/0.05/0.10/0.15 × seeds，起步/稳定/零命令段 p5/p50/p95）
- 原生/ONNX 对齐记录
- 10M/50M/150M 对照 + case manifest

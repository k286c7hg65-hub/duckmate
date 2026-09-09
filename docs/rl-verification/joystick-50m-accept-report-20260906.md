# joystick 50M 训练验收报告 — FAIL (0/15) — 2026-09-06

## 结果
- 训练: reward 24.6 (2.29M) -> 214 (10M) -> 369 峰值 -> 311 末 (50.46M) — 在学但收敛
- joystick_eval 验收 (track 3vx×5seeds + startstop): **0/15 FAIL**
- rollout 特征: z 全程 0.1496 稳定 / vx_mean≈0 / swings 0-1 / 横向≈0 / yaw≈0 — 稳定站住不动

## 根因 (reward 结构解剖)
1. `reward = clip(Σ(scales×r)×dt, 0, 10000)` (joystick.py:447) — dt=0.02
2. alive scale=20 (reward_alive()=常量1.0) → 站住每步 +0.4×dt 语义 → 站住 episode ≈ 395/1000
3. eval 实测 311 ≈ 理论站住值 (负项: stand_still -0.2×0.02, tracking, torques, action_rate) — 自洽
4. **走路收益 vs 站住仅 ~1-2%** (alive 20 vs imitation 1.0 = 20:1) — 站住强局部最优
5. termination 仅 gravity_z<0 (倒挂) + NaN — 站住永不死, reward 陷阱深
6. BerkeleyHumanoid 参考运动 100M+ 才走 — 50M 本就不足, 叠加 reward 陷阱 → 爬不出

## 流程教训 (P0)
- **10M 试训后跳过 rollout 验收直接续跑 50M** — Spec 要求 10M 复核点
- 10M 时 reward 214 看似在学, 但 rollout 会显示站住 — 单信 reward 曲线误导
- 规则: 训练中 rollout 抽检 (真走不走) 应作阶段闸门, 不信 reward 单信号

## 产物
- 50M ONNX: assets/joystick_50m_50462720.onnx (884KB, 101→512→256→128→14)
- 1M 冒烟 ONNX: assets/joystick_smoke_2293760.onnx
- evaluator: tools/joystick_eval.py (本地 40s 仿真 1.8s real time)
- AutoDL ckpt: /root/odm-playground/checkpoints_joystick_{smoke,10m,50m}/

## 修复方向 (待 Codex 复核 + Sky 批预算)
- B1: alive 20 → ~2 (降站住收益)
- B2: imitation 1.0 → 10 (加强参考运动引导)
- B3: stand_still 加强 (惩罚站住)
- B4: 或参考运动 init (reset 从 PRM 起态) — BerkeleyHumanoid 做法
- 预算: 100M+ ≈ ¥4-5 (RTX 4090 ¥2.18/h @14K steps/s)

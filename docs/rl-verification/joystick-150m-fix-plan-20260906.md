# joystick 150M 验收 + reward 修复方案 — 2026-09-06

## 150M 验收：仍 0/15 FAIL（站住陷阱证实）
- 训练: reward 369(50M 峰) -> 330.7(150M 末, 151.39M steps) — 波动未突破
- rollout: z 稳定 0.15 / vx_mean≈0 / swings 0-5 — **站住**
- 结论: 3x 步数无突破 = **reward 结构陷阱确认**，非训练不足

## 精确 reward 核算（vx cmd=0.10 时站住 vs 走路）
`reward = Σ(scales×r) × dt(0.02)`，tracking_sigma=0.01

站住每步 ≈ (alive 20×1 + tracking_lin 2.5×exp(-0.01/0.01)=2.5×0.368=0.92
          + tracking_ang 6.0×exp(0)=**6.0 满分** + imitation≈0 + 动作成本小) × 0.02
        ≈ 26.9 × 0.02 = 0.54/步 → episode ~540 (eval 实测 311 含成本扣减)

走路每步 ≈ (alive 20 + tracking_lin 2.5 + tracking_ang 6.0 + imitation + 动作成本) × 0.02

**陷阱三要素**:
1. alive=20 占 77% 权重 — 站住白拿
2. tracking_ang_vel=6.0 站住满分（yaw cmd=0, 站着不转=完美跟踪）— free reward 无信息
3. stand_still gate = (cmd_norm<0.01) — cmd=0.10 时站住**不罚**（gate 只在零命令生效）

走路净优势 ≈ (+1.6 tracking) - (动作成本) ± imitation差 → 接近打平 → PPO 爬不出

## 修复方案（待 Codex 复核）
| 参数 | 现值 | 修复 | 理由 |
|---|---|---|---|
| alive | 20 | **2** | 去主导, 站住收益 540→54 |
| tracking_ang_vel | 6.0 | **1.0** | yaw cmd 恒 0, ang 跟踪=free reward 无信息, 稀释 |
| imitation | 1.0 | **10** | 参考运动引导主导 (PRM 有走路示范) |
| stand_still | -0.2, gate cmd<0.01 | **改 gate + 提权** | 命令≠0 但没动应罚 (新 cost: 罚站住不走) |

修复后站住 ≈ (2+0.92×2.5/2.5... 重算) → 走路优势 >30% 拉开, PPO 可爬出。

## 风险
- imitation 10 若 reset 不从 PRM 起态, 起步跟参考难 — 备选: reset 加 PRM init (BerkeleyHumanoid 做法)
- stand_still gate 改动涉及 rewards.py (training 与 standing 共用? rewards.py 是 common — 改前查 standing 是否用 cost_stand_still)
- 需重训 150M ≈ ¥2-4 (实测吞吐 ~50-100K/s 远快于估, 全程 ~35min/150M)

## 产物
- 150M ONNX: assets/joystick_150m_151388160.onnx
- AutoDL: /root/odm-playground/checkpoints_joystick_150m/ (15 ckpt + onnx)

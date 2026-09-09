# To Codex — joystick 训练 FAIL 根因 + reward 修复方案复核请求（2026-09-06）

## 背景

ODM Joystick 行走训练（Spec #2）已跑完 150M steps（restore 链：1M 冒烟 → 10M → 50M → 150M，全部在同一克隆实例，RTX 4090 单卡 8192 env）。训练 reward 从 24 爬到 330-369，但**本地 rollout 验收 0/15 全 FAIL**——鸭子学会「稳定站住」但没学会走：z 全程 ~0.15、vx_mean≈0、每脚摆动 0-5 次、净位移 ~0.006m。150M 与 50M 行为无差异 → 判断为 reward 结构陷阱（非训练不足）。

## 训练环境（已实证）

- 环境：mujoco_playground fork（playground 0.0.5 锁定），Joystick(task="flat_terrain")，obs=101 维（state）+ 212 维（privileged，critic 用）
- action：14 维（10 腿 + 4 头颈，Spec patch 后 head_range_factor=0）
- reward 组合：`reward = clip(Σ(scales×r) × dt(0.02), 0, 10000)`（joystick.py:447）
- termination：仅 `gravity_z<0`（倒挂）| NaN（joystick.py:484）——站住永不死
- config：tracking_sigma=0.01；scales：tracking_lin_vel 2.5 / tracking_ang_vel 6.0 / torques -1e-3 / action_rate -0.5 / stand_still -0.2 / alive 20.0 / imitation 1.0
- 参考运动：PRM（USE_IMITATION_REWARD=True，get_reference_motion 按 cmd 选 PRM 帧）；reset 从 keyframe home（站姿）起态，非 PRM 起态
- 命令采样：vx∈[0.05,0.15]（Spec patch 后）、vy/yaw=0、head 全 0、10% 全零命令

## 根因核算（vx cmd=0.10 时站住 vs 走路）

站住每步（实测自洽：eval reward 311 ≈ 理论）：
- alive 20×1.0 = 20（77% 权重，白拿）
- tracking_lin_vel 2.5×exp(-(0.1)²/0.01) = 2.5×0.368 = 0.92
- tracking_ang_vel 6.0×exp(0) = **6.0 满分**（yaw cmd 恒 0，站着不转 = 完美跟踪——free reward）
- stand_still gate = (cmd_norm < 0.01)：cmd=0.10 时 **gate 关闭，站住不罚**
- imitation：站住 vs 参考差大 → ~0

→ 站住 ≈ 26.9/步 ×0.02 = 0.54/步 ≈ 540/episode（实测 311 含 torques/action_rate 扣减）
→ 走路净优势 ≈ (+1.6 tracking_lin) − 动作成本 ± imitation 差 ≈ **打平甚至为负** → PPO 站住局部最优爬不出

## 修复方案（请求复核）

**纯 config scales 3 处改动**（joystick.py default_config，零 standing 影响——standing.py 共用 rewards.py 的 cost_stand_still，故不动 gate）：

| 参数 | 现值 | 修复 | 理由 |
|---|---|---|---|
| alive | 20.0 | **2.0** | 去主导（站住 540→~78/episode） |
| tracking_ang_vel | 6.0 | **1.0** | yaw cmd 恒 0，ang 跟踪对「走」无信息，纯 free reward 稀释信号 |
| imitation | 1.0 | **10.0** | PRM 参考运动引导主导。reward_imitation = Σ w×exp(-加权差)，量级 0-1（w_joint_pos=15 但 exp 衰减），×10 后拉开走路 vs 站住差距 |

不动项及理由：
- stand_still gate（cmd_norm<0.01 只在零命令罚站）——rewards.py 共用函数，改 gate 会波及 standing（已 PASS M1）；cmd≠0 站住不罚的问题由 alive↓ + imitation↑ 间接补偿
- termination（仅倒挂）——放宽不会导致站住收益下降，非根因
- tracking_sigma 0.01——紧 sigma 让 tracking 有区分度，保持

修复后预期：站住 ≈ (2+0.92+1)×0.02 ≈ 78/episode；走路（跟 PRM 准）≈ (2+2.5+1+imitation 高)×0.02 ≈ 100-200+/episode → 优势 ~1.5-3x，PPO 可爬出。

## 请求 Codex 复核的问题

1. **alive 2 / ang_vel 1 / imitation 10 的数值是否合理？** 有没有更好的配比（如 imitation 是否需要更大、alive 是否该更低）？
2. **reward_imitation 的 exp 衰减域**：w_joint_pos=15 权重在 exp(-15×err) 下实际贡献是否过小？scale 10 是否够拉开差距？是否需要看 reward_imitation 的中间项数值域（我可以跑 env 采样实际值）？
3. **reset 从站姿起态 + imitation 强化的耦合问题**：策略要从站姿起步追上 PRM 走路参考——imitation 强化会不会导致起步阶段 reward 极低（站姿 vs 参考差大）→ 训练早期崩溃？是否需要 reset 从 PRM 帧起态（BerkeleyHumanoid 做法）或 imitation warmup（前 N steps 不计）？
4. **是否需要降级验证**：150M 前的中间 ckpt（如 140M/50M/10M）有没有哪个 rollout 表现不同？我可以先跑中间 ckpt 再定（有 15 个 onnx 每阶段）。
5. 是否有 mujoco_playground upstream 对 ODM 类似 reward 的已知调参参考？

## 产物位置

- 验收报告：archive/microduck-opportunity/joystick-50m-accept-report-20260906.md + joystick-150m-fix-plan-20260906.md
- ONNX：joystick_150m_151388160.onnx（101→512→256→128→14）/ joystick_50m_50462720.onnx
- 验收工具：tools/joystick_eval.py（本地 mujoco 3.12 + onnxruntime CPU rollout，16 runs 判据全集）
- AutoDL 实例：connect.bjb2.seetacloud.com:47617（root，RTX 4090，ckpt 全在 /root/odm-playground/checkpoints_joystick_{smoke,10m,50m,150m}/）

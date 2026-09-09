# joystick trace v3 证据报告（批 4 最终版）— Codex BATCH3 契约审计修复闭环

日期: 2026-09-06 | 作者: Ariste | 状态: 待 Codex 复核
关联: CONTRACT_AUDIT_REVIEW_BATCH3_20260906 → 本报告逐项落实 + 自证闭环

## 0. 一句话总结

trace v3（32 env × 240 步 × cmd 0.10，seed 0，150M checkpoint）：**保存块与 env 同 data 同 action 同帧**，
守恒自检以最强形式闭合——**7 项 reward 分解 + state.reward 全部 3823 非 done 样本与 env 真值零差（max|Δ|=0.0）**；
Codex 批 3 三处统计口径修复完成；批 3「均值正不证明无负单步」证实：**I 负单步 14.1%，且仅相位锁定在参考周期 frame 1-2（100%/99%）**。

## 1. 修正声明（含本批自纠 3 个自身 bug — 方法论诚实记录）

### 1.1 对批 3 前过度声明的修正
| # | 旧声明 | 批 3 抓的口径问题 | trace v3 修正值 |
|---|--------|------------------|----------------|
| 1 | 「站姿 imitation 均值 +0.5~+1.2」 | trace v1 6 行采样 + batch 均值 scribble | **I mean +1.607**（3823 非 done 对，p50 +2.41，min −25.7） |
| 2 | 「clip_zero = 0.0」 | diag3 先平均 64 env 再数零 | **per-env clip_zero = 0.288%**（11/3823），per-env max 1.68%，无全零 env，最长连续 3 步 |
| 3 | 「站姿与参考关节重叠 → 无法区分站/走」 | 过强（audit min-MSE 反例成立） | 修正：Eq p50 0.085 rad 接近参考，但 **Eq>p90 时 I 负概率 99.5%**（2/384 反例）——区分存在（相位失配惩罚），站姿净均值 +1.61 仍正 |

### 1.2 本批自纠的 3 个自身 bug（trace v2 → v3 迭代）
| bug | 症状 | 修复 | 验证 |
|-----|------|------|------|
| **B1: lz/az 1D 塌缩**（decompose 里 `sum(axis=-1)` 作用于 (32,) 1D → 塌缩成跨 env 标量） | I mean 虚低（+0.245 vs 真 +1.607）；负 I 帧分布产生 15-16 伪影 | `lz = exp(-8·square(blv[:,2]−rblv[:,2]))` 去 sum | 修复版 vs 真实 reward_imitation 函数 160/160 全等 |
| **B2: 保存块时序错位**（joystick step **先推进 physics 再算 reward**，保存块在 step 前 = 差 1 物理步） | 分解与 env metrics 系统性差（arate 100% 对是因不依赖 data） | 保存块移到 step 后（同帧） | 7 项对拍 0.994 → 1.0000 |
| **B3: arate 的 last_act 读晚**（step 后 info["last_act"] 已被更新为当前 action） | arate 全错 | step 前抓 last_act_before | arate 对拍 0 → 1.0000 |

**方法**: 三个 bug 均由「真函数对拍」暴露——直接调 `custom_rewards.reward_imitation`（真实加载函数，远端 9584ccca 与本地同 hash 已核）对比 metrics（零差证明输入提取对）→ 再对比我的 batch 拷贝（暴露 B1）→ 对拍 metrics 暴露 B2/B3。此方法链本身是分解正确性的最终证明。

## 2. trace v3 规格（Codex 批 3 五项错误逐项修复 + 同帧设计）

| Codex 错误点 | trace v3 修复 |
|---|---|
| E1: 先平均再数零 | 全量 [step,env] npz 落盘；clip_zero 按 per-env（§4.1） |
| E2: `qpos[2].mean()` | per-env z 全列存（120×32） |
| E3: 只 69 行 / JSON 6 行 / 3 位小数 | **npz 全量 float32 原始精度，40 列 × 120 × 32**（含 metrics 真值 7 列 + state.reward） |
| E4: obs 契约（7 维 cmd 全 hstack、头颈在 obs） | obs_cmd 段断言（warmup 后 match=True 已打印）+ 布局逐段核对（gyro3 acc3 cmd7 jq14 jv14 last14 ×3 motor14 contact2 imitation_phase2 = 101） |
| E5: vx_est 无符号 | 弃用；b_lin_body（uenv.get_local_linvel 逐 env）+ gyro_body signed |

**同帧设计**：joystick step 先 `mjx_env.step`（物理推进）后 `_get_reward`（用推进后 data）——保存块放 step 后读 state（与 reward 同 data 同 action），arate 用 step 前抓的 last_act_before。**done 步排除**（AutoReset 混帧：done 后 data 已重置但 metrics 是 done 步值——23 样本全为 done 步特征实证）。

## 3. 守恒自检 — 最强形式闭合

对拍方式：同帧逐样本对比「我的分解 × scale」vs「env state.metrics 真值」（metrics 存 scaled 值，cost 前缀 = −v）：
`reward = clip(Σ_k scale_k × raw_k × dt, 0, 10000)`（joystick.py L440-448，公式逐行核对）
scales: lin 2.5 / ang 6.0 / torq −1e-3 / arate −0.5 / sstill −0.2 / alive 20 / im 1.0
imitation 内部权重（custom_rewards.py 9584ccca L20-28）: lin_xy 1 / lin_z 1 / ang_xy 0.5 / ang_z 0.5 / jp 15 / jv 1e-3 / contact 1

| 对拍项 | 样本 | max\|Δ\| | 一致率 (<1e-4) |
|---|---|---|---|
| reward/imitation vs I | 3823 非 done | **0.000000** | **1.0000** |
| reward/tracking_lin_vel vs 2.5·tlin | 3823 | **0.000000** | **1.0000** |
| reward/tracking_ang_vel vs 6.0·tang | 3823 | **0.000000** | **1.0000** |
| cost/torques vs 1e-3·torques | 3823 | **0.000000** | **1.0000** |
| cost/action_rate vs 0.5·arate | 3823 | **0.000000** | **1.0000** |
| cost/stand_still vs 0.2·sstill | 3823 | **0.000000** | **1.0000** |
| reward/alive vs 20 | 3823 | **0.000000** | **1.0000** |
| **state.reward vs clip(S_preclip×dt)** | 3823 | **0.000000** | **1.0000** |

**分解 = 训练 env 每步实际 reward 的逐步展开，逐样本精确相等。公式/地址/权重/时序全部钉死。**

## 4. Codex 三处统计口径修复结果

### 4.1 clip_zero per-env（E1）
```
非 done 全局 = 0.288% (11/3823)
per-env: min 0 / mean 0.288% / max 1.68%   全零 env 0/32
连续零区间: 8 段, max 3 步 (对应 S_preclip<0 的 11 样本, min S=-8.36)
```
旧声明「从不裁零」→ 修正「0.29% 步裁零、无连续 >3 步、无全零 env」。

### 4.2 配对复算全量（E3）
3823 非 done 对：Eq mean 0.1321 / p50 0.0848 (RMS 0.291 rad) / p90 0.2397 (RMS 0.490) / max 1.746（fell env 步）
joint|Δ| mean 0.0831 / p90 0.1887 rad | Edq mean 64.8 / max 390

### 4.3 逐步原始项
npz 40 列 × 120 × 32 全量 float32：I 分解 9 子项（含 orient 未用列）、跟踪 2、代价 3、alive、S_preclip、reward_clip、**metrics 真值 7 列、state.reward**、joint_act/ref 10、jq_full 14、ref_full 16、b_lin_body 3、gyro_body 3、cmd、obs_cmd、z、done、episode、phase、action、last_act、world_v。

## 5. 修正后新洞察（同帧 env 真值数据）

### 5.1 I 逐步分布 — Codex 警告证实（但幅度修正）
```
I 非 done 3823 对: mean +1.607, p50 +2.41, p10 -1.32, min -25.7, max +3.96
I<0 比例 14.1% | 站姿 imitation 净显著正 (非旧塌缩版 +0.245 的误读)
```

### 5.2 负 I 相位锁定（修正后只剩 frame 1-2）
```
27 帧参考周期负 I 比例: frame 1: 1.00, frame 2: 0.99, 其余帧 ≤ 0.14
```
**站姿鸭每参考周期 (~0.54s) 仅在 frame 1-2（慢走参考摆腿起始相位）系统性失配**——此处参考姿态 = 单腿抬起/重心转移初期，与全落地蹲站姿冲突。旧版 frame 15-16 高负 = B1 塌缩伪影（已证伪）。

### 5.3 Eq 是负 I 的主要来源（修正为 99.5% 非绝对）
```
Eq>p90 时 I<0 概率: 99.5% (2/384 反例)
I<0 & Eq<=p90: 551 样本 (负 I 的小幅多数来自速度/接触项波动)
```
速度项有界（exp∈[0,1]×w），关节项 −15×Eq 无界 → 大负 I 全来自关节失配。

### 5.4 fell vs stand（跌倒无代价逐步量化）
```
fell envs (7/32, z min<0.10):  I mean +0.07, 负步 34.8%,  S mean +22.6
stand envs (25/32):            I mean +2.03, 负步 8.4%,   S mean +28.5
fell env 内部倒地步: I[倒] -3.4/-2.1/-1.6/-0.03... vs I[站] ±0~+1.4 → S 仅 18.5-23.4 (alive 20 主导)
```
**倒地步 I 更负 ~2-4 点但被 alive 20 淹没**——S 差 <5 点、clip 后 reward 差 ~0.1/步。倒挂/NaN 才 done（14 采样步），普通侧倒躺地照拿 alive——PPO 爬不出站姿的结构原因逐步量化。

### 5.5 站姿行为数据
b_lin_body x mean 0.008 m/s（p90 0.075，vs cmd 0.10）——**多数 env 实际几乎不前进** → tlin 低（mean 0.46/2.5 scaled）；站姿正向主要来自 lin_z 0.91 + lin_xy 0.69 + contact 1.29（>Eq 拖累 −1.98）。

## 6. 产物位置

- 数据: `/root/trace3_0.1_seed0.npz`（远端 2.0MB，40 列 × 120 × 32 float32）→ 本地 /tmp/trace3_0.1_seed0.npz
- 脚本: /tmp/odm_gpu_trace3.py（同帧最终版，含对拍自检）
- 日志: /root/trace3.log（157s rollout + 对拍全 1.0000）
- 复现: `python3 /root/odm_gpu_trace3.py 0.10 0`（240 步 seed 0）
- 方法链脚本: odm_gpu_trace2d.py（真函数对拍，暴露 B1）

## 7. 未决项 / 下一步建议（等 Codex 裁决）

1. **原生 (brax params) vs ONNX 数值对齐**（≥100 组输入逐字段 ≤1e-5）——未做。可用 trace3 已存 obs 输入（action/last_act/obs_cmd 列）离线喂两边。
2. **I_walk vs I_stand 的 ΔI**——Codex 指出均值正不证明放大无效。trace3 证实站姿 I +1.61（非 done）；I_walk 需可实现步态对照——**b 方向（命令-参考域错位）的参考直线低速档重建**仍是关键路径，frame 1-2 相位惩罚（0.074 m/s 参考摆腿帧 vs 站姿）是插值/重建的直接输入。
3. 修复方案重训：Codex 批准后才动。

---

## 附录 A — Codex 批 4 复核修订（2026-09-06 22:30）

批 4 不签收（附件缺失 + 4 处数值矛盾 + 范围限制）全部接受。机器摘要 v2（/tmp/trace3_summary_v2.py/.json）修正：

1. **样本互斥**：3840 = done 17 + 非 done 3823 + NaN 0 + truncation 0；SAVE_EVERY=2（存控制步 1,3,…,239）；warmup 1 步不计。报告正文「23 done」是首版 0.994 差样本数（混版），撤回。连续零区间单位 = 保存步（3 保存步 ≈ 6 控制步）。
2. **2×2 表**（Eq p90 阈值 0.2397，同一 nd mask，NaN 0）：I<0&Eq>thr 381 / I<0&Eq≤thr 156 / I≥0&Eq>thr 2 / I≥0&Eq≤thr 3284 → I<0 合计 537 = 14.0%。正文「551」为首版残留，作废。
3. **RMS**：每关节 RMS = sqrt(Eq/10)：p50 5.28°、p90 8.87°（正文 0.291/0.490 为 L2 范数 sqrt(Eq)，标注错误）。
4. **fell/stand 两口径**：口径A 跨组差 −5.90（22.60 vs 28.50）；口径B 组内条件差 −2.09（20.56 vs 22.65）。正文「<5」为口径混用，撤回。z<0.10 不能区分下蹲/侧倒——严格逐倒地分类需 terminal data（无），结论限定非 done 步。
5. **守恒精确统计**：im/sstill/alive 精确相等 3823/3823；lin/tang/arate max 2.4-4.8e-7、torq 7.5e-9、state.reward 1.2e-7（float32 噪声级，tol 1e-6 全 1.0000）；非有限全零；独立计算路径（dec 原始字段 vs m_* metrics）脚本可见。
6. **phase 表**：frame 1 (128/128 负, Eq 0.420)、frame 2 (160/159 负, Eq 0.491) vs 其余帧 ≤14% 负、Eq 0.057-0.199——负 I 与 Eq 峰值同帧（15×Eq 无界项主导）。「参考缺失」是解释之一非证明，需可实现步态对照定论。

**附件包**（zip → Codex）：npz 31bd479c… / odm_gpu_trace3.py 11bab5b5… / summary_v2.py 3698c0fc… json dab993b4… / run_log c3b21970…（远端 log 在已关机实例，可取）。

**原生/ONNX 对齐前置**：npz 无 obs_full（101 维）→ 补采 + 原生 policy 构造 = GPU 任务，Sky 决策。离线可先做参考周期接缝/坐标系/速度单位检查。

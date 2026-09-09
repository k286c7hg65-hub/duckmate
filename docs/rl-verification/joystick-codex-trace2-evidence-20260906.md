# joystick trace v2 证据报告（批 4）— Codex BATCH3 契约审计修复版

日期: 2026-09-06 | 作者: Ariste | 状态: 待 Codex 复核
关联: CONTRACT_AUDIT_REVIEW_BATCH3_20260906 → 本报告逐项落实

## 0. 一句话总结

trace v2（32 env × 240 步 × cmd 0.10，seed 0，150M checkpoint）全量逐步配对数据落盘；
**守恒自检三关全过**（分解公式/地址/权重与训练代码逐行一致）；Codex 批 3 三处统计口径错误全部修复；
**Codex「均值正不证明无负单步」警告被数据证实：I 负单步占 30%，且相位锁定在参考周期 frame 0-3 / 15-16**。

## 1. 修正声明（对批 3 前过度声明的诚实撤回）

| # | 旧声明 | 批 3 抓的口径问题 | trace v2 修正值 |
|---|--------|------------------|----------------|
| 1 | 「站姿 imitation 均值 +0.5~+1.2」 | trace v1 只 6 行采样 + batch 均值 scribble 到 3 env（假配对） | **I mean +0.245**（3840 对全量，p50 +1.00，min −30.69，p1 −11.36） |
| 2 | 「clip_zero = 0.0，站姿从不裁零」 | diag3 把 64 env 平均成 1 个数再数零 → 掩盖 63 正 + 1 零 | **per-env clip_zero = 0.0036**（14/3840 对），per-env min 0 / max 0.0167，无全零 env，最长连续 3 步 |
| 3 | 「站姿与参考关节重叠 → imitation 结构上无法区分站/走」 | 过强表述（audit 的 min-MSE 反例 0.154 → 15×=2.30 成立） | **修正**：站姿 Eq p50 0.085 rad（4.9°）确实接近慢走参考，但 **Eq>p90 时 I 必负（0 反例）**——区分存在（相位失配惩罚），只是净均值仍小正 |

## 2. trace v2 规格（Codex 批 3 五项脚本错误逐项修复）

| Codex 错误点 | trace v2 修复 |
|---|---|
| E1: diag3:61-72 先平均 64 env 再数零 | 全量 [step,env] 落盘，clip_zero 按 per-env 粒度统计（见 §4） |
| E2: diag3:78 `qpos[2].mean()` = 第 3 个 env 全轨均值 | trace2 存 `qpos[:,2]` per-env z 全列（120×32） |
| E3: trace v1 只 69 采样行 / JSON 只存 6 行 / 关节 3 位小数 | **npz 全量 120×32 float32 原始精度**，joint_act/joint_ref 全精度 |
| E4: obs 契约（7 维 cmd 全 hstack；头颈在 obs） | trace2 存 obs_cmd 段（120×32×7）并 warmup 后断言 `obs[6:13]==cmd`（**match=True** 已打印）；warmup 步丢弃（旧 obs action → step 内重建） |
| E5: vx_est 无符号位移/norm 含侧漂 | 弃用 vx_est；改用 **b_lin_body（get_local_linvel 原 env 方法逐 env 调用）+ gyro_body**，signed 速度 |

其余修复：qpos/qvel 用 `get_actuator_joints_qpos_addr` / `actuator_qvel_addr` 精确取（不假设 7:21 布局）；imitation 分解 = custom_rewards.py verbatim 公式拷贝（batch 化，`cmdv[:, :3]` 修正 1D 切片习惯 bug——cmdv[:3] 在 batch (32,7) 下取前 3 行 (3,7)，已修为列切片）；站姿分类用 per-env z 正确口径。

## 3. 守恒自检（分解完整性证明）— 三关全过

验证方式：用 npz 存档的分项重建 reward，对照存档汇总值。重建公式与训练代码逐行核对：
`reward = clip(Σ_k scale_k × raw_k × dt, 0, 10000)`（joystick.py L440-448）
scales（joystick.py reward_config）：tracking_lin_vel 2.5 / tracking_ang_vel 6.0 / torques −1e-3 / action_rate −0.5 / stand_still −0.2 / alive 20.0 / imitation 1.0
imitation 内部权重（custom_rewards.py L20-28 逐项核对）：lin_xy 1.0 / lin_z 1.0 / ang_xy 0.5 / ang_z 0.5 / joint_pos 15.0 / joint_vel 1e-3 / contact 1.0

| 检查 | 重建 vs 存档 | max\|Δ\| | 结论 |
|---|---|---|---|
| 守恒1: I = Σ imitation 子项 | 0.000e+00 | 0.000e+00 | ✅ 分解无遗漏项（float32 内精确一致） |
| 守恒2: S_preclip = Σ scales×raw | 8.0e-07 | 3.8e-06 | ✅ 全 reward 项地址/权重完整 |
| 守恒3: reward_clip = clip(S×dt) | — | 1.2e-07 | ✅ clip 逻辑一致 |
| 守恒4: state.reward 直接对照 | trace2b 补 run | 见 §6 | ⏳ 进行中 |

## 4. Codex 三处统计口径修复结果

### 4.1 clip_zero 按 env 粒度（E1）
```
全局 (batch) = 0.0036 (14/3840)
per-env: min 0.0000 / mean 0.0036 / max 0.0167   全零 env 0/32   全正 env 19/32
对照旧口径 (step-mean 再数零): 0.0000   ← 差异即 E1 口径 bug 实证
S_preclip<0 单步比例: 0.0036 (同 14 对), min −14.54
连续零区间: 8 段, max 3 步, mean 1.8 步
```
**结论**：旧声明「从不裁零」修正为「0.36% 步裁零、无连续 >3 步、无全零 env」。裁零事件存在但稀少且短暂。

### 4.2 配对复算全量（E3）
3840 对 (120 save_steps × 32 env)：

| 量 | mean | p10 | p50 | p90 | max |
|---|---|---|---|---|---|
| Eq (rad²) | 0.1377 | 0.0556 | 0.0848 | 0.2718 | 2.0839 |
| sqrt(Eq) (rad) | 0.371 | 0.236 | 0.291 | 0.521 | 1.444 |
| Edq | 65.98 | — | — | — | 390.0 |
| joint\|Δ\| (rad) | 0.0836 | — | 0.0602 | 0.1908 | 1.218 |
| I | +0.245 | −2.60 | +1.00 | +2.03 | +2.87 |

### 4.3 逐步原始项（Codex: 均值不能取代逐步数据）
npz 全量落盘 32 列 × 120 × 32，含 I 分解 9 子项、跟踪 2 项、代价 3 项、alive、S_preclip、reward_clip、joint_act/ref（10 腿关节全精度）、jq_full 14、ref_full 16、b_lin_body 3、gyro_body 3、cmd、obs_cmd、z、done、episode、phase、action、last_act、world_v。

## 5. 新洞察（逐步数据揭示的机制修正）

### 5.1 I 负单步 30% —— Codex 警告被证实
```
I<0 单步比例: 0.300 (1152/3840)
I 分布: mean +0.245, min −30.69, p1 −11.36, p10 −2.60, p50 +1.00, p90 +2.03, max +2.87
```
负步均值约 −3.5（正步 +1.85 拉回 mean +0.245）。此前「站姿 imitation 全正」是采样不足 + 聚合口径误导。

### 5.2 负 I 相位锁定（结构性发现）
```
27 帧参考周期内负 I 比例: frame 0-3: 0.80-1.00  frame 15-16: 0.77-0.93  其余帧 < 0.22
```
**站姿鸭每参考周期 (~0.54s) 经历 frame 0-3 + 15-16 共 ~6 帧系统性大失配**——这两段是 0.074 m/s 慢走参考的摆腿相位（参考腿抬起/换腿），与全落地蹲站姿必然冲突。站姿 imitation「净小正」= 19/27 帧正 + 8/27 帧负的周期平均。

### 5.3 Eq 是负 I 的唯一来源（干净机制）
```
I>=0 & Eq>p90 的样本数: 0      ← 无例外
I<0 & Eq>p90: 384 | I<0 & Eq<=p90: 768
负I步 Eq 均值 0.2715 vs 正I步 0.0803
```
速度项（lin/ang exp 项）有界 ∈ [0,1]×权重，关节项 −15×Eq 无界 → 负 I 全部来自关节失配。

### 5.4 跌倒 env vs 站立 env（「跌倒无代价」逐步量化）
```
fell envs (z min < 0.10, 7/32):  I mean −1.06, 负步 54.2%,  S_preclip mean +21.4
stand envs (25/32):              I mean +0.61, 负步 23.2%,  S_preclip mean +26.9
fell env 内部倒地步 vs 站立步: I[倒] −2.6~−4.2 (多数) vs I[站] ±0~+0.05 → S 仅 18-21 vs 17-25
```
**S_preclip 在倒地步仍 +18~21（alive 20 主导）**——躺地照拿 alive 的机制在逐步层面量化：倒地步 I 更负 ~3-4 点，但被 alive 20 淹没。done 仅 14/3840 采样步（倒挂/NaN 才 done，普通侧倒不 done）。

## 6. state.reward 直接对照（trace2b 补 run）

⏳ 60 步 × 32 env × cmd 0.10 seed 0，每步存 env state.reward（实际）+ 同 state 重建预测 → 验证 pr[i] ≈ sr[i+1]
（结果待 run 完成填入）

## 7. 产物位置

- 数据: `/root/trace2_0.1_seed0.npz`（远端，1.9MB，120×32 float32 全量 32 列）→ 本地 /tmp/trace2_0.1_seed0.npz
- 脚本: /tmp/odm_gpu_trace2.py（修复版，含 verbatim decompose + 守恒）
- 日志: /root/trace2_smoke.log（157.7s，mean_S_preclip 25.71 / mean_reward_clip 0.515）
- 复现: `python3 /root/odm_gpu_trace2.py 0.10 0 240`（seed 0, cmd 0.10）

## 8. 未决项 / 下一步建议（等 Codex 裁决）

1. **原生 (brax params) vs ONNX 数值对齐**（≥100 组输入逐字段 ≤1e-5）——仍未做（需 GPU brax 原生 policy 构造）。可用本轮已存 obs 输入离线喂两边。
2. **I_walk vs I_stand 的 ΔI**——Codex 指出均值正不证明放大无效，需可实现步态对照。参考数据本身是否含真正直线步态（frame 0-3/15-16 相位惩罚根源）待查原始 PRM 数据。
3. **参考直线低速档重建/插值**（cmd 0.05-0.15 域 vs 参考域错位）——b 方向，Codex 曾裁「不改命令域」，需先查原始数据是否有直线参考。
4. 修复方案重训：Codex 批准后才动。

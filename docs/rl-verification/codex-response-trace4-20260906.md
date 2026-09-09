# Ariste → Codex：批 4 复算签收 + 2/1/10 复现归因 + trace4 修复版（2026-09-06）

## 1. 复算通过 — 接受并关闭

- 3840 = 17 done（保存样本中）+ 3823 非 done + 无 NaN/Inf ✅
- 负 I 537/3823 = 14.0466%（高 Eq 381 / 低 Eq 156 / 非负 2/3284）✅
- I mean 1.6070057；clip_zero 0.2877%（11/3823）；frame1 128/128、frame2 159/160 ✅
- 分解 vs metrics：imitation 逐值相等、其余 ≤4.768e-7（1e-6 容差内）；独立 NumPy 路径 Eq 1.19e-7 / imitation 1.9e-6 / 总 reward 1.19e-7（1e-4 内）✅
- 范围限定接受：无关节速度/actuator_force/40 维完整参考 → 签收范围 = 分项 vs env metrics 数值一致性，不延伸完整物理语义/原生 actor/ONNX/全训练验收
- 措辞修正接受：「位级零差」→ 容差表（1e-6/1e-4）

## 2. 2/1/10 同轨迹重打分 — 复现精确一致 + 单因子归因

复现（固定 3823 非 done 状态/动作/参考，S_new = 2.5·tlin + 1·tang + 2 + 10·I − 1e-3·torq − 0.5·arate − 0.2·sstill）：
- 裁零 **510/3823 = 13.3403%**（与 Codex 一致）；mean reward 0.471875（一致）

单因子归因（从原配比逐一改）：
| 变体 | 裁零率 | mean_r |
|---|---|---|
| 原配比 (2.5tlin/6tang/20alive/1I) | 0.2877% (11) | 0.5788 |
| tang 6→1 单独 | 0.2877% (11) | 0.4658 |
| alive 20→2 单独 | 4.1067% (157) | 0.1888 |
| I 1→10 单独 | 4.6822% (179) | 0.8717 |
| alive→2 + I→10 | 13.0264% (498) | 0.5446 |
| 全改 (2/1/10) | 13.3403% (510) | 0.4719 |

结论：
- 裁零上升主因 = **alive 降权 + I 升权叠加**（tang 改动无贡献）。alive 20→2 单独就暴露 157 个「靠 alive 撑正」的低质量步；I 放大 10 倍再暴露 179 个负 I 步；叠加后 498（非线性，交互放大）。
- 裁零样本画像：100% I<0（mean −3.11 vs 全体 +1.61）、tlin 0.078（全体 0.397）、arate 0.72（全体 0.20）、Eq mean 0.376（≈每关节 8.8°）——即「偏离参考且跟踪差」的步。2/1/10 的设计意图（惩罚站住偷懒）在数据上成立，但 13.34% 步零奖励 = 每 ~7.5 步 1 步无梯度信号，支持「暂缓盲目采用」判定。
- 结构性观察（供参考，非证据）：alive 20 是当前策略 mean reward 的最大单项来源（20×dt≈0.4/步，超过 tlin 均值贡献）——这解释「学到站住」的奖励结构成因，与 Codex「当前数据不证明局部最优/唯一根因」不冲突。
- 采纳为工具：同轨迹重打分 = 零 GPU 配比预筛（本轮已完成，脚本 sha256 6706f21e…，npz 数据 /tmp/trace3_0.1_seed0.npz）。

## 3. trace4 — reset/命令同步修复版（Codex 裁决第 2 条落地）

对 Codex 报告「尚存的命令/重置问题」逐条对应：

| Codex 问题 | trace4 修复 | trace3 根因 |
|---|---|---|
| 17 done 样本 obs_cmd 与 cmd 不匹配 (max 0.0347) | cmd/obs_before 全部 **step 前抓**（与动作同帧）；保存列含终检 `obs_cmd vs cmd max\|Δ\|` 应全零 | 保存块 step 后抓 obs → AutoReset 已重置 obs，cmd 仍旧 |
| 首动作前未完成命令同步（先改 info、旧 obs warmup） | `sync_cmd()`：reset 后注入固定命令 → 重建 info.command / current_reference_motion / obs（调 uenv._get_obs 同 reset 路径）→ **首动作即目标 cmd**；失败 fallback warmup + WARNING | warmup act0 用 reset 默认 cmd 的 obs |
| 推理前无命令一致性断言 | 每 50 步全 env 断言 obs_cmd == cmd，desync 即打印 | — |
| 保存 obs_before/action/obs_after | 每步保存 **obs_before（101 维全量）+ action + obs_after**（SAVE_EVERY=1） | 仅存 obs_cmd 7 维 |
| episode 37 vs done 17（隔步采样漏 done） | done_events_total 每步累加（全量），输出区分「运行总终止数」vs「保存样本中 done」 | SAVE_EVERY=2 隔步采样 |
| reset 后首步覆盖 | STEPS=300 全程首步即保存 | — |
| 多命令覆盖 | 前 16 env 直线走 cmd_vx / 后 16 env 站立 cmd=0（gate<0.01 分支覆盖） | 单命令 |
| 多相位 / home 初态 | 相位由 rollout 自然覆盖 0-26（存 phase 列）；初态噪声保持 reset 默认（如需 home 无噪声初态需另加场景组，待议） | 同 |

- 守恒对拍保留（同帧、排除 done，Codex 已签收口径）——trace4 首跑将重出对拍表
- obs 布局断言：运行首行打印 obs_dim，期望 101
- 产物：`/root/trace4_<cmd>_seed<n>.npz`（含 obs_before 101 维列 → 后续原生/ONNX 对齐的直接数据源）
- 脚本本地已备：`/tmp/odm_gpu_trace4.py` sha256 **b33ff0e7f0ef39258376d259ae534d1974e659ce7d532dd879a4b44bd2ac6248**，py_compile 通过
- 运行：AutoDL 834 实例开机后（Sky 控制台）→ `/root/odm_gpu_trace4.py 0.1 0`（1 轮 GPU run，~2-3 min + 冷编译）

## 4. 待 Sky/Codex 决策

- 明天开机后先跑 trace4（补 obs_full + 修复验证）→ 再原生/ONNX 对齐（裁决第 3 条，需原生 checkpoint 构造脚本）
- 2/1/10 相关：如 Codex 有其他候选配比想预筛，同轨迹重打分可零成本扩展（改 /tmp/ratioscan_210.py variants 段即可）

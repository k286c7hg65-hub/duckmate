# trace4 v3 GPU 首跑报告（Codex 二审 TRACE4V2_REVIEW 三处修正后）

> 2026-09-07 08:17 UTC | AutoDL 834 (RTX 4090) | odm_gpu_trace4_v3.py 0.1 0 300
> 状态：**冒烟 PASS（强化门槛）+ 正式采集 PACK（14 项终检）** | GPU ~7min
> 对应：Codex 二审 #1/#2/#3 全部落地；范围修正声明已入脚本

## 二审三处修正 → 实测

| Codex 二审项 | v3 实现 | 实测 |
|---|---|---|
| #1 冒烟非 fail-fast | 冒烟每动作前复用 hard_check（含 action shape/有限检查；shape 错先存诊断退出防广播异常绕过） | 冒烟 150 步 × 2 检查点零失败 |
| #2 done 门槛 | `np.all(done_seen>=2)` + ep_id 增量==done 数断言 + truncation 分列 | per-env done **min=3 max=7**；ep_id==done **32/32**；truncation min0 max3（冒烟 ep40） |
| #3 terminal 语义 | terminal_state_available 恒 False + terminal_reward_available=done 分列；post_state_valid=对动作后继/分解有效 | 本地复算：tsa 全 False / tra==done / psv==~done ✅ |

## 范围修正（已入脚本 record 行 + 实证对照）

- first_obs 同步**仅精确修复 cmd 槽**；history/phase/last_act/motor_targets 槽 = reset 原样
  （远端 wrapper 保留 info — 训练原样路径，不宣称一致）
- **v2 npz 离线对照实证**（Codex 要求记录）：done 后首行 obs last 槽（41:55）
  norm mean≈1.91 **非零**（非重置清零）；last2/last3/mt 槽均非零；phase 槽 mean≈−0.08/−0.10
  （接近非精确 0）；obs.last槽 vs info.last_act 范数差 mean 0.41（普通行 0.24）—
  普遍存在的训练原样行为，非 cmd 槽类干净重置。结论：cmd 槽同步精确；其余槽不宣称。
- wrapper 版本记录（运行输出）：模块路径 + sha256[0:16]=6d427b43e9df60d9 +
  关键行检查（first_obs=True / full_reset=False / where_done=True）— key 存在≠版本同，
  已按远端实际（2026-09-03 简版, key `first_obs`/`first_state`）适配

## 正式采集数据

- 9600 样本 = 300 步 × 32 env；done 44 **全物理终止（trunc=0，300<1000 ep）**；
  per-env done 0-15（后 16 stand env 0 done = 站住不终止，语义正确；walk env 反复倒）
- obs_cmd vs cmd（9600 含 done）max|Δ|=**0.0**；done 行 next_policy_obs cmd 槽 max|Δ|=**0.0**
- 守恒对拍 7 项 max|Δ|=0.0；state.reward vs reward_clip 1.19e-7（float32 舍入）
- episode_id 单调 + 末值==per-env done 数 ✅；terminal 三列语义本地复算一致 ✅
- I<0 有效样本 7.43%（walk env gate 分支）

## 产物

- `/root/trace4_0.1_seed0.npz`（sha256 **9c12fab7**…，16.98MB，**49 keys**：
  +terminal_state_available/+terminal_reward_available/+truncation vs v2）
- 本地副本 /tmp/trace4_0.1_seed0_v3.npz（远端/本地 sha 一致验证）
- 脚本 odm_gpu_trace4_v3.py（sha256 1b3c1219…）run log 随包

## 验收边界（Codex 二审要求的小日志项 → 交付位置）

1. 逐 env done/truncation 计数 → smoke PASS 行（min3 max7 / trunc min0 max3）+ 正式 44 done/0 trunc
2. 首次及 reset 后首次动作前检查 → 冒烟每动作 hard_check 零失败 + done 后 obs_cmd 断言
3. 完整 obs 有限数 → 终检全列 finite + 每步 hard_check isfinite
4. episode 关系 → ep_id==done 32/32（冒烟）+ 末值==done 数（正式）
5. 失败时未执行 policy/step → hard_check 抛 SyncError 在 policy/step 调用前（ABORT exit 1）

## 下一步

- 原生/ONNX 对齐（Codex 裁决第 3 条）：本 npz obs_before（101 维同帧）即输入数据源
- PD 可达性诊断（第 4 条）
- 0.05/0.15 + 命令切换覆盖扩展（小样本可靠后）

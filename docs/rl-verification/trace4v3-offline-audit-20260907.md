# trace4 v3 离线核验补充（回应 Codex TRACE4V3_RUN_REVIEW 20260907）

> 基于已交付 v3 npz（sha256 9c12fab7…，/tmp/trace4_0.1_seed0_v3.npz）离线计算
> 脚本: /tmp/codex_pairing_audit.py

## 1. 配对语义核验（Codex 提示正确 — v2 原对照差帧）

joystick step 内部顺序：`_get_obs`（读 info.last_act=action[t-1]）→ ... → `info["last_act"]=action[t]`（463 行后）。

- **配对 A（正确同帧）**: `obs_after[t].last槽 vs info.last_act[t]`（t 步 step 前抓）
  - 普通行 mean/max = **0.0 / 0.0**（位级零差）
  - done 后首行 = **0.0 / 0.0**（reset 后第一步正常步进也零差 → 训练原样自洽铁证）
  - 全部行 mean 0.0091 = 仅来自 done 行 44 个（其 obs_after 被 wrapper 替换为 first_obs，预期差异）
- **配对 B（v2 原对照 — 差一步）**: `obs_before[t].last槽 vs info.last_act[t]`
  - 普通行 mean 0.244（obs_before[t]=obs_after[t-1].last=action[t-2] vs info.last_act[t]=action[t-1]）
  - → **v2 对照确实差帧配对**；0.24 为差一步的正常现象（普通行同量级），
    **撤回 v2 对照「reset 后 last 槽非零=原样行为」的解读强度** — Codex 判断成立

## 2. episode 计数核验

- 末帧(299) done 数 = **0** → episode_id 末值无遗漏
- global_done_total 末值 = 44 == done.sum() = 44 ✅
- episode_id 末值 == per-env done 数 **True** ✅（ep_id 列 = pre-step 值 = 动作所属旧 episode；末帧无 done 故计数完整）
- 300 步内 episode_step max = 299 < 999 → 无超时 done 可能 → truncation 列全 0 正确

## 3. done 行终止性质

- joystick `_get_termination` = `fall(gravity_z<0) | isnan(qpos).any() | isnan(qvel).any()` — **仅两类终止源**
- done 行 jq_full/joint_act/z/obs_after 全 finite → **NaN 终止排除** → 44 done 全 = 跌倒终止
- 注意: done 帧物理态不可见（wrapper 已替换为 first_data，done 行 z=0.15 全同）— fall 判定来自
  env 内部替换前 data（_get_termination 在 wrapper 替换前执行）— 与二审 #3 语义一致

## 4. truncation 语义承认

- is_trunc 用 `pre_ep_step>=999` 推定，**未读 wrapper 实际 truncation 字段** — 应称「按步数推定超时」
- 300 步数据内不可能触发（max 299）；episode 末步 env 跌倒会被混分 — 代码语义保留（本数据不触发，
  严格区分需读 EpisodeWrapper truncation 字段，v4 如需可纳入）

## 结论

三处静态修复已关闭；配对 A 位级零差 = 观测构造与 info 更新自洽；v2 对照差帧撤回解读；
44 done 全跌倒（NaN 排除）；episode 计数无遗漏。**无需重跑** — 进入原生/ONNX 对齐（裁决第 3 条）。

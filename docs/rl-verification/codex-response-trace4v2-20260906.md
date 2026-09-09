# Codex → Sky / Ariste：trace4 v2 修正版（静态复核 4+1 项全落地）

> 2026-09-06 | 对应 Codex TRACE4_SCRIPT_REVIEW_20260906 | 交付：odm_gpu_trace4_v2.py + wrapper 源码核对依据
> 状态：脚本已就绪（py_compile + 静态不变量 9/9），**未执行 GPU**（实例未开，等 Sky）

## 0. 前置：核对实际 wrapper 实现（非猜测）

Codex #1 要求「核对实际 wrapper 实现，不能盲改猜测缓存结构」。本地无 mujoco_playground
源码（AutoDL 关机），从 google-deepmind/mujoco_playground 官方仓库拉取 wrapper.py
（sha256 46b99e7f…，随包附全文）。关键机制确认：

```
wrap_for_brax_training = VmapWrapper → EpisodeWrapper → BraxAutoResetWrapper(full_reset=False)
reset(): 缓存 info['AutoResetWrapper_first_data'/'first_obs'] = reset 时 data/obs
step():  done env 的 data/obs 被 where_done 替换为上述缓存值; info 不重置
         (仅 done_count 累加、rng 更新、steps 归零)
```

⇒ **Codex #1 判断精确成立**：trace4 原版只改 `state.obs`/`info.command`，不改
`first_obs` 缓存 → 中途 done env 会被替换回**旧 first_obs（默认 warmup cmd）**，
命令失配在 done 后复现。正解 = sync 时同步更新 first_obs 缓存。

## 1. 修正对照表（Codex 4+1 项 → v2 实现）

| Codex 项 | v2 实现 |
|---|---|
| #1 done 后初态/命令/参考/缓存一致 | `sync_cmd` 在重建 obs 后同步更新 `info['AutoResetWrapper_first_obs']`（per-env tree `.at[:].set`）。obs 由 reset 后 data 构造 → 与 first_data 物理一致，安全替换。此后 wrapper 的 where_done 自动把 done env 带回**目标 cmd 的 reset 状态**，无需逐 done 重 sync（避免 _get_obs 副作用——见下）。**key 缺失 = 远端 wrapper 版本与核对源码不符 → 硬失败**（不静默降级） |
| #2 同步失败仍继续 / 检查放每步 | ① 删除 warmup fallback（sync 失败 → SyncError → ABORT exit 1）② `hard_check` 每动作前全 env 执行：shape==101 / isfinite / `\|obs_before[:,6:13] − cmd_arr\|`≤1e-5（rtol=0 未舍入 float32）且 info.command==cmd_arr；失败 → 诊断落盘 `trace4_<tag>_DIAG.npz` + exit 1，**不调用 policy/step** |
| #3 episode 列被全局计数广播 | 删广播列。per-env `episode_id`（done 时 +1，单调）+ `episode_step`（每 env 独立，done 归零重计）+ `global_done_total` 单列。保存行记 `pre_ep_id/pre_ep_step`（step 前值）→ **动作归属标清 = 产生 action 的旧 episode**（done 行的动作属旧 episode 末步，非 reset 后新 episode） |
| #4 done 行字段语义 | 新增 `post_state_valid`(=~done) + `terminal_available`(=done) 列。分解/高度/ref 等 step 后字段仅 post_state_valid 样本有效（终检同帧对拍也用此 mask 而非全样本）。`obs_after` 注释明确 = wrapper 返回状态；另存 `next_policy_obs`（=obs_after——因 first_obs 已同步，done env 被替换为目标 cmd reset 状态，两者一致；语义注明） |
| +1 守恒/一致性统一检查 + 冒烟 | 终检 5 组全过才 PACK：a) 全列有限数 b) 守恒对拍 7 项 + reward（≤1e-4，post_state_valid mask）c) obs_cmd vs cmd 全样本含 done ≤1e-5（trace3 缺陷修复验证）d) done 行 next_policy_obs cmd 槽 vs cmd_arr ≤1e-5（first_obs 同步生效的直接证据）e) episode_id 单调 + episode_step ∈[0,1000]。任一 fail → `trace4_<tag>.FAILED` + exit 1。**冒烟 = episode_length=40 短跑 120 步**（每 env ≥2 次 done 边界），断言 done 出现 / done 后 obs_cmd==cmd_arr / ep_id 单调，PASS 才进正式采集 |

## 2. 实现注意逐条回应（Codex 提醒）

- **_get_obs 副作用**：sync 只在 reset 后全 env 各一次，循环中对未 done env 零调用；
  done env 的重置由 wrapper where_done 用已同步 first_obs 完成（wrapper 内部路径，
  不额外调 _get_obs）→ 不改变随机过程。
- **首动作/reset 后首动作**：reset 后 sync（含缓存）→ 冒烟与正式采集每步 hard_check
  覆盖「每次推理前命令一致」；冒烟覆盖「reset 后首次动作」（done 后 ep_step=0 的
  下一动作即 reset 后首动作，pre-step 检查 + done 后 obs 检查双重覆盖）。
- **原生/ONNX 对齐数据源**：obs_before（101 维全量）与 action 严格同帧保存；
  obs_after 仅作 wrapper 状态参考，**不用 obs_after 与 action 配对**。
- **覆盖范围**：本轮 0.10（walk gate 分支）+ 0.0（stand，后 16 env）两分支；
  0.05/0.15 与命令切换留在小样本可靠后扩展（Codex 同意先可靠小样本）。

## 3. 待执行（等 Sky 开机 834）

1. 开机 → 先核对远端 wrapper.py 与官方源码行差异（`sed` 关键段比对；若版本差异导致
   key 结构变化，脚本会硬失败并提示，届时按实际结构微调）
2. 跑 `python3 /root/odm_gpu_trace4_v2.py 0.1 0 300`（冒烟 ~1min + 正式 ~3-4min + 冷编译）
3. 交付短冒烟日志（证明：每次推理前命令一致 / 首次动作与 reset 后首动作正确 /
   per-env episode 可追踪）→ 过门槛后采完整对齐数据（无需重训 150M）

## 4. 附件

- `odm_gpu_trace4_v2.py`（sha256 a9244909…）— 修正版主脚本
- `mjpl_mujoco_playground__src_wrapper.py`（sha256 46b99e7f…）— 官方 wrapper.py 全文
  （google-deepmind/mujoco_playground，核对依据；远端安装版本以开机比对为准）

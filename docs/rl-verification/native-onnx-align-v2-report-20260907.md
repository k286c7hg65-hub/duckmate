# 原生/ONNX 对齐 v2 报告（回应 ALIGN_REVIEW_AND_OBS_PD_SCOPE）

> 2026-09-07 01:45 UTC | AutoDL 834 (CPU) | odm_align_v2.py | 三路对拍全 PASS

## 复核修正逐条落实

1. **补 Brax 训练原生 API 路**（Codex 核心要求）：network_factory + make_inference_fn
   deterministic=True——构造与训练同源（brax_ppo_config network_factory：
   policy_hidden (512,256,128) / policy_obs_key='state' / value_obs_key='privileged_state'
   + preprocess_observations_fn=running_statistics.normalize = train.py 439-449 同款接线）
2. **reset_after_mask 方向修正**：`mask[1:]=done[:-1]`（原 `nxt[:-1]=done[1:]` 选的是
   done 前一行——Codex 复算交集 0 确认）——修正后分组 44 行（全 9600 对拍不受影响）
3. **输入相位用 obs_before[99:101]**（非 step 后 info phase 列）——walk 输入相位对唯一数 28
4. **哈希补全**：input npz 9c12fab7…（log 全文含 14 ckpt 文件 + onnx 1f5cb9bb +
   exporter 30bc705b + config 7fd74d85 + count/std_eps 落 log）
5. **逐样本差落盘** /root/align_per_sample.npz（brax-vs-onnx / numpy-vs-onnx /
   brax-vs-numpy 全 9600 样本 max-per-sample）+ 有限数检查 + FAIL exit 1

## 三路对拍结果（全部 ≤1e-4 ✅）

| 组 | n | numpy-vs-brax | numpy-vs-onnx | brax-vs-onnx |
|---|---|---|---|---|
| 全部 9600 | 9600 | 8.9e-7 | 1.01e-6 | **1.07e-6** |
| walk cmd=0.10 | 4800 | 7.9e-7 | 1.01e-6 | 8.8e-7 |
| stand cmd=0 | 4800 | 8.9e-7 | 8.3e-7 | 1.07e-6 |
| 首次动作 | 76 | 6.3e-7 | 7.6e-7 | 6.0e-7 |
| reset 后首动作(修正) | 44 | 4.5e-7 | 3.3e-7 | 6.0e-7 |
| done 行 | 44 | 6.1e-7 | 1.01e-6 | 7.5e-7 |

- 最大误差样本 t=154 env=17 cmd=0（stand env）：brax-vs-onnx 1.07e-6
- **三路两两互证**：手写 numpy（独立实现）↔ ONNX ↔ brax 训练原生 API 全在 1e-6 量级

## 关键调试发现（vmap 陷阱）

brax 路初版用 `jax.vmap` 批量——**vmap 引入 4.6e-4 差异**（XLA 批量化重排浮点，
200 行样本实测 4.56e-4）；改逐行 loop 后即达 1e-6。单样本验证 brax logits 与手写
位级一致（7e-7）→ 差异纯为 vmap 数值重排，非模型不一致。已记录防复用。

## 边界声明（同前）

- 对拍验证三路同输入同输出（权重搬运 + 图执行 + 归一化接线正确性）
- 不证明观测构造器等价（Codex：逐字段 ≤1e-5 独立项待做）
- 环境语义零改动

## 产物

- 脚本 odm_align_v2.py（sha256 见 SHA256SUMS）
- run log align_v2_run.log（哈希清单/config dump/覆盖/对拍全文）
- /root/align_per_sample.npz（逐样本三路差）
- 输入 trace4_0.1_seed0.npz（9c12fab7）

# 原生/ONNX 动作对拍报告（Codex FULL_VERIFIED 下一步最小交付）

> 2026-09-07 00:45 UTC | AutoDL 834 (RTX 4090, CPU 推理) | odm_native_onnx_align.py
> 状态：**PASS — 全部 9600 组 max|Δ|=1.01e-6（规格 ≤1e-4 的 1/100）** | 运行 ~3min

## 规格落实（Codex 逐条）

1. **≥100 组覆盖**：全样本 9600 组（超规格）——walk cmd=0.10 ×4800 / stand cmd=0 ×4800 /
   首次动作（ep_step=0）×76 / reset 后首动作 ×44 / done 行 ×44；
   **walk 覆盖全部 27 参考相位**
2. **原生归一化状态 + 确定性 actor**：原生前向 = orbax restore 的 norm（mean/std state 101）
   + 手动 numpy swish dense×3 + tanh(loc) —— 确定性无采样
3. **独立加载路径**：orbax PyTreeCheckpointer.restore 原生 ckpt（训练原产物）→ 手动 numpy 前向，
   **不经 tf2onnx/tf.keras 链**（exporter 仅作结构参考）；ONNX 侧 onnxruntime 逐行推理（模型静态 batch=1）
4. **哈希清单**（见 log 全文）：
   - ckpt 目录 14 文件逐文件 sha256（manifest.ocdbt b9b3a5e7…/ocdbt.process_0/d/* 等）
   - onnx: 1f5cb9bb2f0025ab…ffc6069
   - exporter: 30bc705bf59babb5…e61139cc
   - config: mujoco_playground/config/locomotion_params.py 7fd74d85…，
     policy_hidden_layer_sizes=(512,256,128)（与 _METADATA 实证 101→512→256→128→28 一致）

## 对拍结果

| 组 | n | max\|Δ\| | mean |
|---|---|---|---|
| 全部 9600 | 9600 | **1.01e-6** | 6e-8 |
| walk cmd=0.10 | 4800 | 1.01e-6 | 6e-8 |
| stand cmd=0 | 4800 | 8.3e-7 | 6e-8 |
| 首次动作 ep_step=0 | 76 | 7.6e-7 | 1e-7 |
| reset 后首动作 | 44 | 6.0e-7 | 1.1e-7 |
| done 行（终止转换） | 44 | 1.01e-6 | 1.2e-7 |

- 最大误差样本：t=125 env=7 cmd=[0.1,0,...] max|Δ|=1.01e-6
- 全部组 ≤1e-4 阈值 ✅（余量 ~100×）；1e-6 量级 = float32 精度极限附近

## 边界声明

- 本对拍验证：ONNX 导出/推理与原生存量参数同输入同输出（权重搬运 + 图执行正确性）
- **不证明观测构造器等价**（Codex 原文：同一输入模型对拍不能证明两构造器等价——
  逐字段 ≤1e-5 为另一项检查，待做）
- 环境语义未改动（Codex：不在对照实验中顺手改环境语义）

## 产物

- 脚本 odm_native_onnx_align.py（sha256 dab6e43c…）
- run log align_run.log（哈希清单 + 覆盖 + 结果全文）
- 输入数据源：trace4_0.1_seed0.npz（9c12fab7，obs_before 101 维同帧）

## 下一步（裁决顺序）

1. 逐字段观测构造器检查（≤1e-5，独立项）
2. PD 可达性诊断（裁决第 4 条）→ 决定是否重建直线参考
3. 0.05/0.15 命令 + 命令切换最小补采

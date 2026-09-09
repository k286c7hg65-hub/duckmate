# Codex 附件包 batch2 — 实际文件交付（2026-09-06）

对应 Codex BATCH2 复核「下一次人工交接最小包」要求。SHA256SUMS.txt 为全部文件哈希。

## 包内容

| 文件 | 说明 |
|---|---|
| gpu_diag3_sweep.json | cmd sweep (0/0.05/0.10/0.15 × 2 seeds) 训练 env 分项原始 JSON（64 env × 480 步） |
| gpu_diag_0.1_seed0.json | 首 run (cmd 0.10 seed 0) 原始 JSON |
| gpu_trace_0.1_seed0.json | **逐步 trace**：32 env × 240 步 cmd 0.10，每 10 步采样 3 env 的 actual joint / ref joint / Eq / I_metric / z（Eq stats 含分位数） |
| odm_gpu_diag2.py | 诊断脚本（单 run） |
| odm_gpu_diag3.py | sweep 脚本（4 cmd × 2 seeds 单进程） |
| odm_gpu_trace.py | trace 脚本（可复现） |
| custom_rewards.py | **实际加载的 imitation reward 函数**（SHA 锁定） |
| joystick.py_current | **实际生效训练配置**（Spec v1 patch 后, AutoDL 当前运行版, SHA 锁定） |
| joystick.py.bak_20260906_1325 | Spec patch 前备份（diff 用） |
| poly_reference_motion.py | PRM 读写/采样代码（vel_to_index / sample_polynomial） |
| prm_meta.json | pkl 元数据（240 entries, 网格, period 0.54@50fps=27 步） |
| prm_ref_0.074_-0.037_-0.074.csv | **27 帧参考导出**（cmd 0.10 选中键, 40 维/帧 + 字段见下） |
| prm_ref_0.148_-0.037_-0.074.csv | cmd 0.15 选中键 27 帧 |
| prm_ref_0.0_-0.037_-0.074.csv | cmd 0 选中键 27 帧 |
| prm_ref_0.222_0.037_0.185.csv | 高速档 27 帧（对照） |
| prm_vel_to_index_map.csv | 命令 vx 0.05-0.15 细扫 → 选中键映射表 |

## 40 维字段表（poly_reference_motion.py 注释一致）

dim 0-15: joint pos [L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle, neck_pitch, head_pitch, head_yaw, head_roll, L_antenna, R_antenna, R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle]
dim 16-31: 对应 joint vel
dim 32-33: foot_contacts（**连续力值** 0~1.1, reward 内 >0.5 阈值化）
dim 34-36: base world linear vel
dim 37-39: base world angular vel
（root_pos/quat/toe 已在拟合时剔除; 无 root quat 字段）

## 已核对结论摘要（详见 joystick-codex-contract-audit-20260906.md）

1. custom_rewards: orientation 计算未入总和（注释）; contact = 匹配计数非 exp; B_max=5 = lin_xy1+lin_z1+ang_xy0.5+ang_z0.5+contact2
2. 切片对齐正确: xml 14 关节（无 antenna）vs ref 16（含 2 antenna 占位）; actual [:5]+[9:] = ref [:5]+[11:] = 10 腿关节
3. vel_to_index (0.10,0,0) → (0.074, −0.037, −0.074); dx 网格 = 平均速度语义（b_lin_x mean 0.077 ≈ dx 0.074）
4. Spec patch 生效（diff 实证）: vx [0.05,0.15] / vy [0,0] / yaw [0,0] / head×0 / push off; 10% 全零命令保留
5. obs 含 command[:3], 每步从 info 读（无缓存）; AutoReset 恢复仅在 done（diag 无 done）
6. **站姿 vs 参考 Eq mean 0.11**（蹲姿 home 落在慢走参考摆动范围内）→ imitation 无法区分站/走
7. 终止仅倒挂/NaN → 普通跌倒无代价仍拿 alive 20/步

## 未完成（待 Codex 决策/批准）

- 原生 (brax params) vs ONNX 数值对齐（≥100 组输入逐字段 ≤1e-5）— 需 GPU brax 原生 policy 构造
- 2/1/10 逐步原始项离线重算（含 stand_still 项 + 连续零区间统计）— trace 数据已具备可算（需补 B 分解全量 trace v2）
- 参考 PD 跟踪 rollout（契约已核清, 可做）

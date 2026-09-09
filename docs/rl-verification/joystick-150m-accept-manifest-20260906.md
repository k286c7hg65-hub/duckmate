# joystick 150M 完整验收 manifest — 0/20 FAIL（2026-09-06）

## Case 全表（Spec #2: 15 速度 + 5 起停）
track（静止 5s → vx 30s → 停 5s; 统计窗 5-35s）:
- vx=0.05 × seeds 0-4: 全 FAIL（vx_mean≈0.0003, swings 1/1, disp 0.003-0.06m）
- vx=0.10 × seeds 0-4: 全 FAIL（同上）
- vx=0.15 × seeds 0-4: 全 FAIL（vx_mean≈0.0001, swings 1/1, disp 0.006m）
startstop（停 5s → 0.10 15s → 停 10s）:
- seeds 0-4: 全 FAIL（disp_cmd 0.039m < 1.04m 需要; tail 停 OK 但从未动过）

**总计 0/20**。判据详情（z/tilt/横向/yaw 均过，只挂 vx/位移/摆动）= 策略稳定站住。

## 3-way ckpt 对照（vx=0.10 × 3 seeds, 2026-09-06）
| ckpt | vx_mean | 净位移 | swings |
|---|---|---|---|
| 10M | -0.003 | 0.01m | 0/0 |
| 50M | -0.0006 | 0.003m | 1/1 |
| 150M | +0.0003 | 0.06m | 1/5 |
无中间更好 → 10M 起站住锁定。排除「训练不足中途会走」；原生/ONNX 对齐排除导出问题待 GPU 验证。

## 证据链
- 工具: tools/joystick_eval.py（101 维独立 obs + 判据全集; 40s sim ≈ 1.8s real CPU）
- ONNX: joystick_{10m,50m,150m}_*.onnx（101→512→256→128→14）
- AutoDL ckpt 15/阶段（含每 eval 步 onnx + orbax tensorstore）

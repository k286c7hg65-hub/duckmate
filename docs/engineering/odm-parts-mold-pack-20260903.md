# ODM v2 结构件清单（给友B 的注塑模具评估包）

> 来源：apirrone/Open_Duck_Mini 默认分支 v2（2026-09-03 抓取，⭐3923）
> 仓库总文件 436：129 STL + 93 part + 47 3mf + 4 step + 2 onnx
> 用途：友B 评估 ~20 结构件注塑开模可行性 + 报价

## 一、实际打印件（print/ 目录 36 件，含左右配对）

| # | 零件 | 大小 | 对称配对 | 注塑评估要点 |
|---|---|---|---|---|
| 1 | body_back | 345KB | 单件 | 大件，外观面 |
| 2 | body_front | 190KB | 单件 | 大件，外观面 |
| 3 | body_middle_bottom | 331KB | 单件 | 承重框架 |
| 4 | body_middle_top | 238KB | 单件 | 承重框架 |
| 5 | trunk_bottom | 217KB | 单件 | 连接结构 |
| 6 | trunk_top | 706KB | 单件 | 连接结构 |
| 7 | head | 1742KB | 单件 | **最大件**，外观+壳 |
| 8 | head_bot_sheet | 233KB | 单件 | 头颈连接 |
| 9 | head_pitch_to_yaw | 523KB | 单件 | 传动件 |
| 10 | head_yaw_to_roll | 275KB | 单件 | 传动件 |
| 11 | head_roll_mount | 222KB | 单件 | 舵机座 |
| 12 | left_roll_to_pitch | 389KB | ↔ right 同 | 髋部传动（×2）|
| 13 | left_antenna_holder | 70KB | ↔ right | 天线座（×2）|
| 14 | left_cache | 84KB | ↔ right | 线缆管理（×2）|
| 15 | left_eye | 103KB | ↔ right | 眼（×2，外观）|
| 16 | knee_to_ankle_left_sheet | 221KB | ↔ right | 小腿板（×2）|
| 17 | neck_left_sheet | 167KB | ↔ right | 颈部板（×2）|
| 18 | foot_bottom_pla | 46KB | 左右共用? | 足底（+TPU 版）|
| 19 | foot_bottom_tpu | 75KB | 同上 | **TPU 软底**（需双色/软胶工艺）|
| 20 | foot_side | 185KB | 左右 | 足侧 |
| 21 | foot_top | 177KB | 左右 | 足上 |
| 22 | leg_spacer | 111KB | 多件 | 腿部垫片 |
| 23 | roll_motor_bottom | 66KB | 单件 | 舵机座 |
| 24 | roll_motor_top | 179KB | 单件 | 舵机座 |
| 25 | speaker_interface | 113KB | 单件 | 喇叭口 |
| 26 | speaker_stand | 1044KB | 单件 | 喇叭支架 |
| 27 | bulb | 136KB | 单件 | 灯 |
| 28 | flash_light_module | 321KB | 单件 | 灯模块 |
| 29 | flash_reflector_interface | 43KB | 单件 | 灯反光 |
| 30 | battery_pack_lid | 94KB | 单件 | 电池盖 |

> 注：36 件含 5 组左右配对 + 2 组 foot 系列，实际**独立结构件形态 ≈ 22-24 种**（配对共用一套模具型腔）

## 二、模具分件建议（友B 视角初判）

- **大件注塑候选**（需 PA/ABS 增强，壁厚/缩水要评估）：head(最大)、body_back/front、trunk_top、body_middle_* 
- **板件候选**（sheet 类，可考虑玻纤增强）：knee/neck/head_bot sheet、roll_to_pitch
- **小件群**（可多腔共模）：spacer/eye/holder/antenna/lid ~10 种小件合并 1-2 套模具
- **TPU 特殊**：foot_bottom_tpu 需软胶（TPU 注塑单独工艺，或 V1 保留 3D 打印）
- **V1 建议**：3D 打印（PLA 500g ≈ ¥150/台）零模具费验证；量产开模从「大件 4-6 套 + 小件 2 套 + 板件 2 套」起步

## 三、给友B 的现场问题（M0 共创会）

1. 这套结构件（22-24 形态）注塑开模，**分几套模、各报多少**？（区分：外观大件/承重框架/板件/小件群）
2. 打印件→注塑件的壁厚/圆角/缩水适配，需要改多少 STL？（我们有全部源文件）
3. head 1.7MB 大件的分模线/浇口方案可行性
4. TPU 足底：双色注塑 or 单色+贴片，成本差多少？

## 四、数据位置

- 全部 STL：apirrone/Open_Duck_Mini v2 分支（print/ + mini_bdx/robots/bdx/）
- CAD 源：仓库含 4 个 STEP + Onshape 装配体链接（官方 BOM 内）
- 3mf 打印配置：47 个（含切片参数，可直接参考打印）

---
_生成：Ariste 2026-09-03 | 对应 ariste-tasklist A1.2 | 上游：bp-duckmate-odm-20260903.md 附录 A_

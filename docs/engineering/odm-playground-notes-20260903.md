# ODM Playground 训练栈实测笔记（A1.4 交付 · M1 gate 前置）

> 来源：apirrone/Open_Duck_Playground clone 实测（2026-09-03）| mujoco_playground 系（Berkeley Humanoid 血统）
> 对应 ariste-tasklist A1.4 | 用途：技能重训操作手册（M1 gate = 跑通 1 个最小重训技能）

## 一、仓库结构（实测）

```
Open_Duck_Playground/
├── pyproject.toml          # uv 管理，jax + mujoco + mujoco_playground
├── playground/
│   ├── common/             # 共享：runner/export_onnx/onnx_infer/poly_reference_motion/randomize/rewards
│   └── open_duck_mini_v2/  # 唯一机器人（ODM v2）
│       ├── base.py         # MjxEnv 基类（ODM v2 模型，含 trunk freejoint + 14 执行器）
│       ├── constants.py    # XML 路径（flat/rough terrain ± backlash）、足部站点
│       ├── joystick.py     # 走跑任务（USE_IMITATION_REWARD=True，需参考动作）
│       ├── standing.py     # ★ 站立任务（USE_IMITATION_REWARD=False = 无需参考动作）
│       ├── runner.py       # 训练入口（--env joystick|standing --task flat_terrain）
│       ├── mujoco_infer.py # 真机/仿真推理（-o xxx.onnx）
│       ├── custom_rewards.py / custom_rewards_numpy.py
│       ├── xmls/           # MJCF 场景（scene_flat_terrain.xml 等 4 个）
│       └── data/polynomial_coefficients.pkl  # 参考动作已就位（joystick 用）
```

## 二、训练入口（实测参数）

```bash
uv run playground/open_duck_mini_v2/runner.py \
  --env joystick|standing --task flat_terrain|rough_terrain \
  --output_dir checkpoints --num_timesteps 150000000
```

关键默认（joystick.py default_config）：
- ctrl_dt=0.02 / sim_dt=0.002 / action_repeat=1
- action_scale=0.25 / max_motor_velocity=5.24 rad/s
- episode_length=1000（=20s 仿真）
- 域随机化：playground/common/randomize.py（vin/压降/延迟/回差——继承 microduck_rl 同源配方）

## 三、★ M1 gate 最小重训候选 = standing 任务

**为什么选 standing 而非 joystick**：
1. USE_IMITATION_REWARD=False → **不需要参考动作数据**（省 auto_waddle 环节，joystick 才有）
2. 奖励 = 姿态稳定/扭矩/动作率（cost_orientation/cost_torques/cost_action_rate/cost_stand_still/reward_alive）——通用任务定义
3. 661 行纯任务文件，无外部依赖
4. 产出 = 站立 ONNX → 是 ODM 走路策略之外最安全的「第二技能」验证

**joystick（走路）注意**：USE_IMITATION_REWARD=True 需 data/polynomial_coefficients.pkl（已就位 ✅）+ 走跑是 150M timesteps 全量训练

## 四、推理/导出（实测文件）

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <策略.onnx>   # mujoco 仿真验证
# export: playground/common/export_onnx.py（训练 checkpoint → ONNX）
```

## 五、训练成本估算（MicroDuck 系经验，✅ 参考）

| 任务 | timesteps | GPU 时数（4096 env）| 单技能成本 |
|---|---|---|---|
| standing（M1 gate）| 可先 10-30M 验证收敛 | ~0.5-1h | <$1 |
| joystick 走路全量 | 150M | 2-4h | $2-5 |
| 自定义踢球（后续）| 需参考动作 + 定制奖励 | 数小时-1 天迭代 | $5-20 |

> AutoDL 4090 ≈ ¥2-3/h（setup_autodl.sh 已备，⏳ Sky 注册实例）

## 六、M1 gate 执行计划（AutoDL 就绪后）

1. clone 本仓库 + uv sync（pyproject 已锁定 jax-cuda 版本）
2. 小步验证：`--num_timesteps 10M` 跑 standing，看 reward 曲线收敛
3. 全量训练 standing → export ONNX → mujoco_infer 验证站立稳定
4. 产出：standing.onnx + 训练日志 + 成本台账（对齐 virt-duck-plan 轨 B 纪律）
5. 通过标准：仿真站立 20s 不倒 + 全程 <$3 GPU 成本

## 七、与 microduck-rl-fork 的关系（方法论迁移）

- microduck-rl-fork（父会话已跑通微缩版）：BAM 执行器模型 + PPO + verify_kick_window
- ODM Playground：mujoco_playground + jax（同源 Berkeley Humanoid 配方）
- 迁移内容：踢球窗口验证方法论（verify_kick_window 实证）、域随机化参数、失败模式统计
- 差异：执行器模型（MicroDuck BAM vs ODM 用 mujoco 原生 + backlash XML）——ODM 用 scene_flat_terrain_backlash.xml 处理回差，无需 BAM

## 八、诚实标注

- ⚠️ 本笔记为静态代码阅读，未实际跑训练（AutoDL 未注册）
- ⚠️ 150M timesteps 是 joystick 默认；standing 无官方参考时长，需实测收敛
- 🔵 domain_randomize 细节（randomize.py）需训练前通读确认覆盖范围
- ✅ polynomial_coefficients.pkl 3.1MB 已就位（joystick 走路的参考动作数据完整）

---
_生成：Ariste 2026-09-03 | clone: /tmp/odm-playground | 下一步：Sky 注册 AutoDL → 执行 M1 gate_

# wrapper 版本差异确认（Codex RUN_REVIEW 要求：源码/路径/版本/完整 SHA/diff）

## 远端实际运行版本（834 机）

- 模块路径: `/root/odm-playground/.venv/lib/python3.12/site-packages/mujoco_playground/_src/wrapper.py`
- 完整 sha256: `6d427b43e9df60d95fc2a0523096a57257ca6006e608c53703d924d852aeb016`
- 行数: 315 | mujoco_playground `__version__`: 未定义（'?'）
- 特征: 2025 DeepMind 版权头；`wrap_for_brax_training` 无 full_reset/vision 参数；
  `BraxAutoResetWrapper.__init__(self, env)`（无 full_reset 选项）——
  直接缓存 `info['first_state']=data, info['first_obs']=obs`；
  step() 中 done env 的 data/obs 由 where_done 替换回缓存；**info 不重置**
- 完整源码随包（remote_wrapper_834.py）

## 对照：官方 master（google-deepmind/mujoco_playground, 2026 Google LLC 版）

- 本地存档 /tmp/mjpl_mujoco_playground__src_wrapper.py（sha256 46b99e7f…）
- 特征: 2026 版含 `full_reset: bool = False` 参数 + vision/MadronaWrapper +
  `AutoResetWrapper_first_data`/`first_obs` key 前缀 + preserve_info 机制
- diff 共 290 行（随包 wrapper_diff_master_vs_remote.txt）

## 对 v2/v3 脚本的影响

- v2/v3 均按**远端实际 key**（`first_obs`/`first_state`，无前缀）适配——
  key 缺失硬失败保护未触发 = 适配与远端一致
- info 保留语义：远端版 step() 不重置 info（与 master full_reset=False 同语义）——
  v3 record 行已声明 first_obs 同步仅修复 cmd 槽，history/phase 不宣称一致

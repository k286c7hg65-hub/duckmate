#!/usr/bin/env python3
"""2/1/10 配比重打分复现 + 归因分解（离线，零 GPU）
Codex 报告: S_new = 2.5*tlin + 1*tang + 2 + 10*I - 0.001*torques - 0.5*arate - 0.2*sstill
裁零率 0.288% -> 13.3403% (510/3823)
本脚本: 复现 + 单因子归因（哪个权重改动主导裁零上升）
数据: /tmp/trace3_0.1_seed0.npz (40 列 x 120 x 32, 非 done 3823)
"""
import numpy as np

d = np.load("/tmp/trace3_0.1_seed0.npz")
nd = ~(d["done"] > 0)  # 非 done mask (与 Codex 同口径: 3823)
N = int(nd.sum())
print(f"非 done 样本: {N}")

tlin = d["tlin"][nd]; tang = d["tang"][nd]; I = d["I"][nd]
torq = d["torques"][nd]; arate = d["arate"][nd]; sstill = d["sstill"][nd]

def score(S):
    r = np.clip(S * 0.02, 0, 10000)
    return r, (r == 0).mean()

print("\n=== 复现 Codex 2/1/10 ===")
S_new = 2.5*tlin + 1.0*tang + 2.0 + 10.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill
r_new, cz_new = score(S_new)
print(f"裁零率: {cz_new:.6f} = {int((r_new==0).sum())}/{N}")
print(f"mean reward: {r_new.mean():.6f}")

print("\n=== 单因子归因 (从原配比出发逐一改成 2/1/10 值) ===")
S_base = 2.5*tlin + 6.0*tang + 20.0 + 1.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill
_, cz_base = score(S_base)
print(f"原配比 裁零率: {cz_base:.6f} = {int((np.clip(S_base*0.02,0,10000)==0).sum())}/{N}")

variants = [
    ("tang 6->1",        2.5*tlin + 1.0*tang + 20.0 + 1.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill),
    ("alive 20->2",      2.5*tlin + 6.0*tang + 2.0  + 1.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill),
    ("I 1->10",          2.5*tlin + 6.0*tang + 20.0 + 10.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill),
    ("tang6->1 + I1->10",2.5*tlin + 1.0*tang + 20.0 + 10.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill),
    ("alive20->2 + I1->10", 2.5*tlin + 6.0*tang + 2.0 + 10.0*I - 1e-3*torq - 0.5*arate - 0.2*sstill),
    ("全改 (2/1/10)",    S_new),
]
for name, S in variants:
    r, cz = score(S)
    print(f"{name:22s} 裁零率 {cz:.6f} ({int((r==0).sum()):5d}/{N})  mean_r {r.mean():.6f}")

print("\n=== 2/1/10 裁零样本的 I/Eq 特征 (理解为何打零) ===")
z_mask = r_new == 0
print(f"裁零样本中 I<0 比例: {(I[z_mask] < 0).mean():.4f}")
print(f"裁零样本 mean I: {I[z_mask].mean():.4f} (全体 mean I: {I.mean():.4f})")
print(f"裁零样本 mean tlin: {tlin[z_mask].mean():.4f} (全体: {tlin.mean():.4f})")
print(f"裁零样本 mean tang: {tang[z_mask].mean():.4f} (全体: {tang.mean():.4f})")
print(f"裁零样本 mean |arate|: {arate[z_mask].mean():.4f} (全体: {arate.mean():.4f})")

print("\n=== 每关节 RMS (Eq 分解, 裁零 vs 非裁零) ===")
Eq_all = d["Eq"][nd]
print(f"全体 Eq p50: {np.median(Eq_all):.4f} -> 每关节 {np.sqrt(np.median(Eq_all)/10)*180/np.pi:.2f}°")
print(f"裁零样本 Eq mean: {Eq_all[z_mask].mean():.4f} (全体: {Eq_all.mean():.4f})")

# 2/1/10 的逐项贡献 (非裁零样本上) —— 理解新配比下奖励结构
print("\n=== 2/1/10 配比下奖励结构 (非裁零样本) ===")
r_ok = ~z_mask
print(f"非裁零 {r_ok.sum()}: mean 2.5tlin={np.mean(2.5*tlin[r_ok]):.3f} 1tang={np.mean(1.0*tang[r_ok]):.3f} "
      f"+2={2.0:.3f} 10I={np.mean(10*I[r_ok]):.3f} -torq={np.mean(-1e-3*torq[r_ok]):.4f} "
      f"-arate={np.mean(-0.5*arate[r_ok]):.3f} -sstill={np.mean(-0.2*sstill[r_ok]):.3f}")

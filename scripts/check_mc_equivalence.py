# -*- coding: utf-8 -*-
"""小样本端到端等价性检验：改动后的 mc_deviation 与改动前逐样本求解是否给出同一结果。

只跑 300 个样本（全量 5000×12 支线在本机会跑一小时），目的是证明
"复用标称 λ + s³ 缩放"与"每样本新建求解器"在数学与实现上都等价。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(r"F:\AI+力学\Final_Submission")
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "scripts"))
import Auto_Design as ad            # noqa: E402
import angle_optimize as ao         # noqa: E402
import mc_deviation as mc           # noqa: E402

N = 300


def old_pipeline(case, sig_theta, sig_t):
    """改动前的实现：每样本新建求解器、D 先乘 s³ 再解。"""
    rng = np.random.default_rng(mc.SEED)
    base = np.array(case["theta4"], float)
    out = np.empty(N)
    for i in range(N):
        th = base + rng.normal(0.0, sig_theta, 4)
        ts = 1.0 + rng.normal(0.0, sig_t)
        stack = ad.symmetric_stack(list(th), case["n_plies"])
        D = ad.d_of_angles(stack) * ts ** 3
        lam = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16).ncr(D, mc.NX, mc.NY)
        out[i] = lam * max(mc.NX, mc.NY)
    return out


def new_pipeline(case, sig_theta, sig_t):
    """改动后的实现：标称 λ 缓存一次，再乘 s³。"""
    rng = np.random.default_rng(mc.SEED)
    base = np.array(case["theta4"], float)
    mc.STATE["n_plies"] = case["n_plies"]
    mc._LAM0.clear()
    out = np.empty(N)
    for i in range(N):
        th = base + rng.normal(0.0, sig_theta, 4)
        ts = 1.0 + rng.normal(0.0, sig_t)
        out[i] = mc.ncr_of(list(th), ts)
    return out


for key, case in mc.CASES.items():
    a = old_pipeline(case, 1.0, 0.02)
    b = new_pipeline(case, 1.0, 0.02)
    rel = np.abs(a - b) / np.abs(a)
    print(f"{case['label']}: n={N}  max|Δ|/λ = {rel.max():.3e}  "
          f"均值差 {abs(a.mean() - b.mean()):.3e}  等价(1e-12): {bool(rel.max() < 1e-12)}")

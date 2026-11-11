# -*- coding: utf-8 -*-
"""核对 D03 的解析缩放是否与逐样本求解一致（按 1e-12 比对）。

D03 的收益只发生在"仅厚度扰动"这条支线上：角度固定，λ 与 (1+ε)³ 严格成正比，
所以标称 λ 只需算一次。这里就按这条支线做 A/B 对照。
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(r"F:\AI+力学\Final_Submission")
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "scripts"))
import Auto_Design as ad            # noqa: E402
import angle_optimize as ao         # noqa: E402
import mc_deviation as mc           # noqa: E402

N = 5000


def run_old(case, sig_t):
    rng = np.random.default_rng(mc.SEED)
    solver = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)
    stack = ad.symmetric_stack(case["theta4"], case["n_plies"])
    D0 = ad.d_of_angles(stack)
    out = np.empty(N)
    for i in range(N):
        ts = 1.0 + rng.normal(0.0, sig_t)
        out[i] = solver.ncr(D0 * ts ** 3, mc.NX, mc.NY) * max(mc.NX, mc.NY)
    return out


def run_new(case, sig_t):
    rng = np.random.default_rng(mc.SEED)
    mc.STATE["n_plies"] = case["n_plies"]
    mc._LAM0.clear()
    out = np.empty(N)
    for i in range(N):
        ts = 1.0 + rng.normal(0.0, sig_t)
        out[i] = mc.ncr_of(case["theta4"], ts)
    return out


for key, case in mc.CASES.items():
    t0 = time.perf_counter()
    old = run_old(case, 0.02)
    t1 = time.perf_counter()
    new = run_new(case, 0.02)
    t2 = time.perf_counter()
    rel = np.abs(new - old) / np.abs(old)
    print(f"{case['label']}: max|Δ|/λ = {rel.max():.3e}  "
          f"(逐样本 {t1 - t0:.1f} s → 解析缩放 {t2 - t1:.2f} s)  "
          f"1e-12 内一致: {bool(rel.max() < 1e-12)}")

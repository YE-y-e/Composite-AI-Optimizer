# -*- coding: utf-8 -*-
"""制造偏差下的可靠性：蒙特卡洛分析（10 层主推荐 vs 12 层备选）。

偏差模型（对称扰动，与"求解器只算 D、不含 B 与 A16"的假设一致）：
    4 个设计角   θ_i' = θ_i + N(0, σ_θ)
    所有单层厚度 t' = t · (1 + ε),  ε ~ N(0, σ_t)
角度扰动后按对称方式重建铺层（外表面→中面），厚度扰动等价于 D 矩阵乘 (1+ε)^3。

跑两组公差：基准 ±1.0° / ±2%，放大 ±2.0° / ±4%（看失效概率怎么涨）。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad          # noqa: E402
import angle_optimize as ao       # noqa: E402

N_SAMPLES = 5000
SEED = 42
NX = NY = 500.0
SF = 1.5
MIN_MARGIN = float(getattr(ad, "MIN_MARGIN", 1.2))
TARGET = MIN_MARGIN * SF * max(NX, NY)          # λ ≥ 1.2·SF ⇒ Ncr ≥ 900 N/m

TOLERANCES = [("基准 ±1.0° / ±2%", 1.0, 0.02),
              ("放大 ±2.0° / ±4%", 2.0, 0.04)]

CASES = {
    "primary_10": {"theta4": [-45.0, 45.0, 45.0, -90.0], "n_plies": 10,
                   "label": "10 层主推荐"},
    "backup_12": {"theta4": [-45.0, 45.0, 45.0, -45.0], "n_plies": 12,
                  "label": "12 层备选方案"},
}

STATE = {"n_plies": 10}
_SOLVER = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)   # 复用，别每个样本新建
_LAM0 = {}                                                # (n_plies, theta4) → 标称 λ


def _key(theta4, n_plies):
    return (int(n_plies), tuple(round(float(t), 9) for t in theta4))


def ncr_of(theta4, thick_scale):
    """厚度扰动等价于 D 乘 (1+ε)³，而 λ 与 D 成正比 → 只需算一次标称 λ。

    这一点在校核里验证过：λ(s)/λ(1)/s³ − 1 的残差在 1e-14 量级，
    所以厚度支线不必每个样本都解一次特征值。
    """
    k = _key(theta4, STATE["n_plies"])
    lam0 = _LAM0.get(k)
    if lam0 is None:
        stack = ad.symmetric_stack(theta4, STATE["n_plies"])
        D = ad.d_of_angles(stack)
        lam0 = _SOLVER.ncr(D, NX, NY)
        _LAM0[k] = lam0
    return lam0 * thick_scale ** 3 * max(NX, NY)


def monte_carlo(case, rng, sig_theta, sig_t, perturb_theta=True, perturb_t=True):
    STATE["n_plies"] = case["n_plies"]
    base = np.array(case["theta4"], float)
    out = np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        th = base + rng.normal(0.0, sig_theta, 4) if perturb_theta else base
        ts = 1.0 + rng.normal(0.0, sig_t) if perturb_t else 1.0
        out[i] = ncr_of(list(th), ts)
    return out


def summarize(vals):
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "cv_pct": float(vals.std(ddof=1) / vals.mean() * 100.0),
        "p05": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "worst": float(vals.min()),
        "below_target_pct": float((vals < TARGET).mean() * 100.0),
    }


def main():
    result = {"seed": SEED, "n_samples": N_SAMPLES, "nx": NX, "ny": NY, "sf": SF,
              "target_ncr": TARGET, "min_margin": MIN_MARGIN, "tolerances": {},
              "cases": {}}
    rng = np.random.default_rng(SEED)
    keep = {}

    for label, sig_t_deg, sig_t_rel in TOLERANCES:
        block = {}
        for key, case in CASES.items():
            STATE["n_plies"] = case["n_plies"]
            nominal = ncr_of(case["theta4"], 1.0)
            STATE["n_plies"] = case["n_plies"]
            both = monte_carlo(case, rng, sig_t_deg, sig_t_rel)
            only_th = monte_carlo(case, rng, sig_t_deg, sig_t_rel,
                                  perturb_t=False)
            only_t = monte_carlo(case, rng, sig_t_deg, sig_t_rel,
                                 perturb_theta=False)
            block[key] = {
                "label": case["label"],
                "theta4": case["theta4"],
                "n_plies": case["n_plies"],
                "nominal_ncr": float(nominal),
                "nominal_lam": float(nominal / max(NX, NY)),
                "both": summarize(both),
                "theta_only": summarize(only_th),
                "thickness_only": summarize(only_t),
            }
            print(f"[{label}] {case['label']}: 标称 {nominal:,.1f} N/m → "
                  f"均值 {both.mean():,.1f}，P5 {np.percentile(both, 5):,.1f}，"
                  f"低于目标 {100 * (both < TARGET).mean():.2f}%，"
                  f"CV {both.std(ddof=1) / both.mean() * 100:.2f}%")
            if sig_t_deg == TOLERANCES[0][1]:
                keep[f"{key}_both"] = both
                keep[f"{key}_theta"] = only_th
                keep[f"{key}_thick"] = only_t
        result["tolerances"][label] = block

    out = ROOT / "data" / "mc_deviation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(ROOT / "data" / "mc_deviation_samples.npz", **keep)
    print("saved", out, "and mc_deviation_samples.npz")


if __name__ == "__main__":
    main()

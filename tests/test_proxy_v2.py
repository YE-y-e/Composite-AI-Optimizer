# -*- coding: utf-8 -*-
"""代理筛选实验（C1–C4 / A1–A7）的回归：折角去重、制造规则过滤、划分互斥、统计口径。"""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "codes"))

import Auto_Design as ad          # noqa: E402
import proxy_experiment as pe     # noqa: E402


def _cfg(angles=None, n_plies=12):
    return {"n_plies": n_plies, "nx": 500.0, "ny": 500.0, "sf": 1.5, "m": 16,
            "domain": "grid", "n_raw": 4000, "n_pool": 3000, "n_train": 0,
            "n_test": 0, "budget": [4], "seeds": [0], "n_repeat": 5,
            "angles": angles or [], "enforce_balance": True,
            "min_margin": ad.MIN_MARGIN}


def test_fold_angle_is_180_periodic():
    """±90° 是同一层；θ 与 θ+180° 必须折到同一值。"""
    assert pe.fold_angle(90.0) == pytest.approx(pe.fold_angle(-90.0))
    assert pe.fold_angle(135.0) == pytest.approx(-45.0)
    assert pe.fold_angle(-135.0) == pytest.approx(45.0)
    for a in (-90, -45, 0, 30.5, 89.9):
        assert pe.fold_angle(a + 180.0) == pytest.approx(pe.fold_angle(a))


def test_grid_pool_is_dedup_and_balanced():
    """A1/A2：折角去重后不得有重复物理铺层；开均衡过滤后池内必须全部合规。"""
    cfg = _cfg([-90, -45, 0, 45, 90])
    pool = pe.gen_grid(cfg)
    keys = [pe.layup_key(c, cfg["n_plies"]) for c in pool]
    assert len(keys) == len(set(keys)), "池内出现重复物理铺层"
    bad = [c for c in pool
           if not (ad.check_rules(c, cfg["n_plies"])["symmetric"]
                   and ad.check_rules(c, cfg["n_plies"])["balanced"])]
    assert bad == [], f"池内仍有 {len(bad)} 组不满足对称+均衡"
    # 5 档角折角后只剩 4 个不同值：4^4 = 256，过滤后应显著更少
    assert 20 <= len(pool) <= 60


def test_split_is_disjoint_by_layup_key():
    """A1：按铺层键划分后，训练/测试/候选池三集交集必须为空（此前有 30/91 的重叠）。"""
    cfg = _cfg()
    pool = pe.gen_grid(cfg)
    rng = np.random.default_rng(0)
    train, test, screen = pe.split_disjoint(pool, 0, 0, 3000, rng)
    k = lambda c: pe.layup_key(c, cfg["n_plies"])          # noqa: E731
    kt, ke, ks = {k(c) for c in train}, {k(c) for c in test}, {k(c) for c in screen}
    assert kt & ke == set(), "训练集与测试集存在同一物理铺层"
    assert kt & ks == set(), "训练集与候选池存在同一物理铺层"
    assert ke & ks == set(), "测试集与候选池存在同一物理铺层"


def test_summary_reports_three_statistics_consistently():
    """A3：命中率、含失败均值、命中时中位数必须分开报，且口径一致。"""
    vals = [4.257193, None, 4.176562, None, 4.257193]     # None = 未命中
    runs = []
    for v in vals:
        rf = {"found_feasible": v is not None,
              "best_verified_lam": v, "loss_pct": None, "real_evaluations": 5}
        rand = {"best_verified_lam_median": 2.0, "loss_pct": None, "fail_runs": 1,
                "n_repeat": 10, "expected_lam": 2.0, "real_evaluations": 5}
        runs.append({"seed": 0, "by_budget": {"5": {
            "rf": rf, "random": rand,
            "reference_full_sweep": {"best_lam": 4.254851}}},
            "counts": {}, "model": {}, "timing_s": {}})
    s = pe.summarize(runs)["5"]
    assert s["rf_hit_rate"] == pytest.approx(3 / 5)
    assert s["rf_mean_score"] == pytest.approx((4.257193 + 0 + 4.176562 + 0 + 4.257193) / 5)
    assert s["rf_median_when_hit"] == pytest.approx(4.257193)
    assert s["rf_mean_score"] <= s["rf_median_when_hit"]
    assert s["random_mean_score"] == pytest.approx(2.0)
    assert s["random_hit_rate"] == pytest.approx(0.9)

# -*- coding: utf-8 -*-
"""B01 回归：同层数下，"不可行但承载更高"的候选不得挤掉可行候选。"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad  # noqa: E402


def _cand(n, lam, feasible, tag):
    return {
        "n_plies": n, "areal": ad.areal_weight(n), "lam": lam,
        "ncr": lam * 500.0, "fi": 1e-6, "feasible": feasible,
        "theta4": [-45.0, 45.0, 45.0, 0.0], "layup": f"[{tag}]s",
        "method": "fake", "seconds": 0.0, "seed": 0,
    }


def test_infeasible_high_load_loses_to_feasible(monkeypatch):
    """每层都返回两个候选：先给不可行的高承载，再给可行的低承载。"""
    calls = {"n": 0}

    def fake_optimize_at_n(n_plies, nx, ny, sf, seed=None, method="ga"):
        # 第 4 层给一个"lam 高但不可行"，第 6 层给可行
        if n_plies == 4:
            return _cand(4, 9.0, False, "high-but-infeasible")
        if n_plies == 6:
            return _cand(6, 1.9, True, "feasible")
        return _cand(n_plies, 0.1, False, "weak")

    monkeypatch.setattr(ad, "optimize_at_n", fake_optimize_at_n)
    sel = ad.select_design(500, 500, 1.5, fast=True, methods=("fake",))

    assert sel["primary"] is not None, "应当找到可行方案"
    assert sel["primary"]["n_plies"] == 6
    assert sel["primary"]["feasible"] is True
    # 第 4 层那行必须记成不可行，而不是因为 lam=9 就成为候选
    row4 = next(r for r in sel["rows"] if r["n_plies"] == 4)
    assert row4["feasible"] is False
    assert row4["lam"] == pytest.approx(9.0)

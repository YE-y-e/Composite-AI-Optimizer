"""失败状态贯穿后处理的守卫（第四版复核清单 2-①）。"""
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "codes"))
import Auto_Design as ad  # noqa: E402


def test_select_design_rejects_empty_methods():
    with pytest.raises(ad.DesignInputError):
        ad.select_design(500, 500, 1.5, fast=True, methods=())


def test_select_design_rejects_bad_n_max():
    with pytest.raises(ad.DesignInputError):
        ad.select_design(500, 500, 1.5, fast=True, n_max=3)


def test_grid_元数据不丢状态(monkeypatch):
    """grid 路径：最终结果应直接复用复核结果，带 grid_checked / grid_evaluated，
    且不得对同一候选重复评价（评价次数 == 唯一候选数）。"""
    calls = []
    real = ad.evaluate_candidate

    def counting(theta, n_plies, nx, ny, sf):
        calls.append(tuple(round(float(x), 6) for x in theta))
        return real(theta, n_plies, nx, ny, sf)

    monkeypatch.setattr(ad, "evaluate_candidate", counting)
    r = ad.optimize_at_n(12, 500, 500, 1.5, method="grid")
    assert r["grid_checked"] == ad.GRID_TOPK
    assert r["grid_evaluated"] == len(set(calls)), (r["grid_evaluated"], len(calls))
    assert len(calls) == len(set(calls)), "同一候选被重复评价"
    assert "grid_no_feasible" not in r        # 本例有可行解


def test_全部不可行时保留状态(monkeypatch):
    """人为把 feasible 全置 False，确认返回的是"最好的不可行解"且带 grid_no_feasible。"""
    real = ad.evaluate_candidate

    def no_feasible(*a, **k):
        ev = dict(real(*a, **k))
        ev["feasible"] = False
        return ev

    monkeypatch.setattr(ad, "evaluate_candidate", no_feasible)
    r = ad.optimize_at_n(12, 500, 500, 1.5, method="grid")
    assert r.get("grid_no_feasible") is True
    assert r["feasible"] is False
    assert math.isfinite(r["ncr"])

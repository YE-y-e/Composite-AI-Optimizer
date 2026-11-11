"""C1：公共输入校验（奇数层 / 零载荷 / 非有限角度 / 负载荷 / 非正 SF）。

这些用例锁定三件事：
1. 非法输入给出 `DesignInputError`（领域错误），而不是 `LinAlgError` 之类的底层异常；
2. 奇数层被明确拒绝——历史上 `symmetric_stack(..., 9)` 会静默返回 8 层，
   而厚度仍按 9 层算，导致"刚度 8 层、重量 9 层"的错配；
3. 合法输入的结果与校验前一致（层数、厚度、λ 都不变）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import Auto_Design as ad  # noqa: E402


def test_odd_layers_rejected():
    with pytest.raises(ad.DesignInputError):
        ad.symmetric_stack([-45, 45, 45, -90], 9)
    with pytest.raises(ad.DesignInputError):
        ad.evaluate_candidate([-45, 45, 45, -90], 9, 500, 500, 1.5)
    with pytest.raises(ad.DesignInputError):
        ad.validate_design([-45, 45, 45, -90], 11, 500, 500, 1.5)


def test_non_integer_layers_rejected():
    for bad in (7.5, "8", None, True):
        with pytest.raises(ad.DesignInputError):
            ad.validate_design([-45, 45, 45, -90], bad, 500, 500, 1.5)


def test_layers_outside_scan_range_rejected():
    with pytest.raises(ad.DesignInputError):
        ad.validate_design([-45, 45, 45, -90], ad.N_MIN - 2, 500, 500, 1.5)
    with pytest.raises(ad.DesignInputError):
        ad.validate_design([-45, 45, 45, -90], ad.N_MAX + 2, 500, 500, 1.5)


def test_zero_load_gives_domain_error_not_linalg():
    with pytest.raises(ad.DesignInputError) as ei:
        ad.evaluate_candidate([-45, 45, 45, -90], 8, 0, 0, 1.5)
    assert "载荷为零" in str(ei.value)


def test_non_finite_angles_rejected():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ad.DesignInputError):
            ad.evaluate_candidate([bad, 0.0, 0.0, 0.0], 8, 500, 500, 1.5)


def test_negative_load_and_bad_sf_rejected():
    with pytest.raises(ad.DesignInputError):
        ad.evaluate_candidate([-45, 45, 45, -90], 8, -500, 500, 1.5)
    for sf in (0.0, -1.0, math.nan):
        with pytest.raises(ad.DesignInputError):
            ad.evaluate_candidate([-45, 45, 45, -90], 8, 500, 500, sf)


def test_wrong_number_of_angles_rejected():
    with pytest.raises(ad.DesignInputError):
        ad.validate_design([-45, 45], 8, 500, 500, 1.5)


def test_legal_input_unchanged():
    """合法输入的层数/厚度/λ 与校验前一致（回归锚点）。"""
    ev8 = ad.evaluate_candidate([-45, 45, 45, -90], 8, 500, 500, 1.5)
    assert len(ev8["stack"]) == 8
    assert ev8["thickness"] == pytest.approx(8 * ad.T_PLY, rel=0, abs=1e-12)
    assert ev8["lam"] == pytest.approx(1.24358, rel=0, abs=5e-4)

    ev10 = ad.evaluate_candidate([-45, 45, 45, -90], 10, 500, 500, 1.5)
    assert len(ev10["stack"]) == 10
    assert ev10["thickness"] == pytest.approx(10 * ad.T_PLY, rel=0, abs=1e-12)
    assert ev10["lam"] == pytest.approx(2.41066, rel=0, abs=5e-4)

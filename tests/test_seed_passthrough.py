"""C5 最小修复的回归：seed 必须真正传进 GA，缺省时保持历史值。

不跑真正的 GA（一次约 90 s），把 ``pymoo_minimize`` 与 ``scipy_minimize``
都换成桩，只验证"传进去的 seed 是不是我们给的那个"。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import Auto_Design as ad  # noqa: E402


class _Pop:
    def __init__(self, x):
        self._x = np.atleast_2d(np.asarray(x, float))

    def get(self, key):
        if key == "X":
            return self._x
        if key == "F":
            return np.zeros((self._x.shape[0], 1))
        raise KeyError(key)


class _Res:
    def __init__(self, x):
        self.pop = _Pop(x)


@pytest.fixture()
def stub_solver(monkeypatch):
    seen = {}

    def fake_minimize(problem, algo, *args, **kwargs):
        seen["seed"] = kwargs.get("seed")
        return _Res([[-45.0, 45.0, 45.0, -90.0]])

    def fake_scipy(neg, x0, **kwargs):
        class R:
            x = np.asarray(x0, float)
            success = True
        return R()

    monkeypatch.setattr(ad, "pymoo_minimize", fake_minimize)
    monkeypatch.setattr(ad, "scipy_minimize", fake_scipy)
    return seen


def test_explicit_seed_reaches_ga(stub_solver):
    ad.optimize_layup(10, 500, 500, 750.0, seed=7)
    assert stub_solver["seed"] == 7


def test_default_seed_falls_back_to_module_constant(stub_solver):
    ad.optimize_layup(10, 500, 500, 750.0, seed=None)
    assert stub_solver["seed"] == ad.SEED


def test_optimize_at_n_forwards_seed(stub_solver):
    ad.optimize_at_n(10, 500, 500, 1.5, seed=13, method="ga")
    assert stub_solver["seed"] == 13


def test_optimize_at_n_default_seed_matches_constant(stub_solver):
    ad.optimize_at_n(10, 500, 500, 1.5, method="ga")
    assert stub_solver["seed"] == ad.SEED

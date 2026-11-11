"""C2/C3 回归：空候选要早退，参照最优只取可行候选。

背景（第三版清单 C2/C3）：
- C2：连续域经"对称+均衡"过滤后可能为空，旧代码在空池判断之前就调 `features`，
  会抛 IndexError；要求训练/复核前给出结构化状态。
- C3：参照最优曾用 `pool_lam.max()`（不管可行），当 SF 大到全不可行时会返回一个
  "不可行方案的大 λ"当基准，制造出代理永远追不上的假差距；应只在可行集上取。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "codes"))

import proxy_experiment as px  # noqa: E402

BASE = dict(
    n_plies=12, nx=500, ny=500, m=16, angles=(-90, -45, 0, 45, 90),
    n_train=20, n_test=8, n_pool=16, budget=(8,), n_repeat=3,
    enforce_balance=True, domain="grid",
)


def test_continuous_domain_returns_insufficient_status():
    """连续域经均衡过滤后为空 → 返回 insufficient_candidates，而不是崩。"""
    cfg = dict(BASE, domain="continuous", n_raw=3000)
    out = px.run_seed(0, cfg)
    assert out["status"] == "insufficient_candidates"
    assert out["counts"]["train"] < 2 or out["counts"]["test"] < 1
    assert out["by_budget"] == {}


def test_reference_best_is_feasible_only():
    """全不可行（SF 大到判据超过该域最大 λ）时，参照最优必须是 None 而不是大 λ。"""
    out = px.run_seed(0, dict(BASE, sf=10.0))
    ref = out["by_budget"]["8"]["reference_full_sweep"]
    assert ref["n_feasible"] == 0
    assert ref["best_lam"] is None
    assert out["by_budget"]["8"]["rf"]["loss_pct"] is None


def test_reference_best_matches_feasible_max():
    """有可行解时，参照最优等于可行集内最大 λ。"""
    out = px.run_seed(0, dict(BASE, sf=1.5))
    ref = out["by_budget"]["8"]["reference_full_sweep"]
    assert ref["n_feasible"] > 0
    assert ref["best_lam"] == pytest.approx(4.2572, abs=5e-4)


def test_normal_run_reports_all_seed_fields():
    """正常路径的返回结构保持不变（顶层字段不能被改动掉）。"""
    out = px.run_seed(0, dict(BASE, sf=1.5))
    for key in ("seed", "counts", "model", "timing_s", "by_budget"):
        assert key in out
    assert "status" not in out
    assert set(out["counts"]) == {"pool_generated", "train", "test", "screen_pool"}

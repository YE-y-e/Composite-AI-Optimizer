# -*- coding: utf-8 -*-
"""运行模式开关：界面上的「在本机运行完整优化」必须真的切到 GA。

这条用例来自一个真实 bug：`_demo_now()` 只加进了说明文案，计算分支仍读 `IS_DEMO`，
于是勾选开关后界面写着"本机完整计算"，实际跑的却还是离线预设。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from conftest import APP
from apptest_helpers import texts

TIMEOUT = 600


def _params_page():
    """进入参数页（自定义工况），返回 AppTest。"""
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    at.button(key="pick_custom").click().run()
    return at


def test_demo_branch_is_default():
    """默认不勾选，走演示路径，不算 GA。"""
    at = _params_page()
    at.button(key="run_btn").click().run()
    assert at.session_state["_res_cache"]["res"]["source"] in ("preset", "estimate")
    assert "运行模式 demo" in texts(at)


def test_switch_to_local_actually_runs_ga():
    """勾选后必须走 auto_design：来源变 ga/pure0，用时明显超过预设读取。"""
    at = _params_page()
    at.checkbox(key="run_local").check().run()
    assert any("快速模式" in str(c.label) for c in at.checkbox)
    assert "本机完整计算" in texts(at)

    at.button(key="run_btn").click().run()
    res = at.session_state["_res_cache"]["res"]
    assert res["source"] in ("ga", "pure0"), res["source"]
    assert "运行模式 local" in texts(at)
    assert "完整优化计算" in texts(at)
    assert at.session_state["_res_cache"]["elapsed"] > 1.0


def test_switch_back_returns_to_demo():
    """取消勾选后回到演示路径，不残留 GA 结果。"""
    at = _params_page()
    at.checkbox(key="run_local").check().run()
    at.button(key="run_btn").click().run()
    at.button(key="back_params").click().run()
    at.checkbox(key="run_local").uncheck().run()
    at.button(key="run_btn").click().run()
    assert at.session_state["_res_cache"]["res"]["source"] in ("preset", "estimate")
    assert not at.exception

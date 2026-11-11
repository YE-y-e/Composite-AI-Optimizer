"""P1-4 回归：载荷输入、范围说明、预设带入、参数改动提示、返回与重设计。

用 streamlit 自带的 AppTest 跑真实脚本（APP_MODE=demo），不运行 GA。
"""

from __future__ import annotations

import pytest

from apptest_helpers import enter_params, run_design, texts


# --- 默认值与控件构成 ----------------------------------------------------
def test_defaults_seeded_for_slider_and_number(at):
    assert at.session_state["step"] == "mode"
    for key, want in (("nx", 500), ("ny", 500), ("sf", 1.5)):
        assert at.session_state[f"{key}_slider"] == want
        assert at.session_state[f"{key}_num"] == want


def test_three_sliders_and_three_number_inputs(at):
    enter_params(at)
    assert [s.key for s in at.slider][:3] == ["nx_slider", "ny_slider", "sf_slider"]
    assert [n.key for n in at.number_input][:3] == ["nx_num", "ny_num", "sf_num"]


def test_range_and_hint_caption(at):
    enter_params(at)
    blob = texts(at)
    assert "0–2000 N/m（步长 10）" in blob
    assert "1.00–2.00（步长 0.05）" in blob
    assert "双向同步" in blob


# --- 滑块与精确值双向联动 ------------------------------------------------
def test_slider_updates_number(at):
    enter_params(at)
    at.slider(key="nx_slider").set_value(1230).run()
    assert at.session_state["nx_num"] == 1230


def test_number_updates_slider_with_off_step_value(at):
    enter_params(at)
    at.number_input(key="nx_num").set_value(1233).run()
    assert at.session_state["nx_slider"] == 1233


# --- 预设带入 ------------------------------------------------------------
def test_preset_fills_all_six_keys(at):
    enter_params(at, preset=1)          # 1000/200, SF 1.20
    for key, want in (("nx", 1000), ("ny", 200), ("sf", 1.2)):
        assert at.session_state[f"{key}_slider"] == want
        assert at.session_state[f"{key}_num"] == want


def test_target_preview_follows_inputs(at):
    at.button(key="pick_custom").click().run()
    at.slider(key="nx_slider").set_value(1000).run()
    at.slider(key="ny_slider").set_value(800).run()
    at.slider(key="sf_slider").set_value(2.0).run()
    blob = texts(at)
    assert "2,000.0" in blob          # max(1000, 800) * 2.00
    assert "随上方滑块实时更新" in blob


# --- 演示模式的算例清单 --------------------------------------------------
def test_demo_preset_list_excludes_isolated_keys(at):
    enter_params(at)
    blob = texts(at)
    for official in ("500/500", "1000/200", "1500/800"):
        assert official in blob
    for isolated in ("1040/1040", "1070/870"):
        assert isolated not in blob


# --- 参数改动提示与旧结果隔离 -------------------------------------------
def test_stale_parameter_hint(at):
    enter_params(at)
    at.button(key="run_btn").click().run()
    assert at.session_state["step"] == "result"

    at.button(key="back_params").click().run()
    assert "参数已改动" not in texts(at)

    at.slider(key="nx_slider").set_value(700).run()
    assert "参数已改动" in texts(at)
    assert "结果页上仍是上一次的工况" in texts(at)


def test_second_design_uses_new_job(at):
    run_design(at, preset=0)
    job1 = dict(at.session_state["design_job"])
    assert (job1["nx"], job1["ny"], job1["sf"]) == (500, 500, 1.5)
    assert len(at.session_state["history"]) == 1

    at.button(key="restart_top").click().run()
    assert at.session_state["step"] == "mode"
    assert at.session_state["design_job"] is None
    assert at.session_state["calc_mode"] is None
    assert at.session_state["scene"] is None
    assert not at.session_state["_res_cache"]
    assert len(at.session_state["history"]) == 1      # 历史不清空

    run_design(at, preset=2)                          # 1500/800, SF 2.00
    job2 = dict(at.session_state["design_job"])
    assert (job2["nx"], job2["ny"], job2["sf"]) == (1500, 800, 2.0)
    assert "Nx = 1500 N/m" in texts(at)
    assert len(at.session_state["history"]) == 2


def test_entering_custom_resets_loads(at):
    enter_params(at, preset=2)          # 1500/800/2.0
    at.button(key="back_mode").click().run()
    assert at.session_state["step"] == "mode"
    at.button(key="pick_custom").click().run()
    for key, want in (("nx", 500), ("ny", 500), ("sf", 1.5)):
        assert at.session_state[f"{key}_slider"] == want
        assert at.session_state[f"{key}_num"] == want

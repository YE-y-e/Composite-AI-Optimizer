"""P0 收尾测试：不可行分支、结论判定、隔离预设、ABAQUS 条件性对比。

测试不启动界面、不运行 GA：用最小 streamlit 桩执行 app.py，
拿到真实函数对象后做纯逻辑断言。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

CODES = Path(__file__).resolve().parents[1] / "codes"
APP_PATH = CODES / "app.py"


# --- 最小 streamlit 桩 ---------------------------------------------------
class _Stop(Exception):
    """st.stop() / st.rerun() 在桩里的替身。"""


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Status(_Ctx):
    def update(self, **kwargs):
        return None


class _State(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        self.pop(name, None)


def _noop(*args, **kwargs):
    return None


def _raise_stop(*args, **kwargs):
    raise _Stop()


def _cache_data(func=None, **kwargs):
    def wrap(fn):
        def inner(*a, **k):
            return fn(*a, **k)

        inner.clear = lambda: None
        inner.__wrapped__ = fn
        return inner

    return wrap if func is None else wrap(func)


def _columns(spec, **kwargs):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]


def _tabs(labels, **kwargs):
    return [_Ctx() for _ in labels]


def _slider(*args, **kwargs):
    if "value" in kwargs:
        return kwargs["value"]
    return args[2] if len(args) > 2 else 0


def _number_input(*args, **kwargs):
    if "value" in kwargs:
        return kwargs["value"]
    return args[2] if len(args) > 2 else 0


def _checkbox(*args, **kwargs):
    return bool(kwargs.get("value", False))


def _radio(label, options, index=0, **kwargs):
    return options[index]


def _selectbox(label, options, index=0, **kwargs):
    return options[index]


def _button(*args, **kwargs):
    return False


def _build_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _State()
    for name in (
        "set_page_config", "markdown", "write", "caption", "text", "title",
        "header", "subheader", "divider", "warning", "error", "success", "info",
        "metric", "dataframe", "table", "image", "plotly_chart", "pyplot",
        "download_button", "file_uploader", "text_input", "progress", "toast",
        "code", "latex", "audio", "video", "json", "empty_state",
    ):
        setattr(st, name, _noop)
    st.cache_data = _cache_data
    st.cache_resource = _cache_data
    st.columns = _columns
    st.tabs = _tabs
    st.slider = _slider
    st.number_input = _number_input
    st.checkbox = _checkbox
    st.radio = _radio
    st.selectbox = _selectbox
    st.button = _button
    st.form_submit_button = _button
    st.stop = _raise_stop
    st.rerun = _raise_stop
    st.experimental_rerun = _raise_stop
    st.status = lambda *a, **k: _Status()
    st.spinner = lambda *a, **k: _Ctx()
    st.container = lambda *a, **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx()
    st.form = lambda *a, **k: _Ctx()
    st.empty = lambda *a, **k: _Ctx()
    st.sidebar = _Ctx()
    return st


def _load_app():
    """执行 app.py，返回其命名空间（脚本在 st.stop() 处停住）。"""
    codes = str(CODES)
    if codes not in sys.path:
        sys.path.insert(0, codes)
    previous = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _build_stub()
    try:
        namespace = {"__name__": "app_under_test", "__file__": str(APP_PATH)}
        source = APP_PATH.read_text(encoding="utf-8")
        try:
            exec(compile(source, str(APP_PATH), "exec"), namespace)
        except _Stop:
            pass
    finally:
        if previous is not None:
            sys.modules["streamlit"] = previous
        else:
            sys.modules.pop("streamlit", None)
    return namespace


@pytest.fixture(scope="module")
def app():
    assert APP_PATH.is_file(), f"找不到工作文件 {APP_PATH}"
    return _load_app()


def _scan_rows(ncr_last: float = 600.0):
    """合法扫描记录，但全部 ok=False（无可行层数）。

    末层（N = 20）的 Ncr 取 600 N/m，低于本测试用的目标载荷，
    这样"全程不可行"在物理上与数据自洽，need_plies 才会算出 > 20。
    """
    rows = []
    for n in range(4, 21):
        ncr = ncr_last * (n / 20.0) ** 3
        rows.append({"N": n, "t": n * 0.125, "ncr": ncr,
                     "lam": ncr / 1000.0, "ok": False})
    return rows


# --- P0-A：不可行分支 ----------------------------------------------------
def test_auto_design_infeasible(app, monkeypatch):
    ad = app["ad"]
    monkeypatch.setattr(ad, "scan_pure_zero", lambda nx, ny, target: _scan_rows())
    app["auto_design"].clear()

    res = app["auto_design"](500, 500, 1.5, False)

    assert res["status"] == "infeasible"
    assert res["nx"] == 500.0
    assert res["ny"] == 500.0
    assert res["sf"] == 1.5
    assert res["target"] == pytest.approx(750.0)
    assert len(res["rows"]) == 17 and all(not r["ok"] for r in res["rows"])
    assert res["rows"][-1]["ncr"] > 0
    assert res["need_plies"] > ad.N_MAX
    assert res["source"] == "ga"


def test_demo_design_infeasible(app, monkeypatch):
    ad = app["ad"]
    monkeypatch.setattr(ad, "scan_pure_zero", lambda nx, ny, target: _scan_rows())
    app["demo_design"].clear()

    res = app["demo_design"](1000, 200, 1.2)

    assert res["status"] == "infeasible"
    assert res["demo"] is True
    assert res["target"] == pytest.approx(1200.0)
    assert res["source"] == "estimate"
    assert res["need_plies"] > ad.N_MAX


def test_infeasible_verdict_is_not_pass(app):
    verdict = app["verdict_of"]({"status": "infeasible", "nx": 500, "ny": 500})
    assert verdict["overall"] == "infeasible"
    assert verdict["buckling"] is None
    assert verdict["strength"] is None


def test_error_verdict_is_not_pass(app):
    verdict = app["verdict_of"]({"status": "error"})
    assert verdict["overall"] == "error"
    assert verdict["buckling"] is None


# --- P0-B：结论判定 ------------------------------------------------------
def _result(status="ok", source="ga", ncr=1000.0, target=800.0, fi=0.5):
    return {"status": status, "source": source, "ncr": ncr,
            "target": target, "fi": fi, "stack": []}


@pytest.mark.parametrize(
    "kwargs,overall,buckling,strength",
    [
        (dict(ncr=1000.0, fi=1.2), "strength_fail", True, False),
        (dict(ncr=700.0, fi=1.2), "both_fail", False, False),
        (dict(ncr=700.0, fi=0.4), "buckling_fail", False, True),
        (dict(ncr=1000.0, fi=0.4), "pass", True, True),
    ],
)
def test_verdict_branches(app, kwargs, overall, buckling, strength):
    verdict = app["verdict_of"](_result(**kwargs))
    assert verdict["overall"] == overall
    assert verdict["buckling"] is buckling
    assert verdict["strength"] is strength


def test_estimate_source_is_never_pass(app):
    verdict = app["verdict_of"](_result(source="estimate", ncr=1000.0, fi=0.4))
    assert verdict["overall"] == "unverified"
    assert verdict["buckling"] is True
    assert verdict["strength"] is True


# --- P0-C：隔离两条来源未核实的预设 --------------------------------------
ISOLATED = {(1040, 1040, 1.5), (1070, 870, 1.5)}
OFFICIAL = {(500, 500, 1.5), (1000, 200, 1.2), (1500, 800, 2.0)}


def test_isolated_presets_not_in_runtime_table(app):
    assert set(app["DEMO_PRESETS"]) == OFFICIAL
    for key in ISOLATED:
        assert key not in app["DEMO_PRESETS"]


def test_isolated_presets_kept_verbatim(app):
    kept = app["UNVERIFIED_DEMO_PRESETS"]
    assert set(kept) == ISOLATED
    for key in ISOLATED:
        record = kept[key]
        assert record["n_plies"] == 14
        assert record["theta4"] == [-40.3, 49.8, -75.4, 51.0]
        assert record["ncr"] > record["base_ncr"] > 0
        assert record["thinner"]["n"] < record["n_plies"]


@pytest.mark.parametrize("key", sorted(ISOLATED))
def test_isolated_key_falls_back_to_estimate(app, key):
    nx, ny, sf = key
    res = app["demo_design"](nx, ny, sf)
    assert res["status"] == "ok"
    assert res["source"] == "estimate"
    assert res["isolated_preset"] is True
    assert "来源未核实" in res["source_label"]
    assert app["verdict_of"](res)["overall"] == "unverified"


@pytest.mark.parametrize("key", sorted(OFFICIAL))
def test_official_presets_still_hit(app, key):
    nx, ny, sf = key
    res = app["demo_design"](nx, ny, sf)
    assert res["source"] == "preset"
    assert res["isolated_preset"] is False


def test_params_page_lists_only_official_presets():
    source = APP_PATH.read_text(encoding="utf-8")
    marker = "本模式可用的离线算例"
    idx = source.index(marker)
    block = source[max(0, idx - 600):idx]
    assert "DEMO_PRESETS" in block
    assert "UNVERIFIED_DEMO_PRESETS" not in block


# --- P0-D：ABAQUS 条件性参考对比 -----------------------------------------
def test_aba_compare_no_case(app):
    cmp = app["aba_compare"]({"ncr": 1000.0, "stack": []}, 800, 600)
    assert cmp["state"] == "no_case"
    assert cmp["ref"] is None
    assert cmp["delta_pct"] is None


def test_aba_compare_stack_mismatch_has_no_delta(app):
    cmp = app["aba_compare"]({"ncr": 2128.6, "stack": [0.0] * 12}, 500, 500)
    assert cmp["state"] == "stack_mismatch"
    assert cmp["ref"] is not None
    assert cmp["ref_layup"]
    assert cmp["delta_pct"] is None


def test_aba_compare_conditional_keeps_number_but_flags_it(app):
    ad = app["ad"]
    stack = [float(a) for a in ad.symmetric_stack(
        [-44.9, 44.9, 45.2, -45.2], 12)]
    cmp = app["aba_compare"]({"ncr": 2128.6, "stack": stack}, 500, 500)
    assert cmp["state"] == "conditional"
    assert cmp["delta_pct"] == pytest.approx(0.79, abs=0.02)
    assert "未验证" in cmp["note"]


def test_aba_states_never_claim_verification(app):
    for text in app["ABA_STATE"].values():
        assert "验证通过" not in text
        assert "已验证" not in text
    assert "未验证" in app["ABA_STATE"]["conditional"]

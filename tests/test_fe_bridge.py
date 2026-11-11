"""网页内「本机 Abaqus 复核」桥接与执行路径保持的回归测试。"""

import shutil
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "codes" / "app.py"


@pytest.fixture(scope="module")
def app_ns():
    import importlib.util

    spec = importlib.util.spec_from_file_location("p0_app_ns_fe", APP)
    # app.py 顶层会执行 Streamlit 脚本，这里复用已有测试的加载器
    from test_p0_closure import _load_app  # noqa: PLC0415

    return _load_app()


def test_abaqus_cmd_honours_env(monkeypatch, tmp_path, app_ns):
    fake = tmp_path / "abaqus.bat"
    fake.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setenv("ABAQUS_CMD", str(fake))
    assert app_ns["_abaqus_cmd"]() == str(fake)


def test_abaqus_cmd_returns_usable_path_or_none(monkeypatch, app_ns):
    """未配置时必须返回 None 或一个真实存在的 abaqus 可执行文件，不能瞎猜。"""
    monkeypatch.setenv("ABAQUS_CMD", str(Path("Z:/nope/abaqus.bat")))
    got = app_ns["_abaqus_cmd"]()
    assert got is None or Path(got).exists() or shutil.which(got)


def test_explicit_missing_abaqus_cmd_means_no_fe(monkeypatch, app_ns):
    """显式把 ABAQUS_CMD 指向不存在的路径 → 视为本机没有 Abaqus。"""
    monkeypatch.setenv("ABAQUS_CMD", str(Path("Z:/definitely/not/abaqus.bat")))
    assert app_ns["_abaqus_cmd"]() is None


def test_empty_abaqus_cmd_disables_fe(monkeypatch, app_ns):
    """ABAQUS_CMD 设成空串 → 关掉现场复核（用于演示"只有离线算例"的形态）。"""
    monkeypatch.setenv("ABAQUS_CMD", "   ")
    assert app_ns["_abaqus_cmd"]() is None


def test_mode_choice_survives_widget_cleanup(app_ns):
    st = app_ns["st"]
    st.session_state.clear()
    st.session_state["run_local"] = True
    app_ns["_remember_mode"]()
    # 模拟结果页：勾选框不再渲染，Streamlit 清掉了 run_local
    st.session_state.pop("run_local", None)
    assert app_ns["_local_run"]() is True


def test_fe_module_builds_inp():
    import sys

    sys.path.insert(0, str(APP.parent))
    import abaqus_fe as afe  # noqa: PLC0415

    text = afe.build_inp([-45, 45, 45, -45], 12, 800, 600)
    plies = [ln for ln in text.splitlines() if ln.startswith("0.125")]
    assert len(plies) == 12
    assert "EDNOR, 0.75" in text  # Ny/Nx = 600/800

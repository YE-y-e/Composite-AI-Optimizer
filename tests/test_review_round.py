"""本轮审查任务的回归用例（术语/来源、模型假设、不可行措辞、资源与导出）。"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pytest

from apptest_helpers import run_design, texts
from conftest import APP, at  # noqa: F401  (APP 供路径断言，at 是 fixture)

SRC = Path(APP).read_text(encoding="utf-8")


# ---------- P0-1 术语与单位 ----------
def test_density_called_material_density():
    """ρ = 1600 kg/m³ 必须叫材料密度；面密度只用于 kg/㎡。"""
    assert "材料密度" in SRC
    assert "面密度</td>" not in SRC
    assert "kg/㎡" in SRC


def test_units_present():
    for token in ("Ncr (N/m)", "总厚 (mm)", "kg/m³", "MPa"):
        assert token in SRC


def test_strength_allowables_come_from_solver():
    """强度许用值取自 Auto_Design 的常量，不在页面里硬编码。"""
    for name in ("ad.XT", "ad.XC", "ad.YT", "ad.YC", "ad.S12"):
        assert name in SRC


# ---------- P0-2 模型假设 ----------
def test_assumptions_present_and_folded():
    block = SRC.split("def _assumptions_text()")[1].split("def _design_conditions")[0]
    for key in ("几何与边界", "材料与铺层", "求解方法", "未覆盖的真实结构因素"):
        assert key in block
    assert "不是对应产品的完整有限元模型" in block
    assert 'st.expander("模型假设与适用范围")' in SRC


# ---------- P0-3 / P0-4 来源与措辞 ----------
def test_source_labels_distinguish_paths():
    for label in ("完整优化计算", "离线预设算例", "纯 0° 简化估算", "未运行 GA"):
        assert label in SRC


def test_no_marketing_words():
    for bad in ("智能设计", "最优方案", "自动寻优", "一键", "赋能"):
        assert bad not in SRC


def test_local_switch_wording():
    assert "在本次会话运行完整优化" in SRC
    assert "计算发生在托管该页面的主机上" in SRC


def test_fast_timing_consistent():
    assert "约 10 s" in SRC and "约 5 s" not in SRC


# ---------- P1-1 不可行分支措辞 ----------
def test_infeasible_wording_not_global():
    block = SRC.split("# 不可行分支")[1].split("# 结果区")[0]
    assert "未找到可行层数" in block
    assert "没有搜索铺层角度" in block
    assert "不能断定该工况整体无可行铺层" in block
    assert "无法满足 Ncr_target" not in SRC


# ---------- P1 稳定性 / 资源 ----------
def test_standard_path_takes_lock():
    """标准模式也进同一把锁，避免快速模式降级漏给并发会话。"""
    fn = SRC.split("def auto_design")[1].split("def demo_design")[0]
    # 逐层数主流程在锁内调用 select_design（快速模式的参数降级也在同一把锁里）
    assert "with _FAST_LOCK:" in fn
    assert "ad.select_design(" in fn
    guard = SRC.split("def _ga_guard")[1].split("def _lower_accuracy")[0]
    assert "with _FAST_LOCK:" in guard


def test_mode_images_exist_on_disk():
    names = set(re.findall(r'"(mode_[a-z0-9_]+\.png)"', SRC))
    assert len(names) >= 4
    roots = [Path(APP).parent / "figures", Path(APP).parents[1] / "figures"]
    for name in names:
        assert any((root / name).exists() for root in roots), name


def test_missing_image_placeholder():
    assert "示意图缺失" in SRC and "mode-img-missing" in SRC


def test_pdf_font_failure_is_explicit():
    assert "未找到中文字体" in SRC
    assert "未安装 fpdf2" in SRC


# ---------- P1 设计记录与导出 ----------
def test_history_has_required_fields():
    for col in ("运行模式", "来源", "求解方法", "优化规模", "计算用时 (s)",
                "记录号", "版本"):
        assert f'"{col}"' in SRC


def test_history_row_and_csv_export(at):
    run_design(at, preset=0)
    hist = at.session_state["history"]
    assert hist, "设计记录为空"
    blob = hist[0]
    assert blob["来源"] == "离线预设算例"
    for col in ("运行模式", "求解方法", "优化规模", "计算用时 (s)", "记录号", "版本"):
        assert col in blob

    import test_p0_closure as closure
    ns = closure._load_app()
    data = ns["_history_csv"]([blob])
    assert data.startswith(b"\xef\xbb\xbf"), "CSV 应带 UTF-8 BOM 以便 Excel 打开"
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert rows and rows[0]["来源"] == "离线预设算例"
    assert rows[0]["版本"] == "v1.0.0"


# ---------- P1 PDF 导出 ----------
def _app_ns():
    import test_p0_closure as closure
    return closure._load_app()


def test_pdf_builds_and_carries_source_fields():
    ns = _app_ns()
    res = ns["demo_design"](500, 500, 1.5)
    data = ns["build_report_pdf"](res)
    assert data and data.startswith(b"%PDF")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        pytest.skip("PyMuPDF 不可用，跳过 PDF 文本抽取")
    with fitz.open(stream=data, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    for token in ("运行模式", "计算来源", "校核结论", "面密度"):
        assert token in text
    assert "无法" not in text.split("校核结论")[0][-40:]


def test_pdf_returns_none_when_font_missing(monkeypatch):
    ns = _app_ns()
    monkeypatch.setitem(ns, "_cjk_font_path", lambda: None)
    res = ns["demo_design"](500, 500, 1.5)
    assert ns["build_report_pdf"](res) is None


# ---------- 页面级 ----------
def test_params_page_shows_strength_table(at):
    at.button(key="pick_preset_0").click().run()
    assert at.session_state["step"] == "params"
    blob = texts(at)
    assert "强度许用值" in blob
    assert "X<sub>t</sub> / X<sub>c</sub>" in blob

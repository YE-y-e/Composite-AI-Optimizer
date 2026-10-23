"""P2 界面整理回归：状态条、表格样式、页签分工。

这些断言原先散在临时脚本里，改成常驻用例，避免以后整理样式时又退化。
"""

from __future__ import annotations

from apptest_helpers import run_design, texts


def test_result_page_drops_status_bar_and_keeps_elapsed_time(at):
    """计算完不再留一个整行状态框，用时并进设计条件行。"""
    run_design(at)
    assert not [s for s in at.status if "计算完成" in str(getattr(s, "label", ""))]
    assert "计算用时" in texts(at)


def test_design_result_tab_leads_with_verdict(at):
    """结论前置：校核表出现在推荐铺层之前。"""
    run_design(at)
    blob = texts(at)
    assert blob.index("校核项") < blob.index("推荐铺层序列")


def test_dmatrix_table_is_spec_html(at):
    """D 矩阵对比用与其它表格一致的 table.spec，而不是画布表格。"""
    run_design(at)
    blob = texts(at)
    assert "D 矩阵对比（同层数）" in blob
    assert "D16 (GPa·mm³)" in blob
    assert "厚度相同，Ncr 的差异全部来自 D 矩阵" in blob
    assert not [d for d in at.dataframe if "D11" in str(getattr(d, "value", ""))]


def test_third_tab_keeps_reference_section(at):
    run_design(at)
    blob = texts(at)
    assert "性能与参考对比" in blob
    assert "离线参考算例对比" in blob
    assert "相对差异" in blob

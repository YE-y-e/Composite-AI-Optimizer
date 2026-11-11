"""AppTest 小工具：拼接页面文本、进入第 2 步。"""

from __future__ import annotations

KINDS = ("markdown", "caption", "warning", "error", "success", "info", "text")


def texts(test) -> str:
    """把页面上所有可见文本拼成一段，方便做包含判断。"""
    out = []
    for kind in KINDS:
        for element in getattr(test, kind, []):
            value = getattr(element, "value", None)
            out.append(str(value if value is not None
                           else getattr(element, "body", "")))
    return "\n".join(out)


def enter_params(test, preset: int = 0):
    """点第 preset 张预设卡，进入参数页。"""
    test.button(key=f"pick_preset_{preset}").click().run()
    assert test.session_state["step"] == "params"
    return test


def run_design(test, preset: int = 0):
    """选预设 → 直接运行，停在结果页。"""
    enter_params(test, preset)
    test.button(key="run_btn").click().run()
    assert test.session_state["step"] == "result"
    return test

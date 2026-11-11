# -*- coding: utf-8 -*-
"""matplotlib 中文字体辅助：让 mathtext 里的中文回退到宋体。

本项目的图统一使用「中文 SimSun + 英文/数字 Times New Roman」，
rcParams 在各绘图脚本里直接写死：

    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
    })

问题在于：只要一段文本里出现成对的 `$...$`，matplotlib 会把**整段**交给
mathtext 解析（`Text._preprocess_math` -> `cbook.is_math_text`），而 mathtext
只认 `mathtext.rm/it/bf` 指定的那一个字体文件（Times New Roman），没有字体
回退，于是「N/m，对数轴」这类混排里的汉字会被替换成 ¤（并打印
"Font 'default' does not have a glyph ... substituting with a dummy symbol"）。

`install_cjk_fallback()` 给 `UnicodeFonts._get_glyph` 打一个小补丁：
遇到 CJK 区段的码位时先用宋体（SimSun）取字形，取不到再走原逻辑。
这样无需改写任何图内文字，混排标签也能正常显示中文。
"""
from __future__ import annotations

from matplotlib import font_manager
from matplotlib._mathtext import UnicodeFonts, get_unicode_index
from matplotlib.ft2font import FT2Font

# CJK 汉字、CJK 标点、全角标点、兼容汉字
_CJK_RANGES = (
    (0x2E80, 0x2EFF),   # CJK 部首补充
    (0x3000, 0x303F),   # CJK 标点（、。「」等）
    (0x3400, 0x4DBF),   # 扩展 A
    (0x4E00, 0x9FFF),   # 基本汉字
    (0xF900, 0xFAFF),   # 兼容汉字
    (0xFE30, 0xFE4F),   # 兼容形式
    (0xFF00, 0xFFEF),   # 全角字符（：（）等）
)


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def install_cjk_fallback(family: str = "SimSun") -> None:
    """让 mathtext 缺字时回退到 *family*（默认宋体）。可重复调用。"""
    if getattr(UnicodeFonts, "_cjk_fallback_font", None) is not None:
        return
    font = FT2Font(font_manager.findfont(family))
    orig = UnicodeFonts._get_glyph

    def _get_glyph(self, fontname, font_class, sym):
        try:
            uni = get_unicode_index(sym)
        except ValueError:
            uni = None
        if uni is not None and _is_cjk(uni) and font.get_char_index(uni) != 0:
            return font, uni, False
        return orig(self, fontname, font_class, sym)

    UnicodeFonts._get_glyph = _get_glyph
    UnicodeFonts._cjk_fallback_font = font

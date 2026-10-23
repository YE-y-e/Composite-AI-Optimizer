"""复合材料铺层优化工作台（Streamlit + Plotly）。

运行模式由环境变量 APP_MODE 明确指定，不按操作系统自动猜测：

    APP_MODE=local   调用 Auto_Design.py 跑完整的 NSGA-II 寻优（约 1 分钟）
    APP_MODE=demo    只读取事先算好的离线预设算例，不运行 GA（默认）

未设置或取值非法时一律按 demo 处理，避免部署环境意外触发耗时优化。

界面按「选择模式 → 参数设置 → 计算结果」三步走，结果页分设计结果、失稳模态、
性能对比三个页签。计算、曲线和校核结论都来自 angle_optimize.py 与
Auto_Design.py 的同一套结果，这里只负责展示。
"""

import contextlib
import io
import os
import re
import threading
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:                      # PDF 导出依赖；缺失时其余功能照常运行
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:       # pragma: no cover
    FPDF = None
    HAS_FPDF = False

import Auto_Design as ad
from angle_optimize import (
    A_PLATE,
    B_PLATE,
    DENSITY,
    E1,
    E2,
    G12,
    M_VERIFY,
    T_PLY,
    V12,
    BucklingSolver,
    navier_orthotropic,
)

# 环境判断
VALID_MODES = ("local", "demo")
DEFAULT_MODE = "demo"      # 与部署方式一致：不配置时不跑耗时优化


def _resolve_mode() -> str:
    """读取 APP_MODE，只接受 local / demo；缺失或非法一律退回 demo。"""
    raw = os.environ.get("APP_MODE", "").strip().lower()
    return raw if raw in VALID_MODES else DEFAULT_MODE


APP_MODE = _resolve_mode()                 # "local" | "demo"
IS_DEMO = APP_MODE == "demo"
_raw_mode = os.environ.get("APP_MODE", "").strip()
MODE_INVALID = bool(_raw_mode) and _raw_mode.lower() not in VALID_MODES

# 页面与样式
st.set_page_config(
    page_title="复合材料铺层优化工作台",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<style> .stApp { background-color: #FAFAFA !important; color: #0f172a !important; } </style>',
    unsafe_allow_html=True,
)

if MODE_INVALID:
    st.warning(
        f'环境变量 APP_MODE = "{_raw_mode}" 不是合法取值（只允许 local / demo），'
        f"已按 {APP_MODE} 运行。"
    )


def _local_run() -> bool:
    """本会话是否已从演示模式显式切换到本机完整计算。"""
    return bool(st.session_state.get("run_local", False))


def _demo_now() -> bool:
    """当前真实执行路径是否走演示分支（离线算例 / 纯 0° 估算）。"""
    return IS_DEMO and not _local_run()


def mode_text() -> str:
    """页面上显示的运行模式，与真实执行路径保持一致。"""
    if not IS_DEMO:
        return "local"
    return "local（本会话切换）" if _local_run() else "demo"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    color-scheme: light;
    --ink: #0f172a;
    --ink2: #334155;
    --ink3: #64748b;
    --paper: #f8fafc;
    --paper2: #ffffff;
    --paper3: #f1f5f9;
    --rule: #e2e8f0;
    --accent: #2563eb;
    --accent-deep: #1d4ed8;
    --accent-light: #eff6ff;
    --accent2: #f97316;
    --ok: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --sans: 'Inter', 'Segoe UI', 'Microsoft YaHei', sans-serif;
    --serif: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    --mono: 'JetBrains Mono', 'Consolas', 'Cascadia Mono', 'SF Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
.stApp { background: var(--paper) !important; color: var(--ink) !important; }

/* 保留框架骨架，只隐藏 Deploy / 菜单 / 页脚 */
[data-testid="stHeader"] { background: transparent; border-bottom: none; height: auto; }
[data-testid="stToolbar"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
#MainMenu,
[data-testid="stToolbarActions"],
footer { display: none !important; }

/* 展开 / 折叠箭头必须可见可点 */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000000;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
}

.block-container {
    max-width: 1240px;
    padding: 1.3rem 1.4rem 3.2rem 1.4rem;
}

/* 报头：深色版画式标题栏 */
.masthead {
    background: #0f172a;
    border: 1px solid #0f172a;
    border-radius: 2px;
    padding: 15px 20px 13px 20px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px 18px;
    flex-wrap: wrap;
}
.masthead-brand {
    font-family: var(--serif);
    font-size: 1.48rem;
    font-weight: 600;
    letter-spacing: 0.015em;
    color: #ffffff;
    line-height: 1.32;
}
.masthead-meta {
    font-family: var(--mono);
    font-size: 0.69rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: none;
    color: #94a3b8;
    white-space: nowrap;
}
.masthead-meta .sep { color: #334155; margin: 0 0.55rem; }
.masthead-sub {
    font-family: var(--mono);
    font-size: 0.67rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 8px 0 20px 2px;
}

/* 版块编号与标题 */
.sec-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 0 0 8px 0;
}
.sec-label b {
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    background: var(--accent-light);
    color: var(--accent);
    padding: 2px 6px;
    border-radius: 3px;
}
.sec-title {
    font-family: var(--serif);
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.35;
    margin: 0 0 12px 0;
}
.card-head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 14px;
}
.card-idx {
    font-family: var(--mono);
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    background: var(--accent-light);
    color: var(--accent);
    padding: 2px 6px;
    border-radius: 3px;
}
.card-title { font-size: 0.95rem; font-weight: 600; color: var(--ink); letter-spacing: 0.02em; }
.card-aside {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.63rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94a3b8;
    white-space: nowrap;
}

h1, h2, h3, h4, h5 { color: var(--ink); letter-spacing: 0; }
h3 { font-size: 1.02rem; font-weight: 600; margin: 0.1rem 0 0.6rem 0; }
h4 { font-size: 0.9rem; font-weight: 600; color: var(--ink2); margin: 1rem 0 0.5rem 0; }
p, li, label, span { line-height: 1.65; }

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: var(--paper2);
    border-right: 1px solid var(--rule);
    box-shadow: none;
}
[data-testid="stSidebarUserContent"] { padding: 1.3rem 1.15rem 2.2rem 1.15rem; }
.side-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.71rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 0 0 10px 0;
}
.side-head b {
    font-family: var(--mono);
    font-size: 0.65rem;
    background: var(--accent-light);
    color: var(--accent);
    padding: 2px 5px;
    border-radius: 3px;
    letter-spacing: 0.06em;
}
.side-spec { margin: 2px 0 4px 0; }
.side-spec div {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--paper3);
    font-size: 0.73rem;
}
.side-spec div:last-child { border-bottom: none; }
.side-spec span { color: var(--ink3); letter-spacing: 0.05em; white-space: nowrap; }
.side-spec em {
    font-style: normal;
    font-family: var(--mono);
    color: var(--ink2);
    font-weight: 500;
    text-align: right;
}

/* 控件标签：小号、加粗、不改变大小写（单位不可大写） */
[data-testid="stWidgetLabel"] p,
.stSlider label p,
.stSelectbox label p,
.stCheckbox label p {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: none;
    color: var(--ink2) !important;
    margin-bottom: 2px !important;
}
[data-testid="stCheckbox"] label p,
[data-testid="stRadio"] label p {
    font-size: 0.78rem !important;
    letter-spacing: 0 !important;
    text-transform: none;
    color: var(--ink2) !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSliderThumbValue"],
[data-testid="stSlider"] output {
    font-family: var(--mono) !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

/* 表单：去掉 Streamlit 默认圆角白卡，侧边栏本身即为面板 */
[data-testid="stForm"] {
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    padding: 0 !important;
}

/* 下拉框：直角、细边、聚焦主题色 */
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--paper) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    min-height: 40px;
    box-shadow: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stSelectbox"] [role="group"]:hover,
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    background: var(--paper) !important;
    border-color: #cbd5e1 !important;
}
[data-testid="stSelectbox"] [role="group"]:focus-within,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    background: var(--paper2) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12) !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] [role="group"] span,
[data-testid="stSidebar"] [data-baseweb="select"] span { color: var(--ink) !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--paper2) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    box-shadow: 0 12px 28px -10px rgba(15, 23, 42, 0.22) !important;
}
[data-baseweb="popover"] * { color: var(--ink) !important; }

/* 主按钮：实心主题色，直角，无渐变无位移 */
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent-deep) !important;
    border-radius: 3px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.14em;
    padding: 0.62rem 1rem !important;
    box-shadow: none !important;
    transition: background 0.15s ease;
}
[data-testid="stBaseButton-primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: var(--accent-deep) !important;
    transform: none !important;
    box-shadow: none !important;
    filter: none !important;
}
[data-testid="stBaseButton-primaryFormSubmit"] *,
[data-testid="stBaseButton-primary"] * { color: #ffffff !important; }

/* 次级按钮（PDF 导出）：描边样式 */
[data-testid="stDownloadButton"] button {
    width: 100%;
    background: var(--paper2) !important;
    border: 1px solid var(--ink) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    padding: 0.55rem 1rem !important;
    transition: background 0.15s ease, color 0.15s ease;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--ink) !important;
    color: #ffffff !important;
}
[data-testid="stDownloadButton"] button:hover * { color: #ffffff !important; }

/* 结果卡片 */
.st-key-card1, .st-key-card2, .st-key-card3 {
    background: var(--paper2);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 20px 22px 22px 22px !important;
    box-shadow: none;
}
.st-key-card1 h3, .st-key-card2 h3, .st-key-card3 h3 {
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}

/* 页签：下划线式 */
[role="tablist"] {
    gap: 26px !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    border-bottom: 1px solid var(--rule) !important;
}
[data-testid="stTab"] {
    height: auto !important;
    padding: 0 0 9px 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--ink3) !important;
    font-weight: 600 !important;
    font-size: 0.79rem !important;
    letter-spacing: 0.1em !important;
    border-bottom: 2px solid transparent !important;
    box-shadow: none !important;
}
[data-testid="stTab"]:hover { color: var(--ink) !important; background: transparent !important; }
[data-testid="stTab"][aria-selected="true"] {
    color: var(--ink) !important;
    background: transparent !important;
    border-bottom-color: var(--accent) !important;
    box-shadow: none !important;
}
[data-testid="stTabHighlight"],
[data-testid="stTabBorder"],
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"],
.react-aria-SelectionIndicator,
[role="tablist"]::after { display: none !important; }

/* 指标条（自定义 HTML） */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
    gap: 1px;
    background: var(--rule);
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
    margin: 4px 0 6px 0;
}
.metric { background: var(--paper2); padding: 13px 15px 14px 15px; }
.metric-l {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 6px;
}
.metric-v {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 1.14rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.22;
    letter-spacing: -0.01em;
}
.metric-v small {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--ink3);
    margin-left: 4px;
    letter-spacing: 0;
}
.metric-n { font-size: 0.69rem; color: var(--ink3); margin-top: 5px; line-height: 1.45; }
.metric.main .metric-v { font-size: 1.55rem; }
.metric.main .metric-v small { font-size: 0.82rem; }

/* 原生 st.metric（保留兼容） */
[data-testid="stMetric"] {
    background: var(--paper2);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 12px 15px !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.28rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase;
    color: var(--ink3) !important;
}
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; font-size: 0.72rem !important; }

/* 表格 */
table.spec {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
    margin: 2px 0 4px 0;
}
table.spec th {
    background: var(--paper3);
    text-align: left;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    /* 表头里会出现 Ncr (N/m) 这类单位，不能转大写 */
    text-transform: none;
    color: var(--ink3);
    padding: 8px 14px;
    border-bottom: 1px solid var(--rule);
    white-space: nowrap;
}
table.spec td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--rule);
    color: var(--ink2);
    vertical-align: top;
    line-height: 1.6;
}
table.spec tr:last-child td { border-bottom: none; }
table.spec td.k {
    color: var(--ink3);
    width: 24%;
    white-space: nowrap;
    font-size: 0.77rem;
    letter-spacing: 0.03em;
}
table.spec td.v {
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    color: var(--ink);
    font-weight: 500;
    font-size: 0.81rem;
}

[data-testid="stTable"] {
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
}
[data-testid="stTable"] table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
[data-testid="stTable"] thead th {
    background: var(--paper3) !important;
    color: var(--ink3) !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase;
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--rule) !important;
    text-align: left !important;
}
[data-testid="stTable"] tbody td {
    padding: 9px 12px !important;
    border-bottom: 1px solid var(--rule) !important;
    color: var(--ink2) !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stTable"] tbody tr:last-child td { border-bottom: none !important; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    overflow: hidden;
    box-shadow: none !important;
}

/* 折叠面板 */
[data-testid="stExpander"] {
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    background: var(--paper2) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    color: var(--ink2) !important;
    letter-spacing: 0.03em;
}
[data-testid="stExpander"] summary:hover { color: var(--ink) !important; }

/* 提示条：白底 + 左侧细色条 */
[data-testid="stAlert"] {
    border: 1px solid var(--rule) !important;
    border-left: 3px solid var(--ink3) !important;
    border-radius: 2px !important;
    background: var(--paper2) !important;
    box-shadow: none !important;
    padding: 11px 15px !important;
}
[data-testid="stAlert"] p { font-size: 0.82rem !important; line-height: 1.65; color: var(--ink2) !important; }
[data-testid="stAlert"] svg { display: none; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: var(--ok) !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { border-left-color: var(--warn) !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { border-left-color: var(--danger) !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { border-left-color: var(--accent) !important; }

/* 状态框：压成单行标记，不占整行宽度 */
[data-testid="stStatusWidget"], [data-testid="stStatus"] {
    display: inline-flex !important;
    width: auto !important;
    max-width: 100%;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    background: var(--paper2) !important;
    box-shadow: none !important;
    padding: 4px 10px !important;
    margin: 0 0 6px 0 !important;
}
[data-testid="stStatusWidget"] p,
[data-testid="stStatus"] p {
    font-size: 0.74rem !important;
    color: var(--ink3) !important;
    line-height: 1.5 !important;
}
[data-testid="stStatusWidget"] svg, [data-testid="stStatus"] svg {
    width: 13px !important;
    height: 13px !important;
}

/* 状态条外层容器不要撑满 */
[data-testid="stStatusWidgetContainer"],
.element-container:has(> [data-testid="stStatusWidget"]) { width: auto !important; }

/* 引用条 */
.note {
    background: var(--paper2);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    padding: 10px 14px;
    font-size: 0.83rem;
    color: var(--ink2);
    line-height: 1.65;
    margin: 4px 0 8px 0;
}

/* 铺层图例 */
.layup-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
    align-items: center;
    font-size: 0.71rem;
    color: var(--ink3);
    letter-spacing: 0.05em;
    margin: 2px 0 6px 0;
}
.layup-legend span {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 1px;
    margin-right: 5px;
    vertical-align: middle;
    border: 1px solid rgba(15, 23, 42, 0.12);
}

/* 说明文字 */
[data-testid="stCaptionContainer"] p {
    font-size: 0.75rem !important;
    color: var(--ink3) !important;
    line-height: 1.6 !important;
}
[data-testid="stCaptionContainer"] strong { color: var(--ink2) !important; font-weight: 600; }

/* 分隔线 */
hr, [data-testid="stDivider"] hr {
    border: none !important;
    border-top: 1px solid var(--rule) !important;
    margin: 18px 0 !important;
    background: transparent !important;
}

/* Plotly 融入卡片 */
[data-testid="stPlotlyChart"] { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stPlotlyChart"] .main-svg { background: transparent !important; }
[data-testid="stPlotlyChart"] .modebar {
    background: transparent !important;
    opacity: 0.4;
    transition: opacity 0.15s ease;
}

.stApp p, .stApp li { color: var(--ink2); }

/* 压过 Streamlit 自带字体与标题尺寸 */
.stMarkdown,
[data-testid="stMarkdownContainer"] {
    font-family: var(--sans) !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {
    font-family: var(--sans) !important;
    color: var(--ink) !important;
    letter-spacing: 0 !important;
}
.stMarkdown h2, [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    margin: 0 0 0.7rem 0 !important;
}
.stMarkdown h3, [data-testid="stMarkdownContainer"] h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin: 0 0 0.6rem 0 !important;
}
.stMarkdown h4, [data-testid="stMarkdownContainer"] h4 {
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: var(--ink2) !important;
    letter-spacing: 0.01em !important;
    margin: 1.05rem 0 0.5rem 0 !important;
}
.stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-family: var(--sans) !important;
}

/* 提示条内部容器不要自带底色，只保留左侧色条 */
[data-testid="stAlert"] div { background: transparent !important; }

/* 小写希腊字母在微标签里保持原样 */
.metric-l .lc { text-transform: none; letter-spacing: 0; }

/* 侧栏窄表：固定列宽、限宽，避免表头折行或数值列拉满整行 */
table.spec.side { table-layout: fixed; max-width: 380px; }
table.spec.side th:first-child,
table.spec.side td.k { width: 40%; }

/* 图表控件：动画按钮、滑块、模式栏 */
.updatemenu-button rect {
    fill: var(--paper2) !important;
    stroke: var(--rule) !important;
    stroke-width: 1 !important;
    rx: 2 !important;
}
.updatemenu-button:hover rect { fill: var(--paper3) !important; stroke: #cbd5e1 !important; }
.updatemenu-button text {
    fill: var(--ink2) !important;
    font-family: var(--sans) !important;
    font-size: 11px !important;
    letter-spacing: 0.04em;
}
.slider-container .slider-group .slider-label {
    font-family: var(--mono) !important;
    font-size: 10.5px !important;
    fill: var(--ink3) !important;
}
[data-testid="stPlotlyChart"]:hover .modebar { opacity: 1; }
[data-testid="stPlotlyChart"] .modebar-btn path { fill: var(--ink3) !important; }

/* 图片：细边框，不带圆角 */
[data-testid="stImage"] img {
    border: 1px solid var(--rule);
    border-radius: 2px;
}

/* 分步向导：步骤条与页面卡 */
.steps {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 16px 0 20px 0;
}
.steps .step {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 11px;
    border: 1px solid var(--rule);
    border-radius: 2px;
    background: var(--paper2);
    color: var(--ink3);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.steps .step b {
    font-family: var(--mono);
    font-size: 0.64rem;
    color: #94a3b8;
}
.steps .step.done { background: var(--paper3); color: var(--ink2); }
.steps .step.done b { color: var(--ink3); }
.steps .step.on { border-color: var(--ink); color: var(--ink); }
.steps .step.on b { color: var(--accent); }
.steps i { font-style: normal; font-family: var(--mono); color: var(--rule); }

.st-key-modecard1, .st-key-modecard2, .st-key-modecard3,
.st-key-modecard9, .st-key-paramcard {
    background: var(--paper2);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 16px 18px 14px 18px !important;
    margin-bottom: 14px;
}
.mode-desc {
    font-size: 0.82rem;
    color: var(--ink2);
    line-height: 1.62;
    margin: 2px 0 10px 0;
}
.mode-spec {
    font-family: var(--mono);
    font-size: 0.73rem;
    color: var(--ink3);
    letter-spacing: 0.04em;
    margin-bottom: 14px;
}

/* 滑块两端的上下限标签会随可用宽度忽隐忽现，导致行高跳动；
   范围已写在控件下方的说明里，这里直接隐藏，保持三行等高。 */
[data-testid="stSliderTickBar"],
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] { display: none; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

FONT = "Inter, 'Microsoft YaHei', 'Segoe UI', sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}
N_GRID = 41

# 载荷输入范围：(最小, 最大, 步长)；滑块与精确值框共用同一组取值
NX_RANGE = (0, 2000, 10)
SF_RANGE = (1.0, 2.0, 0.05)

# 离线 ABAQUS 线性摄动屈曲参考算例。
# 包内能核对的只有云图底部打印的一阶特征值（2.1118 / 1.9976 / 6.5138）与载荷工况；
# .cae / .odb / .inp 模型文件没有随包提供，材料、几何、边界条件与铺层无法在包内复核，
# 因此参考铺层按同名预设案例记录填写，属于「假定与 Ritz 解同一模型」，界面上必须注明。
ABAQUS_PRESETS = {
    (500, 500): {
        "label": "无人机机翼蒙皮",
        "eigen": 2.1118,
        "ncr": 2111.8,
        "n_plies": 12,
        "theta4": [-44.9, 44.9, 45.2, -45.2],
        "odb": "Job_1_Drone.odb",
        "image": "abaqus_drone.png",
    },
    (1000, 200): {
        "label": "新能源电池包上盖",
        "eigen": 1.9976,
        "ncr": 1997.6,
        "n_plies": 10,
        "theta4": [44.96, -45.28, -44.64, -90.0],
        "odb": "Job_2_Battery.odb",
        "image": "abaqus_battery.png",
    },
    (1500, 800): {
        "label": "航天器太阳能帆板",
        "eigen": 6.5138,
        "ncr": 6513.8,
        "n_plies": 16,
        "theta4": [-45.0, 45.0, 45.0, -45.0],
        "odb": "Job_3_Solar.odb",
        "image": "abaqus_solar.png",
    },
}

# 各算例对应的 ABAQUS 云图（位于 figures/ 目录下）
ABAQUS_IMAGES = {k: v["image"] for k, v in ABAQUS_PRESETS.items()}

ABAQUS_TOL_DEG = 1.0     # 铺层角度偏差在此范围内视为同一铺层

# 场景预设：选择后联动滑块；用户之后仍可手动微调。
SCENES = {
    "自定义任意工况": {
        "subtitle": "自定义任意工况 · 双向或单向受压",
        "preset": None,
    },
    "无人机机翼蒙皮": {
        "subtitle": "无人机机翼蒙皮 · 双向等压 · 偏减重",
        "preset": {"nx": 500, "ny": 500, "sf": 1.5},
    },
    "新能源电池包上盖": {
        "subtitle": "新能源电池包上盖 · 双向受压 · 屈曲与局部强度",
        "preset": {"nx": 1000, "ny": 200, "sf": 1.2},
    },
    "航天器太阳能帆板": {
        "subtitle": "航天器太阳能帆板 · 轻量化 · SF ≥ 2.0",
        "preset": {"nx": 1500, "ny": 800, "sf": 2.0},
    },
}

# 云端预设案例：键为 (Nx, Ny, SF)，值为事先算好的结果。
DEMO_PRESETS = {
    (500, 500, 1.5): {
        "label": "500/500 经典等双轴",
        "n_plies": 12,
        "theta4": [-44.9, 44.9, 45.2, -45.2],
        "ncr": 2128.6,
        "base_ncr": 1077.6,
        "thinner": {"n": 10, "ncr": 1230.2},
    },
    (1000, 200, 1.2): {
        "label": "1000/200 电池包上盖",
        "n_plies": 10,
        "theta4": [44.96, -45.28, -44.64, -90.0],
        "ncr": 2008.9,
        "base_ncr": 1213.5,
        "thinner": None,
    },
    (1500, 800, 2.0): {
        "label": "1500/800 太阳能帆板",
        "n_plies": 16,
        "theta4": [-45.0, 45.0, 45.0, -45.0],
        "ncr": 6591.7,
        "base_ncr": 3890.0,
        "thinner": {"n": 14, "ncr": 4412.5},
    },
}

# 隔离区：来源未核实的离线预设。
# (1040, 1040, 1.5) 与 (1070, 870, 1.5) 两条记录缺少可追溯的优化日志，
# 且共用同一组 theta4 = [-40.3, 49.8, -75.4, 51.0]，无法确认各自独立寻优得到，
# 因此原样保留在这里，但不再进入 DEMO_PRESETS 的运行路径。
# 恢复条件：补齐这两条工况的 GA 日志或离线计算结果，并确认铺层与工况一一对应。
UNVERIFIED_DEMO_PRESETS = {
    (1040, 1040, 1.5): {
        "label": "1040/1040 等双轴",
        "n_plies": 14,
        "theta4": [-40.3, 49.8, -75.4, 51.0],
        "ncr": 3144.9,
        "base_ncr": 1711.1,
        "thinner": {"n": 12, "ncr": 1977.9},
    },
    (1070, 870, 1.5): {
        "label": "1070/870 非等双轴",
        "n_plies": 14,
        "theta4": [-40.3, 49.8, -75.4, 51.0],
        "ncr": 3469.1,
        "base_ncr": 2012.0,
        "thinner": {"n": 12, "ncr": 2181.8},
    },
}

_FAST_LOCK = threading.Lock()


@contextlib.contextmanager
def _lower_accuracy():
    """快速模式：临时降低 GA 规模与精修级数，退出时恢复全局参数。"""
    with _FAST_LOCK:
        saved = (ad.GA_POP, ad.GA_GEN, ad._SOLVER_CHK)
        ad.GA_POP, ad.GA_GEN = 40, 30
        ad._SOLVER_CHK = BucklingSolver(A_PLATE, B_PLATE, 10)
        try:
            yield
        finally:
            ad.GA_POP, ad.GA_GEN, ad._SOLVER_CHK = saved


# 结果来源：页面、设计记录、导出报告共用同一套说法，避免各页各写一套
SOURCE_INFO = {
    "ga": ("完整优化计算",
           "NSGA-II 在对称铺层上搜索角度，再用 SLSQP + Nelder–Mead 精修"),
    "pure0": ("单向受压纯 0° 铺层",
              "单向受压时纯 0° 的 D11 最大，已是解析最优，未运行 GA"),
    "preset": ("离线预设算例",
               "读取事先算好的结果，本次没有运行 GA"),
    "estimate": ("纯 0° 简化估算",
                 "未运行 GA：层数按纯 0° 可行性确定，铺层角度没有优化"),
}


def source_of(res: dict) -> str:
    """取结果来源标识；缺字段时按估算处理。"""
    return res.get("source", "estimate")


def source_line(res: dict) -> str:
    """一行来源说明，供 caption 使用。"""
    label, detail = SOURCE_INFO[source_of(res)]
    return f"**计算来源：{label}** —— {detail}。"


# 校核结论：只看 Ncr / FI 与结果来源，不看场景名或预设标签
VERDICT_TEXT = {
    "pass": "校核通过",
    "buckling_fail": "屈曲承载不足",
    "strength_fail": "强度校核不通过",
    "both_fail": "屈曲与强度均不满足",
    "unverified": "未验证（估算结果）",
    "infeasible": "未找到可行解",
    "error": "计算失败",
}


def verdict_of(res: dict) -> dict:
    """由实际计算结果判定屈曲、强度与整体结论。

    演示模式下没有离线算例时给出的是纯 0° 估算：数值是按该 0° 铺层实算的，
    但没有做铺层优化，因此即使达标也判为 unverified，不能当设计结论。
    """
    status = res.get("status")
    if status in ("infeasible", "error"):
        return {"overall": status, "buckling": None, "strength": None,
                "ncr": None, "target": None, "fi": None}

    ncr, target = float(res["ncr"]), float(res["target"])
    fi = float(res["fi"])
    buckling_ok, strength_ok = ncr >= target, fi < 1.0
    if source_of(res) == "estimate":
        overall = "unverified"
    elif buckling_ok and strength_ok:
        overall = "pass"
    elif buckling_ok:
        overall = "strength_fail"
    elif strength_ok:
        overall = "buckling_fail"
    else:
        overall = "both_fail"
    return {"overall": overall, "buckling": buckling_ok, "strength": strength_ok,
            "ncr": ncr, "target": target, "fi": fi}


# 计算层
@st.cache_data(show_spinner=False, max_entries=24)
def auto_design(nx: int, ny: int, sf: float, fast: bool) -> dict:
    """复现 Auto_Design.main() 的设计流程，返回结构化结果。"""
    nx, ny, sf = float(nx), float(ny), float(sf)
    target = max(nx, ny) * sf
    biaxial = nx > 0 and ny > 0

    rows = ad.scan_pure_zero(nx, ny, target)
    feasible = [r for r in rows if r["ok"]]
    if not feasible:
        ncr_top = rows[-1]["ncr"]
        need = ad.N_MAX * (target / ncr_top) ** (1.0 / 3.0) if ncr_top > 0 else np.inf
        return {"status": "infeasible", "nx": nx, "ny": ny, "sf": sf,
                "target": target, "biaxial": biaxial, "rows": rows,
                "need_plies": float(need),
                "source": "ga" if biaxial else "pure0"}

    n_min = feasible[0]["N"]
    if biaxial:
        n_plies = n_min if n_min % 2 == 0 else n_min + 1
        base_ncr = float(next(r for r in rows if r["N"] == n_plies)["ncr"])
        if fast:
            with _lower_accuracy():
                theta, _score, ga_time = ad.optimize_layup(n_plies, nx, ny, target)
        else:
            theta, _score, ga_time = ad.optimize_layup(n_plies, nx, ny, target)
        ncr, lam = ad.ncr_equiv(theta, n_plies, nx, ny, ad._SOLVER_CHK)
        half = ad.half_stack(theta, n_plies)
        s16, s26 = ad.balance_residual(half)
        layup = ad.stacking_text(theta, n_plies)
        theta4 = [float(t) for t in theta]
        stack = ad.symmetric_stack(theta, n_plies)
        thinner = None
        for n in range(ad.N_MIN, n_plies, 2):
            v, _ = ad.ncr_equiv(theta, n, nx, ny, ad._SOLVER_CHK)
            if v >= target:
                thinner = {"n": n, "ncr": float(v)}
                break
    else:
        n_plies = n_min
        base_ncr = ncr = float(feasible[0]["ncr"])
        lam = float(feasible[0]["lam"])
        theta4 = [0.0, 0.0, 0.0, 0.0]
        stack = [0.0] * n_plies
        layup = f"[0]_{n_plies}"
        s16 = s26 = 0.0
        ga_time = 0.0
        thinner = None

    st_res = ad.strength_check(stack, nx, ny, sf)
    d_opt = ad.d_of_angles(stack)
    d_zero = ad.d_of_angles([0.0] * n_plies)

    return {
        "status": "ok", "nx": nx, "ny": ny, "sf": sf, "fast": fast,
        "source": "ga" if biaxial else "pure0",
        "target": target, "biaxial": biaxial,
        "n_min": n_min, "n_plies": n_plies,
        "base_ncr": base_ncr, "ncr": float(ncr), "lam": float(lam),
        "layup": layup, "theta4": theta4, "stack": stack,
        "s16": float(s16), "s26": float(s26),
        "fi": float(st_res["fi"]), "fi_ply": int(st_res["ply"]),
        "fi_theta": float(st_res["theta"]),
        "areal": float(ad.areal_weight(n_plies)),
        "thickness": float(n_plies * T_PLY),
        "d_opt": d_opt.tolist(), "d_zero": d_zero.tolist(),
        "ga_time": float(ga_time), "rows": rows, "thinner": thinner,
    }


@st.cache_data(show_spinner=False, max_entries=32)
def demo_design(nx: int, ny: int, sf: float) -> dict:
    """预设案例模式：只查预设表或做纯 0° 快速估算，不运行 GA。"""
    nx, ny, sf = float(nx), float(ny), float(sf)
    target = max(nx, ny) * sf
    biaxial = nx > 0 and ny > 0

    rows = ad.scan_pure_zero(nx, ny, target)
    feasible = [r for r in rows if r["ok"]]
    if not feasible:
        ncr_top = rows[-1]["ncr"]
        need = ad.N_MAX * (target / ncr_top) ** (1.0 / 3.0) if ncr_top > 0 else np.inf
        return {"status": "infeasible", "nx": nx, "ny": ny, "sf": sf,
                "target": target, "biaxial": biaxial, "rows": rows,
                "need_plies": float(need), "demo": True,
                "source": "estimate"}

    n_min = feasible[0]["N"]
    preset_key = (int(round(nx)), int(round(ny)), round(sf, 2))
    preset = DEMO_PRESETS.get(preset_key)
    isolated = UNVERIFIED_DEMO_PRESETS.get(preset_key)

    if preset is not None:
        n_plies = preset["n_plies"]
        theta4 = [float(t) for t in preset["theta4"]]
        ncr = float(preset["ncr"])
        base_ncr = float(preset["base_ncr"])
        thinner = preset["thinner"]
        source = "preset"
        label = preset.get("label", "预设案例")
    else:
        n_plies = n_min if (not biaxial or n_min % 2 == 0) else n_min + 1
        base_ncr = float(next(r for r in rows if r["N"] == n_plies)["ncr"])
        ncr = base_ncr
        theta4 = [0.0, 0.0, 0.0, 0.0]
        thinner = None
        source = "estimate"
        label = ("纯 0° 快速估算（未运行 GA；该工况命中的离线算例来源未核实，未采用）"
                 if isolated is not None
                 else "纯 0° 快速估算（未运行 GA）")

    if biaxial:
        stack = ad.symmetric_stack(theta4, n_plies)
        layup = ad.stacking_text(theta4, n_plies)
        s16, s26 = ad.balance_residual(ad.half_stack(theta4, n_plies))
    else:
        stack = [0.0] * n_plies
        layup = f"[0]_{n_plies}"
        theta4 = [0.0, 0.0, 0.0, 0.0]
        s16 = s26 = 0.0

    st_res = ad.strength_check(stack, nx, ny, sf)
    d_opt = ad.d_of_angles(stack)
    d_zero = ad.d_of_angles([0.0] * n_plies)
    lam = ncr / max(nx, ny)

    return {
        "status": "ok", "nx": nx, "ny": ny, "sf": sf, "fast": True,
        "demo": True, "source": source, "source_label": label,
        "isolated_preset": isolated is not None,
        "target": target, "biaxial": biaxial,
        "n_min": n_min, "n_plies": n_plies,
        "base_ncr": base_ncr, "ncr": ncr, "lam": float(lam),
        "layup": layup, "theta4": theta4, "stack": stack,
        "s16": float(s16), "s26": float(s26),
        "fi": float(st_res["fi"]), "fi_ply": int(st_res["ply"]),
        "fi_theta": float(st_res["theta"]),
        "areal": float(ad.areal_weight(n_plies)),
        "thickness": float(n_plies * T_PLY),
        "d_opt": d_opt.tolist(), "d_zero": d_zero.tolist(),
        "ga_time": 0.0, "rows": rows, "thinner": thinner,
    }


@st.cache_data(show_spinner=False, max_entries=16)
def buckling_mode(stack, nx: float, ny: float) -> dict:
    """Ritz 临界模态（3D 动画数据）。"""
    D = ad.d_of_angles(stack)
    solver = BucklingSolver(A_PLATE, B_PLATE, M_VERIFY)
    lam, c = solver.ncr_mode(D, nx, ny)

    x = np.linspace(0.0, A_PLATE, N_GRID)
    y = np.linspace(0.0, B_PLATE, N_GRID)
    m_idx = np.arange(1, M_VERIFY + 1)
    phi = np.sin(np.pi * np.outer(x, m_idx) / A_PLATE)
    psi = np.sin(np.pi * np.outer(y, m_idx) / B_PLATE)
    w = phi @ c @ psi.T
    w = w / np.max(np.abs(w))
    m_dom, n_dom = np.unravel_index(int(np.argmax(np.abs(c))), c.shape)

    X, Y = np.meshgrid(x, y)
    w11 = np.sin(np.pi * X / A_PLATE) * np.sin(np.pi * Y / B_PLATE)

    return {"lam": float(lam), "x": x, "y": y,
            "w_actual": w.T, "w11": w11,
            "m": int(m_dom) + 1, "n": int(n_dom) + 1}


# 绘图层
def _surface_trace(X, Y, Z):
    """统一风格的曲面 trace（动画帧复用，颜色标尺固定）。"""
    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale="RdBu_r", cmin=-1.0, cmax=1.0, showscale=False,
        lighting=dict(ambient=0.68, diffuse=0.85, specular=0.15, roughness=0.65),
        contours=dict(z=dict(show=True, usecolormap=True)),
        hovertemplate="x = %{x:.3f} m<br>y = %{y:.3f} m<br>"
                      "w / w_max = %{z:+.2f}<extra></extra>",
        name="w / w_max",
    )


def figure_mode_3d(x, y, Z) -> go.Figure:
    """屈曲波形 3D 曲面 + 幅值演化动画。"""
    X, Y = np.meshgrid(x, y)
    amps = np.linspace(0.06, 1.0, 14)
    frames = [go.Frame(data=[_surface_trace(X, Y, Z * a)], name=f"{a:.3f}")
              for a in amps]
    fig = go.Figure(data=[_surface_trace(X, Y, Z * amps[-1])], frames=frames)

    play = dict(label="播放", method="animate",
                args=[None, {"frame": {"duration": 90, "redraw": True},
                             "fromcurrent": False,
                             "transition": {"duration": 0}}])
    pause = dict(label="暂停", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}}])
    steps = [dict(method="animate", label=f"{a:.0%}",
                  args=[[f.name], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}])
             for a, f in zip(amps, frames)]

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=44),
        paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        updatemenus=[dict(type="buttons", direction="right", showactive=False,
                          x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(l=0, r=4, t=0, b=0),
                          bgcolor="#ffffff", bordercolor="#cbd5e1",
                          font=dict(size=11, color="#334155"),
                          buttons=[play, pause])],
        sliders=[dict(active=len(amps) - 1, x=0.02, y=-0.02, len=0.96,
                      currentvalue=dict(prefix="变形幅值 ", font=dict(size=11)),
                      font=dict(size=10),
                      bgcolor="#f8fafc", bordercolor="#e2e8f0",
                      tickcolor="#94a3b8", steps=steps)],
        scene=dict(
            xaxis=dict(title="x (m)", range=[0.0, A_PLATE],
                       showbackground=True, backgroundcolor="#f8fafc",
                       gridcolor="#e2e8f0", zeroline=False),
            yaxis=dict(title="y (m)", range=[0.0, B_PLATE],
                       showbackground=True, backgroundcolor="#f8fafc",
                       gridcolor="#e2e8f0", zeroline=False),
            zaxis=dict(title="w / w_max", range=[-1.2, 1.2],
                       showbackground=True, backgroundcolor="#f8fafc",
                       gridcolor="#e2e8f0", zeroline=False),
            aspectmode="manual", aspectratio=dict(x=1.05, y=1.05, z=0.5),
            camera=dict(eye=dict(x=1.24, y=-1.44, z=0.92)),
        ),
    )
    return fig


def figure_compare(base_ncr: float, ncr: float, target: float) -> go.Figure:
    """柱状图：优化推荐 vs 纯 0° 同层数铺层。"""
    values = [base_ncr, ncr]
    fig = go.Figure(go.Bar(
        x=["纯 0° 铺层", "优化推荐"], y=values, width=0.55,
        marker=dict(color=["#94a3b8", "#2563eb"],
                    line=dict(color="#0f172a", width=1)),
        text=[f"{base_ncr:,.0f}", f"{ncr:,.0f}"],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13, color="#0f172a", family="Consolas, monospace"),
        hovertemplate="%{x}<br>Ncr = %{y:,.0f} N/m<extra></extra>",
    ))
    fig.add_hline(y=target, line=dict(color="#ef4444", dash="dash", width=1.5))
    fig.add_annotation(
        x=0.015, y=target, xref="paper", yref="y", showarrow=False,
        text=f"目标 Ncr = {target:,.0f} N/m", xanchor="left", yanchor="bottom",
        font=dict(color="#ef4444", size=11),
        bgcolor="rgba(255,255,255,0.88)", borderpad=3,
    )
    fig.update_layout(
        height=372,
        margin=dict(l=84, r=14, t=40, b=10),
        paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        title=dict(text="同层数下的屈曲承载能力", x=0.0, y=0.97,
                   xanchor="left", yanchor="top",
                   font=dict(size=12, color="#64748b")),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title=dict(text="临界屈曲载荷 Ncr (N/m)", standoff=10,
                              font=dict(size=11, color="#64748b")),
                   tickfont=dict(size=11),
                   range=[0, max(values + [target]) * 1.20]),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    return fig


LAYUP_COLORS = {
    "0": "#2563eb",
    "90": "#dc2626",
    "+45": "#0d9488",
    "-45": "#f59e0b",
    "other": "#94a3b8",
}


def _ply_color(angle: float) -> str:
    """±1° 内按 0/90/±45 着色，其余灰色。"""
    a = float(angle)
    if abs(a) <= 1.0:
        return LAYUP_COLORS["0"]
    if abs(abs(a) - 90.0) <= 1.0:
        return LAYUP_COLORS["90"]
    if abs(a - 45.0) <= 1.0:
        return LAYUP_COLORS["+45"]
    if abs(a + 45.0) <= 1.0:
        return LAYUP_COLORS["-45"]
    return LAYUP_COLORS["other"]


def figure_layup(stack) -> go.Figure:
    """水平堆叠条形图：每一层一个色块，悬停显示层号与角度。"""
    fig = go.Figure()
    for i, angle in enumerate(stack, start=1):
        fig.add_trace(go.Bar(
            y=["铺层"], x=[1], orientation="h",
            marker=dict(color=_ply_color(angle),
                        line=dict(color="rgba(255,255,255,0.9)", width=1)),
            hovertemplate=f"第 {i} 层：{float(angle):g}°<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        barmode="stack",
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        bargap=0.0,
    )
    return fig


# 界面小组件
def pretty_layup(layup: str) -> str:
    """去掉无意义的小数尾零（-45.0 → -45）。"""
    return re.sub(r"\.0(?=[/\]])", "", layup)


_SUB_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def subscript(num) -> str:
    """数字转下标字符，用于 [0]₈s 一类标准层合板记号。"""
    return str(num).translate(_SUB_DIGITS)


def layup_card(layup: str) -> None:
    """推荐铺层序列（等宽字体，标准层合板记号）。"""
    shown = pretty_layup(layup)
    size = 21 if len(shown) <= 34 else 17 if len(shown) <= 44 else 14
    shown = shown.replace("/", "/<wbr>")
    st.markdown(
        f"""
        <div style="background:#f1f5f9;border:1px solid #e2e8f0;
                    border-left:3px solid #2563eb;padding:12px 16px 13px 16px;
                    margin-bottom:12px;">
          <div style="font-size:10px;letter-spacing:1.6px;color:#64748b;
                      font-weight:700;text-transform:uppercase;
                      margin-bottom:5px;">推荐铺层序列</div>
          <div style="font-family:'JetBrains Mono',Consolas,monospace;
                      font-size:{size}px;font-weight:600;color:#0f172a;
                      line-height:1.45;word-break:normal;
                      overflow-wrap:normal;">{shown}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scan_table(rows, target: float) -> pd.DataFrame:
    """纯 0° 层数扫描表。"""
    return pd.DataFrame({
        "N": [r["N"] for r in rows],
        "t (mm)": [round(r["t"], 3) for r in rows],
        "Ncr (N/m)": [round(r["ncr"], 1) for r in rows],
        "λ": [round(r["lam"], 2) for r in rows],
        "Ncr / Ncr_target": [round(r["ncr"] / target, 2) for r in rows],
        "判定": ["可行" if r["ok"] else "不足" for r in rows],
    })


def verdict_block(res: dict) -> None:
    """屈曲、强度分别给结论，再给整体结论；全部由计算结果决定。"""
    v = verdict_of(res)
    overall = v["overall"]
    if overall in ("infeasible", "error"):
        return

    ncr, target, fi = v["ncr"], v["target"], v["fi"]
    buckle_mark = "通过" if v["buckling"] else "不足"
    strength_mark = "通过" if v["strength"] else "不通过"
    st.markdown(
        f"""
        <table class="spec">
          <tr><th style="width:18%">校核项</th><th>判据与实算值</th>
              <th style="width:12%;text-align:right">结论</th></tr>
          <tr><td class="k">屈曲承载</td>
              <td class="v">Ncr = {ncr:,.0f} N/m
                  {'≥' if v['buckling'] else '&lt;'} Ncr_target =
                  {target:,.0f} N/m</td>
              <td class="v" style="text-align:right">{buckle_mark}</td></tr>
          <tr><td class="k">面内强度</td>
              <td class="v">Tsai-Hill FI = {fi:.2e}
                  {'&lt;' if v['strength'] else '≥'} 1</td>
              <td class="v" style="text-align:right">{strength_mark}</td></tr>
          <tr><td class="k">整体结论</td>
              <td class="v">{VERDICT_TEXT[overall]}</td>
              <td class="v" style="text-align:right">—</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    if overall == "pass":
        st.success(
            f"整体校核通过：Ncr = {ncr:,.0f} N/m ≥ 目标 {target:,.0f} N/m，"
            f"Tsai-Hill FI = {fi:.2e} < 1。"
        )
    elif overall == "unverified":
        st.warning(
            f"未验证：这是演示模式下的纯 0° 估算（Ncr = {ncr:,.0f} N/m，"
            f"FI = {fi:.2e}，按该 0° 铺层实算），铺层角度没有做优化，"
            "不能作为设计结论。要得到可用的校核结论，请在本地以 "
            "APP_MODE=local 运行，或选择有离线算例的载荷工况。"
        )
    elif overall == "buckling_fail":
        st.warning(
            f"整体未通过：屈曲承载不足（Ncr = {ncr:,.0f} N/m < 目标 "
            f"{target:,.0f} N/m），需增加层数或重新优化铺层角度。"
        )
    elif overall == "strength_fail":
        st.error(
            f"整体未通过：屈曲满足（Ncr = {ncr:,.0f} N/m ≥ 目标 "
            f"{target:,.0f} N/m），但 Tsai-Hill FI = {fi:.2e} ≥ 1"
            + (f"（最危险层为第 {res['fi_ply']} 层，θ = {res['fi_theta']:.1f}°）"
               if res.get("fi_ply") else "")
            + "，需增加厚度。"
        )
    else:
        st.error(
            f"整体未通过：屈曲（Ncr = {ncr:,.0f} N/m，目标 {target:,.0f} N/m）"
            f"与强度（FI = {fi:.2e}）均不满足，需同时增加层数与厚度。"
        )

    st.caption(
        f"安全系数 SF = {res['sf']:.2f} 已计入 Ncr_target"
        f"（λ = {res['lam']:.2f} {'≥' if res['lam'] >= res['sf'] else '<'} SF）。"
    )


def _figure_path(name: str):
    """按包内 / 根目录 / 当前目录布局查找 figures 下的图片，找不到返回 None。"""
    here = Path(__file__).resolve().parent
    for p in (here / "figures" / name,
              here.parent / "figures" / name,
              here.parent.parent / "figures" / name,
              Path.cwd() / "figures" / name,
              Path.cwd() / name):
        if p.exists():
            return p
    return None


def _same_stack(stack, n_plies: int, theta4) -> bool:
    """当前铺层是否与参考算例的铺层一致（角度容差 ABAQUS_TOL_DEG）。"""
    cur = [float(a) for a in stack]
    if len(cur) != int(n_plies):
        return False
    ref = [float(a) for a in ad.symmetric_stack([float(t) for t in theta4],
                                                int(n_plies))]
    for a, b in zip(cur, ref):
        d = abs(a - b) % 360.0
        if min(d, 360.0 - d) > ABAQUS_TOL_DEG:
            return False
    return True


# 离线参考算例的可比状态。证据只能支撑到这里：
# 铺层一致 ≠ 模型等价，材料/几何/网格/边界/加载方式没有随包模型文件可核对，
# 所以"铺层一致"也只是条件性参考，不构成验证结论。
ABA_STATE = {
    "no_case": "该工况没有离线参考算例",
    "stack_mismatch": "铺层不一致，不可直接比较",
    "conditional": "条件性参考对比：铺层一致，模型等价性未验证",
}


def aba_compare(res: dict, nx: float, ny: float) -> dict:
    """判断当前结果与离线 ABAQUS 参考算例的可比状态。

    返回 state ∈ {no_case, stack_mismatch, conditional}、参考记录、参考铺层，
    以及（仅 conditional 时）与参考值的数值差异；后两项之外一律给 None。
    """
    ref = ABAQUS_PRESETS.get((int(round(nx)), int(round(ny))))
    if ref is None:
        return {"state": "no_case", "ref": None, "ref_layup": None,
                "delta_pct": None, "note": ABA_STATE["no_case"]}

    ref_layup = ad.stacking_text([float(t) for t in ref["theta4"]],
                                 int(ref["n_plies"]))
    if not _same_stack(res.get("stack", []), ref["n_plies"], ref["theta4"]):
        return {"state": "stack_mismatch", "ref": ref, "ref_layup": ref_layup,
                "delta_pct": None, "note": ABA_STATE["stack_mismatch"]}

    ref_ncr = float(ref["ncr"])
    delta = None
    if ref_ncr > 0:
        delta = (float(res["ncr"]) - ref_ncr) / ref_ncr * 100.0
    return {"state": "conditional", "ref": ref, "ref_layup": ref_layup,
            "delta_pct": delta, "note": ABA_STATE["conditional"]}


# 分步页面
_STEPS = (
    ("mode", "01", "选择模式"),
    ("params", "02", "参数设置"),
    ("result", "03", "计算结果"),
)


def step_bar(current: str) -> None:
    """顶部步骤指示条；已完成的步骤可用下方按钮回退。"""
    order = [s[0] for s in _STEPS]
    idx = order.index(current) if current in order else 0
    parts = []
    for i, (_, num, label) in enumerate(_STEPS):
        cls = "step on" if i == idx else ("step done" if i < idx else "step")
        parts.append(f'<span class="{cls}"><b>{num}</b>{label}</span>')
    st.markdown(
        '<div class="steps">' + '<i>/</i>'.join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _goto(step: str) -> None:
    """切换步骤并重跑脚本。"""
    st.session_state["step"] = step
    st.rerun()


def _restart() -> None:
    """清空当前任务，回到第 1 步开始新的设计。"""
    st.session_state["design_job"] = None
    st.session_state["calc_mode"] = None
    st.session_state["scene"] = None
    st.session_state.pop("_res_cache", None)
    _goto("mode")


def _record_design(res: dict, job: dict) -> None:
    """把本次设计追加到会话内的设计记录（会话结束即消失）。"""
    ok = res.get("status") != "infeasible"
    stack = res.get("stack") or []
    scene = st.session_state.get("scene")
    hist = st.session_state.setdefault("history", [])
    if len(hist) >= 20:
        del hist[0]
    hist.append({
        "时间": time.strftime("%H:%M:%S"),
        "工况": scene or "自定义工况",
        "来源": SOURCE_INFO[source_of(res)][0],
        "Nx (N/m)": int(job["nx"]),
        "Ny (N/m)": int(job["ny"]),
        "SF": round(float(job["sf"]), 2),
        "目标 Ncr (N/m)": round(float(res["target"]), 1),
        "推荐铺层": res.get("layup") if ok else None,
        "层数": len(stack) if ok else None,
        "总厚 (mm)": round(len(stack) * T_PLY, 3) if ok else None,
        "Ncr (N/m)": round(float(res["ncr"]), 1) if ok else None,
        "λ": round(float(res["lam"]), 2) if ok else None,
        "结论": VERDICT_TEXT[verdict_of(res)["overall"]],
    })


def _render_history() -> None:
    """第 1 步底部的设计记录表。"""
    st.markdown(
        '<div class="card-head"><span class="card-idx">B</span>'
        '<span class="card-title">设计记录</span>'
        '<span class="card-aside">关闭页面即清空</span></div>',
        unsafe_allow_html=True,
    )
    hist = st.session_state.get("history", [])
    if not hist:
        st.caption("本次会话还没有设计记录；每完成一次计算会自动追加一条。")
        return
    st.dataframe(
        pd.DataFrame(hist), hide_index=True, width="stretch",
        column_config={
            "SF": st.column_config.NumberColumn("SF", format="%.2f"),
            "目标 Ncr (N/m)": st.column_config.NumberColumn(
                "目标 Ncr (N/m)", format="%.1f"),
            "总厚 (mm)": st.column_config.NumberColumn("总厚 (mm)", format="%.3f"),
            "Ncr (N/m)": st.column_config.NumberColumn("Ncr (N/m)", format="%.1f"),
            "λ": st.column_config.NumberColumn("λ", format="%.2f"),
        },
    )
    st.caption(
        f"本次会话共 {len(hist)} 条记录（最多保留 20 条）；"
        "记录只保存在当前浏览器会话中，关闭页面后自动清除。"
    )
    if st.button("清空记录", key="clear_history"):
        st.session_state["history"] = []
        st.rerun()


def _design_conditions() -> None:
    """板尺寸、材料与求解器参数（只读说明，三个页面共用）。"""
    st.markdown(
        '<div class="card-head"><span class="card-idx">A</span>'
        '<span class="card-title">设计条件</span>'
        '<span class="card-aside">SS1 四边简支</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <table class="spec">
          <tr><th style="width:34%">项目</th><th>取值</th></tr>
          <tr><td class="k">板尺寸</td><td class="v">a = b = {A_PLATE:.2f} m</td></tr>
          <tr><td class="k">铺层厚度</td><td class="v">t_ply = {T_PLY} mm</td></tr>
          <tr><td class="k">材料体系</td><td class="v">T300/5208 碳纤维/环氧</td></tr>
          <tr><td class="k">面密度</td><td class="v">ρ = {DENSITY:.0f} kg/m³</td></tr>
          <tr><td class="k">层数扫描</td><td class="v">N = {ad.N_MIN}…{ad.N_MAX}，对称铺层</td></tr>
          <tr><td class="k">屈曲求解</td><td class="v">CLT + Rayleigh-Ritz，M = 16</td></tr>
          <tr><td class="k">优化算法</td><td class="v">NSGA-II · 80 × 60（快速 40 × 30）</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <table class="spec side">
          <tr><th>材料常数</th><th style="text-align:right">取值</th></tr>
          <tr><td class="k">E1</td><td class="v" style="text-align:right">{E1:g} GPa</td></tr>
          <tr><td class="k">E2</td><td class="v" style="text-align:right">{E2:g} GPa</td></tr>
          <tr><td class="k">ν12</td><td class="v" style="text-align:right">{V12:g}</td></tr>
          <tr><td class="k">G12</td><td class="v" style="text-align:right">{G12:g} GPa</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.caption("*材料常数与 angle_optimize.py 中的求解器参数一致。*")


# 载荷输入：滑块与精确值框共用同一对数，改任意一个另一个跟着走
def _set_load(nx: int, ny: int, sf: float) -> None:
    """同时写入滑块与精确值框，避免两个控件各自留旧值。"""
    for key, val in (("nx", nx), ("ny", ny), ("sf", sf)):
        st.session_state[f"{key}_slider"] = val
        st.session_state[f"{key}_num"] = val


def _sync_from_slider(slider_key: str, num_key: str) -> None:
    st.session_state[num_key] = st.session_state[slider_key]


def _sync_from_number(slider_key: str, num_key: str) -> None:
    st.session_state[slider_key] = st.session_state[num_key]


def _load_input(label: str, bounds: tuple, key: str,
                fmt: str = "%d") -> float:
    """一行「滑块 + 精确值框」；两个控件通过 on_change 双向同步。"""
    lo, hi, step = bounds
    c_sl, c_num = st.columns([2.4, 1], gap="small",
                             vertical_alignment="bottom")
    value = c_sl.slider(label, lo, hi, step=step, format=fmt,
                        key=f"{key}_slider", on_change=_sync_from_slider,
                        args=(f"{key}_slider", f"{key}_num"))
    c_num.number_input(f"{label} 精确值", lo, hi, step=step, format=fmt,
                       key=f"{key}_num", label_visibility="collapsed",
                       on_change=_sync_from_number,
                       args=(f"{key}_slider", f"{key}_num"))
    return value


def _target_metric(preview: float) -> None:
    """参数页的目标载荷预览（随输入实时刷新）。"""
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric">
            <div class="metric-l">目标临界载荷 Ncr_target</div>
            <div class="metric-v">{preview:,.1f}<small>N/m</small></div>
            <div class="metric-n">= max(Nx, Ny) · SF，随上方滑块实时更新</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _spec_table(headers: list, rows: list) -> None:
    """渲染 table.spec：首列作项目名，其余列作取值。"""
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        f'<tr><td class="k">{escape(str(row[0]))}</td>'
        + "".join(f'<td class="v">{escape(str(cell))}</td>' for cell in row[1:])
        + "</tr>"
        for row in rows
    )
    st.markdown(f'<table class="spec"><tr>{head}</tr>{body}</table>',
                unsafe_allow_html=True)


def _mode_switch(details: list | None = None) -> None:
    """演示模式下提供显式升级开关；local 模式只显示状态行。

    传 details 时把口径文字交给调用方统一折叠，避免一屏堆多个展开框。
    """
    if IS_DEMO:
        st.checkbox(
            "在本机运行完整优化（NSGA-II 寻优）", key="run_local",
            help="演示模式默认只读取离线算例、不跑 GA；勾选后本会话改为调用 "
                 "Auto_Design.py 真正寻优（快速约 5 s，标准约 1 分钟）。",
        )
    st.caption(
        "当前执行路径："
        + ("演示（只读离线算例）" if _demo_now() else "本机完整计算")
    )
    text = (
        ("只读取离线预设算例，不运行 GA；其他工况退回纯 0° 估算。"
         if _demo_now()
         else "调用 NSGA-II 真正寻优，最终 Ncr 按 M = 16 复核。")
        + f"环境变量 APP_MODE = {APP_MODE}。"
    )
    if details is None:
        with st.expander("查看执行口径"):
            st.caption(text)
    else:
        details.append(text)


def _mode_note(nx_in: int, ny_in: int, sf_in: float,
               details: list | None = None) -> bool:
    """按当前执行路径给出说明；返回本次是否走快速 GA（演示分支恒为 True）。"""
    if not _demo_now():
        fast_in = st.checkbox(
            "快速模式", value=True, key="fast_chk",
            help="GA 40×30 + 精修级数 M=10，约 5 s；取消勾选则 80×60，约 1 分钟。"
                 "最终 Ncr 仍按 M=16 复核。",
        )
        st.caption(
            "NSGA-II 寻优规模："
            + ("快速 40 × 30" if fast_in else "标准 80 × 60")
        )
        return fast_in

    presets_txt = "、".join(
        f"{int(a)}/{int(b)}（SF {float(c):.2f}）"
        for a, b, c in sorted(DEMO_PRESETS)
    )
    st.caption(f"本模式可用的离线算例 {len(DEMO_PRESETS)} 组；其他组合退回纯 0° 估算。")
    if details is None:
        with st.expander("查看本模式可用的离线算例"):
            st.caption(presets_txt)
    else:
        details.append(f"可用离线算例：{presets_txt}。")

    key = (int(nx_in), int(ny_in), round(float(sf_in), 2))
    isolated = UNVERIFIED_DEMO_PRESETS.get(key)
    if isolated is not None:
        st.caption(f"该工况命中的离线算例「{isolated['label']}」来源未核实，已改用纯 0° 估算。")
    if key not in DEMO_PRESETS:
        st.caption("当前这组载荷没有离线算例；直接运行只会得到纯 0° 估算结果。")
    return True


def page_mode() -> None:
    """第 1 步：选择计算模式。"""
    st.markdown(
        '<div class="sec-label"><b>01</b>选择模式</div>'
        '<div class="sec-title">选择计算模式</div>',
        unsafe_allow_html=True,
    )
    st.caption("选一个场景进下一步；算完觉得不行，返回改载荷重算就行。")

    presets = [(k, v) for k, v in SCENES.items() if v["preset"] is not None]
    left, right = st.columns(2, gap="medium")
    for i, (name, cfg) in enumerate(presets):
        p = cfg["preset"]
        desc = (cfg["subtitle"].partition(" · ")[2] or cfg["subtitle"]).strip()
        with (left if i % 2 == 0 else right):
            with st.container(key=f"modecard{i + 1}"):
                st.markdown(
                    f'<div class="card-head"><span class="card-idx">M{i + 1}</span>'
                    f'<span class="card-title">{name}</span></div>'
                    f'<div class="mode-desc">{desc}</div>'
                    f'<div class="mode-spec">Nx {p["nx"]:.0f} · Ny {p["ny"]:.0f} N/m'
                    f' · SF {p["sf"]:.2f}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("选择此预设", key=f"pick_preset_{i}", width="stretch"):
                    st.session_state["calc_mode"] = "preset"
                    st.session_state["scene"] = name
                    _set_load(int(p["nx"]), int(p["ny"]), float(p["sf"]))
                    _goto("params")

    with (right if len(presets) % 2 else left, st.container(key="modecard9")):
        st.markdown(
            '<div class="card-head"><span class="card-idx">M0</span>'
            '<span class="card-title">自定义工况</span></div>'
            '<div class="mode-desc">不套用预设，自行输入 Nx / Ny / SF；'
            '层数与铺层角度由程序寻优。</div>'
            '<div class="mode-spec">Nx / Ny / SF 可任意设定</div>',
            unsafe_allow_html=True,
        )
        if st.button("自定义输入", key="pick_custom", width="stretch"):
            st.session_state["calc_mode"] = "custom"
            st.session_state["scene"] = None
            _set_load(500, 500, 1.5)
            _goto("params")

    st.divider()
    _mode_switch()
    st.divider()
    _design_conditions()
    st.divider()
    _render_history()


def page_params() -> None:
    """第 2 步：设定载荷工况。"""
    scene = st.session_state.get("scene")
    st.markdown(
        '<div class="sec-label"><b>02</b>参数设置</div>'
        '<div class="sec-title">载荷工况</div>',
        unsafe_allow_html=True,
    )
    if scene:
        st.caption(SCENES[scene]["subtitle"])
    else:
        st.caption("自定义工况：自行设定双向载荷与安全系数，层数与铺层由程序寻优。")

    left, right = st.columns([1.1, 1], gap="large")

    with left, st.container(key="paramcard"):
        st.markdown(
            '<div class="card-head"><span class="card-idx">B</span>'
            '<span class="card-title">载荷与安全系数</span>'
            '<span class="card-aside">面内线载荷 N/m</span></div>',
            unsafe_allow_html=True,
        )
        nx_in = _load_input("纵向载荷 Nx (N/m)", NX_RANGE, "nx")
        ny_in = _load_input("横向载荷 Ny (N/m)", NX_RANGE, "ny")
        sf_in = _load_input("安全系数 SF", SF_RANGE, "sf", fmt="%.2f")

        st.caption("支持滑块快捷调节与精确值输入（两者双向同步）。")

        _details: list[str] = [
            f"Nx / Ny {NX_RANGE[0]}–{NX_RANGE[1]} N/m（步长 {NX_RANGE[2]}），"
            f"SF {SF_RANGE[0]:.2f}–{SF_RANGE[1]:.2f}（步长 {SF_RANGE[2]:.2f}）。"
            "右侧精确值框可直接键入，与滑块双向同步。",
        ]
        _mode_switch(_details)
        fast_in = _mode_note(nx_in, ny_in, sf_in, _details)
        _target_metric(max(nx_in, ny_in) * sf_in)
        with st.expander("查看口径说明"):
            for _line in _details:
                st.caption(_line)
        _job = st.session_state.get("design_job")
        if _job and (_job.get("nx"), _job.get("ny"),
                     round(float(_job.get("sf", 0)), 2), bool(_job.get("fast"))
                     ) != (int(nx_in), int(ny_in), round(float(sf_in), 2),
                           bool(fast_in)):
            st.caption("参数已改动：需重新点「运行计算」才更新结果，"
                       "结果页上仍是上一次的工况。")

        if st.button("运行计算", type="primary", width="stretch", key="run_btn"):
            if nx_in == 0 and ny_in == 0:
                st.error("Nx 与 Ny 不能同时为 0，请至少施加一个方向的载荷。")
            else:
                st.session_state["design_job"] = {
                    "nx": int(nx_in), "ny": int(ny_in),
                    "sf": float(sf_in), "fast": bool(fast_in),
                }
                _goto("result")

    with right:
        _design_conditions()

    if st.button("← 返回选择模式", key="back_mode"):
        _goto("mode")


# PDF 报告
# 优先使用随项目部署的字体，其次才是系统字体；均为 TrueType（.ttf）。
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simsun.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def _cjk_font_path():
    """查找可用的中文字体路径，找不到返回 None（PDF 会退回内置字体）。"""
    here = Path(__file__).resolve().parent
    for p in (here / "fonts" / "simhei.ttf",
              here.parent / "fonts" / "simhei.ttf",
              here / "simhei.ttf"):
        if p.exists():
            return p
    for name in FONT_CANDIDATES:
        p = Path(name)
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False, max_entries=8)
def build_report_pdf(res: dict):
    """把当前设计结果渲染成单页 PDF 报告，返回 bytes；无中文字体时返回 None。"""
    if not HAS_FPDF:
        return None
    font = _cjk_font_path()
    if font is None:          # 内置 Helvetica 无中文字形，宁可不出报告也不出乱码
        return None

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("复合材料层合板屈曲铺层优化 - 设计方案报告")
    pdf.add_page()
    pdf.add_font("CJK", "", str(font))
    fam = "CJK"

    # 标题
    pdf.set_font(fam, size=18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 12, "复合材料层合板屈曲铺层优化",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font(fam, size=11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, "设计方案报告",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    # 关键参数
    ncr = float(res["ncr"])
    target = float(res["target"])
    rows = [
        ("当前工况", f"Nx = {res['nx']:.0f} N/m，Ny = {res['ny']:.0f} N/m，"
                     f"SF = {res['sf']:.2f}"),
        ("运行模式", f"APP_MODE = {mode_text()}"),
        ("计算来源", SOURCE_INFO[source_of(res)][0]
                     + "（" + SOURCE_INFO[source_of(res)][1] + "）"),
        ("目标临界载荷", f"{target:,.1f} N/m"),
        ("推荐铺层", str(res["layup"])),
        ("层数 / 总厚度", f"N = {res['n_plies']} 层，t = {res['thickness']:.2f} mm"),
        ("面密度", f"{res['areal']:.2f} kg/㎡"),
        ("临界载荷 Ncr", f"{ncr:,.1f} N/m"),
        ("临界倍数 λ", f"{res['lam']:.2f}（要求 ≥ {res['sf']:.2f}）"),
        ("强度校核 FI", f"{res['fi']:.2e}（Tsai-Hill，< 1 为安全）"),
        ("校核结论", VERDICT_TEXT[verdict_of(res)["overall"]]),
    ]
    for label, value in rows:
        pdf.set_font(fam, size=11)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(36, 9, " " + label, fill=True)
        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(0, 9, " " + value, align="L",
                       new_x="LMARGIN", new_y="NEXT")

    # 结论：与页面用同一套判定
    pdf.ln(5)
    pdf.set_font(fam, size=12)
    overall = verdict_of(res)["overall"]
    if overall == "pass":
        pdf.set_text_color(22, 163, 74)
        verdict = (f"整体校核通过：Ncr = {ncr:,.1f} N/m ≥ 目标 {target:,.1f} N/m"
                   f"（等效安全系数 λ = {res['lam']:.2f} ≥ {res['sf']:.2f}），"
                   f"Tsai-Hill FI = {res['fi']:.2e} < 1。")
    elif overall == "unverified":
        pdf.set_text_color(217, 119, 6)
        verdict = ("未验证：本报告对应演示模式下的纯 0° 估算，铺层角度没有做优化，"
                   "数值只代表该 0° 铺层，不能作为设计结论。")
    elif overall == "buckling_fail":
        pdf.set_text_color(220, 38, 38)
        verdict = (f"整体未通过：屈曲承载不足（Ncr = {ncr:,.1f} N/m < 目标 "
                   f"{target:,.1f} N/m），需增加层数或重新优化铺层角度。")
    elif overall == "strength_fail":
        pdf.set_text_color(220, 38, 38)
        verdict = (f"整体未通过：屈曲满足（Ncr = {ncr:,.1f} N/m ≥ 目标 "
                   f"{target:,.1f} N/m），但 Tsai-Hill FI = {res['fi']:.2e} ≥ 1，"
                   "需增加厚度。")
    else:
        pdf.set_text_color(220, 38, 38)
        verdict = (f"整体未通过：屈曲（Ncr = {ncr:,.1f} N/m，目标 {target:,.1f} N/m）"
                   f"与强度（FI = {res['fi']:.2e}）均不满足。")
    pdf.multi_cell(0, 8, "结论：" + verdict, align="L",
                   new_x="LMARGIN", new_y="NEXT")

    if res["fi"] >= 1.0:
        pdf.set_text_color(220, 38, 38)
        pdf.multi_cell(0, 8, f"面内强度：Tsai-Hill FI = {res['fi']:.2e} ≥ 1，不通过。",
                       align="L", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 8, f"面内强度：Tsai-Hill FI = {res['fi']:.2e} < 1，通过。",
                       align="L", new_x="LMARGIN", new_y="NEXT")

    # 页脚
    pdf.ln(4)
    pdf.set_font(fam, size=9)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(0, 6,
                   "本报告依据 CLT + Rayleigh-Ritz 计算结果自动生成，"
                   "仅用于方案比选与设计参考。",
                   align="L", new_x="LMARGIN", new_y="NEXT")

    buf = io.BytesIO()
    buf.write(bytes(pdf.output()))
    return buf.getvalue()


# 页面路由
for _key, _default in (
    ("step", "mode"),
    ("calc_mode", None),
    ("scene", None),
    ("nx_slider", 500),
    ("ny_slider", 500),
    ("sf_slider", 1.5),
    ("nx_num", 500),
    ("ny_num", 500),
    ("sf_num", 1.5),
    ("design_job", None),
    ("_res_cache", {}),
    ("history", []),
):
    st.session_state.setdefault(_key, _default)

# 报头：三个页面共用
st.markdown(
    """
    <div class="masthead">
      <div class="masthead-brand">复合材料铺层优化工作台</div>
      <div class="masthead-meta">CLT<span class="sep">/</span>Rayleigh-Ritz<span class="sep">/</span>NSGA-II<span class="sep">/</span>v1.0.0</div>
    </div>
    <div class="masthead-sub">四边简支方板 · T300/5208</div>
    """,
    unsafe_allow_html=True,
)

step_bar(st.session_state["step"])

if st.session_state["step"] == "mode":
    page_mode()
    st.stop()

if st.session_state["step"] == "params":
    page_params()
    st.stop()

# 计算结果页
job = st.session_state.get("design_job")
if job is None:
    _goto("mode")

scene = st.session_state.get("scene")

st.markdown(
    '<div class="sec-label"><b>03</b>计算结果</div>'
    '<div class="sec-title">设计结果与校核</div>',
    unsafe_allow_html=True,
)
_head_l, _head_m, _head_r = st.columns([2.4, 1, 1], gap="small")
with _head_l:
    st.caption(
        ("预设：" + scene if scene else "自定义工况")
        + f"、Nx = {job['nx']:.0f} N/m、Ny = {job['ny']:.0f} N/m"
        + f"、SF = {job['sf']:.2f}"
    )
with _head_m:
    if st.button("← 修改参数", key="back_params", width="stretch"):
        _goto("params")
with _head_r:
    if st.button("开始新的设计", key="restart_top", width="stretch"):
        _restart()

# 计算结果按 job 缓存，切换页签不再重复求解
_cache_key = (job["nx"], job["ny"], job["sf"], job["fast"], _demo_now())
_cached = st.session_state.get("_res_cache")

if _cached and _cached["key"] == _cache_key:
    res = _cached["res"]
    _elapsed = _cached.get("elapsed")
else:
    # 底层只提供了「一次调用算完」的接口，没有分阶段的回调，
    # 所以这里只给一条与实际执行一致的状态，不按阶段造进度。
    _hit_preset = (int(round(job["nx"])), int(round(job["ny"])),
                   round(float(job["sf"]), 2)) in DEMO_PRESETS
    if _demo_now() and _hit_preset:
        _label = "正在读取离线预设算例（不运行 GA）..."
        _hint = "读取事先算好的结果，通常不到 1 秒。"
    elif _demo_now():
        _label = "正在计算纯 0° 铺层（层数扫描，不运行 GA）..."
        _hint = "只对纯 0° 铺层做层数扫描与强度校核，通常不到 1 秒。"
    else:
        _label = ("正在运行 NSGA-II 寻优（层数扫描 + 铺层角度优化）..."
                  if not job["fast"] else
                  "正在运行 NSGA-II 寻优（快模式 40 × 30）...")
        _hint = "标准模式约 1 分钟，快模式约 10 秒；期间页面不会输出其他内容。"

    _t0 = time.time()
    _slot = st.empty()
    with _slot.container(), st.status(_label, expanded=True) as status:
        st.write(_hint)
        try:
            if _demo_now():
                res = demo_design(job["nx"], job["ny"], job["sf"])
            else:
                res = auto_design(job["nx"], job["ny"], job["sf"], job["fast"])
        except Exception as exc:                      # noqa: BLE001
            status.update(label="计算失败", state="error", expanded=True)
            st.error(f"计算过程中出现异常，没有产生结果："
                     f"{type(exc).__name__}: {exc}")
            st.caption("这是程序异常，不代表设计不达标。")
            with st.expander("怎么处理"):
                st.caption("可返回上一步调整参数后重试；若重复出现，"
                           "请把上面的异常信息反馈给开发者。")
            if st.button("← 返回修改参数", key="back_params_from_error_calc"):
                _goto("params")
            st.stop()
        _dt = time.time() - _t0
        status.update(label=f"计算完成（用时 {_dt:.1f} s）",
                      state="complete", expanded=False)
    # 算完就把状态条收掉，用时挪到下面的设计条件行，页面上不留一个空框
    _elapsed = time.time() - _t0
    _slot.empty()

    st.session_state["_res_cache"] = {"key": _cache_key, "res": res,
                                      "elapsed": _elapsed}
    _record_design(res, job)

st.caption(
    f"**设计条件**：Nx = {res['nx']:.0f} N/m、Ny = {res['ny']:.0f} N/m、"
    f"SF = {res['sf']:.2f}；"
    f"**Ncr_target = max(Nx, Ny)·SF = {res['target']:,.0f} N/m**；"
    f"{'双向受压' if res['biaxial'] else '单向受压'}；"
    f"运行模式 {mode_text()}、来源 {SOURCE_INFO[source_of(res)][0]}"
    + (f"；计算用时 {_elapsed:.1f} s" if _elapsed is not None else "")
)

# 不可行分支
if res["status"] == "infeasible":
    st.error(
        f"纯 0° 铺层在 N ≤ {ad.N_MAX} 层内无法满足 Ncr_target = "
        f"{res['target']:,.0f} N/m。按 Ncr ∝ t³ 估算，约需 "
        f"{res['need_plies']:.1f} 层（{res['need_plies'] * T_PLY:.2f} mm）。"
    )
    st.markdown("#### 纯 0° 层数扫描明细")
    st.dataframe(scan_table(res["rows"], res["target"]),
                 hide_index=True, width="stretch")
    st.caption("建议：减小 Nx / Ny / SF，或扩展 N ≤ 20 的层数扫描范围。")
    if st.button("← 返回修改参数", key="back_params_from_error"):
        _goto("params")
    st.stop()

# 结果区
mode = buckling_mode(tuple(res["stack"]), res["nx"], res["ny"])
ncr, target, lam = res["ncr"], res["target"], res["lam"]

tab1, tab2, tab3 = st.tabs(["设计结果", "失稳模态", "性能与参考对比"])

# Tab 1
with tab1:
    with st.container(key="card1"):
        st.markdown(
            '<div class="card-head"><span class="card-idx">01</span>'
            '<span class="card-title">设计结果</span>'
            '<span class="card-aside">CLT + Rayleigh-Ritz · M = 16</span></div>',
            unsafe_allow_html=True,
        )
        verdict_block(res)
        if scene == "新能源电池包上盖":
            st.caption("另：实际设计时还要补局部踩踏刚度校核。")

        layup_card(res["layup"])
        st.caption(
            f"铺层序列：{pretty_layup(res['layup'])}、"
            f"{res['n_plies']} 层、总厚 {res['thickness']:.2f} mm"
        )

        src = source_of(res)
        st.caption(source_line(res))
        if src == "preset":
            st.caption(f"对应预设算例：{res.get('source_label', '')}。")
        if res.get("isolated_preset"):
            st.caption(
                "该工况命中的离线算例来源尚未核实，本版本未采用，"
                "结果按纯 0° 估算给出。"
            )

        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric main">
                <div class="metric-l">临界屈曲载荷 Ncr</div>
                <div class="metric-v">{ncr:,.1f}<small>N/m</small></div>
                <div class="metric-n">Ncr = λ · max(Nx, Ny)，λ 由 CLT + Rayleigh-Ritz 求解</div>
              </div>
              <div class="metric">
                <div class="metric-l">相对目标</div>
                <div class="metric-v">{(ncr / target - 1) * 100:+.1f}<small>%</small></div>
                <div class="metric-n">目标 Ncr = max(Nx, Ny) · SF = {target:,.0f} N/m</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "计算方法：经典层合板理论（CLT）+ Rayleigh-Ritz 法，级数展开 M = 16。"
        )

        st.markdown("#### 铺层截面")
        st.plotly_chart(
            figure_layup(res["stack"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            '<div class="layup-legend">'
            f'<span style="background:{LAYUP_COLORS["0"]}"></span>0°'
            f'<span style="background:{LAYUP_COLORS["90"]}"></span>90°'
            f'<span style="background:{LAYUP_COLORS["+45"]}"></span>+45°'
            f'<span style="background:{LAYUP_COLORS["-45"]}"></span>−45°'
            f'<span style="background:{LAYUP_COLORS["other"]}"></span>其他'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("注：铺层角度在 ±1° 容差内视为等效（制造工艺误差）。")
        st.caption("铺层顺序：外表面 → 中面 → 外表面")

        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric">
                <div class="metric-l">铺层数 N</div>
                <div class="metric-v">{res['n_plies']}<small>层</small></div>
              </div>
              <div class="metric">
                <div class="metric-l">总厚度 t</div>
                <div class="metric-v">{res['thickness']:.2f}<small>mm</small></div>
              </div>
              <div class="metric">
                <div class="metric-l">面密度</div>
                <div class="metric-v">{res['areal']:.2f}<small>kg/m²</small></div>
              </div>
              <div class="metric">
                <div class="metric-l">Tsai-Hill FI</div>
                <div class="metric-v">{res['fi']:.2e}</div>
                <div class="metric-n">无量纲，&lt; 1 为安全</div>
              </div>
              <div class="metric">
                <div class="metric-l">临界倍数 <span class="lc">λ</span></div>
                <div class="metric-v">{lam:.2f}</div>
                <div class="metric-n">SF = {res['sf']:.2f}，λ {'≥' if lam >= res['sf'] else '<'} SF</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if res["biaxial"]:
            balanced = ad.balance_penalty(
                ad.half_stack(res["theta4"], res["n_plies"])
            ) < 0.02
            st.caption(
                f"均衡性：ΣQ16 = {res['s16']:+.3f} GPa，ΣQ26 = {res['s26']:+.3f} GPa → "
                f"{'均衡铺层（A16 = A26 = 0）' if balanced else '非均衡（存在拉剪耦合）'}"
            )
            st.caption(
                "设计角度 θ = [" + ", ".join(f"{t:.1f}" for t in res["theta4"]) + "]°"
            )

        if res["thinner"]:
            st.caption(
                f"减薄校核：若以角度优化替代纯 0° 基准来定层数，N = {res['thinner']['n']} 层"
                f"（t = {res['thinner']['n'] * T_PLY:.2f} mm）即可满足，"
                f"Ncr = {res['thinner']['ncr']:,.0f} N/m ≥ Ncr_target = {target:,.0f} N/m，"
                f"较当前方案减薄 {res['thickness'] - res['thinner']['n'] * T_PLY:.2f} mm。"
            )

        with st.expander("纯 0° 层数扫描明细（N = 4…20）"):
            if res["n_min"] != res["n_plies"]:
                st.caption(
                    f"纯 0° 最小可行层数 N_min = {res['n_min']}，"
                    f"对称铺层需偶数层，取 N = {res['n_plies']}。"
                )
            st.dataframe(scan_table(res["rows"], target),
                         hide_index=True, width="stretch")

        # 导出设计方案报告
        st.divider()
        pdf_bytes = build_report_pdf(res)
        if pdf_bytes:
            st.download_button(
                "导出设计方案报告 (PDF)",
                data=pdf_bytes,
                file_name="Report.pdf",
                mime="application/pdf",
                key="download_report",
                width="stretch",
            )
        elif not HAS_FPDF:
            st.caption("PDF 导出不可用：未安装 fpdf2（pip install -r requirements.txt）。")
        else:
            st.caption(
                "PDF 导出不可用：未找到中文字体（如 "
                r"C:\Windows\Fonts\simhei.ttf）。"
            )

# Tab 2
with tab2:
    with st.container(key="card2"):
        st.markdown(
            '<div class="card-head"><span class="card-idx">02</span>'
            '<span class="card-title">失稳模态</span>'
            '<span class="card-aside">SS1 四边简支</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"#### 一阶屈曲模态 (m = {mode['m']}, n = {mode['n']})")
        choice = st.radio(
            "显示模态",
            ["m = 1, n = 1（基本模态）",
             f"Ritz 临界模态 (m = {mode['m']}, n = {mode['n']})"],
            horizontal=True, key="mode_choice",
        )
        if choice.startswith("m = 1"):
            Z, mode_label = mode["w11"], "m = 1, n = 1（基本模态）"
        else:
            Z, mode_label = mode["w_actual"], f"m = {mode['m']}, n = {mode['n']}（Ritz 临界）"

        st.caption(
            f"当前显示：**{mode_label}**；板 a = b = {A_PLATE:.2f} m，四边简支 (SS1)"
        )
        st.plotly_chart(
            figure_mode_3d(mode["x"], mode["y"], Z),
            width="stretch", config=PLOTLY_CONFIG,
        )
        st.caption(
            f"色标 = w / w_max（无量纲）；Ritz 临界模态 m = {mode['m']}, n = {mode['n']}。"
        )

# Tab 3
with tab3:
    with st.container(key="card3"):
        st.markdown(
            '<div class="card-head"><span class="card-idx">03</span>'
            '<span class="card-title">性能与参考对比</span>'
            '<span class="card-aside">基准对比 / 离线算例</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"#### 同层数基准对比（N = {res['n_plies']} 层）")
        st.plotly_chart(
            figure_compare(res["base_ncr"], ncr, target),
            width="stretch", config={"displaylogo": False},
        )

        gain = (ncr / res["base_ncr"] - 1.0) * 100.0 if res["base_ncr"] > 0 else 0.0
        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric">
                <div class="metric-l">较 [0]{subscript(res['n_plies'])}<span class="lc">s</span> 基准的 Ncr 提升</div>
                <div class="metric-v">{gain:+.1f}<small>%</small></div>
                <div class="metric-n">基准为同层数 N 下的纯 0° 铺层，与优化铺层严格同厚度</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"量化对比：基准 [0]{subscript(res['n_plies'])}s = {res['base_ncr']:,.1f} N/m → "
            f"优化方案 = {ncr:,.1f} N/m，增量 {ncr - res['base_ncr']:+,.1f} N/m"
            f"（{gain:+.1f} %）；Ncr_target = {target:,.1f} N/m。"
            "Ncr 均由 CLT + Rayleigh-Ritz 求解（M = 16）。"
        )
        if not res["biaxial"]:
            st.caption("单向受压时纯 0° 铺层的 D11 最大，已是最优解，角度优化没有提升空间。")

        st.markdown("#### D 矩阵对比（同层数）")
        d0, d1 = res["d_zero"], res["d_opt"]
        n_ply, thick = res["n_plies"], res["thickness"]
        _spec_table(
            ["指标", f"纯 0°（N={n_ply}）", f"优化（N={n_ply}）"],
            [
                ("D11 (GPa·mm³)", f"{d0[0][0]:.1f}", f"{d1[0][0]:.1f}"),
                ("D22 (GPa·mm³)", f"{d0[1][1]:.1f}", f"{d1[1][1]:.1f}"),
                ("D66 (GPa·mm³)", f"{d0[2][2]:.1f}", f"{d1[2][2]:.1f}"),
                ("D16 (GPa·mm³)", f"{d0[0][2]:.2f}", f"{d1[0][2]:.2f}"),
                ("厚度 (mm)", f"{thick:.2f}", f"{thick:.2f}"),
            ],
        )
        st.caption("厚度相同，Ncr 的差异全部来自 D 矩阵；D16 由铺层次序决定，"
                   "角度配对只保证 A16 = A26 = 0。")

        st.divider()
        st.markdown("#### 离线参考算例对比")
        st.caption("同一铺层分别给出 Ritz、Navier 与离线 ABAQUS 的结果。")
        with st.expander("查看对照口径"):
            st.caption(
                "只有铺层与载荷都对应同一个模型时才给出差异，"
                "否则只标明不可比较及原因；差异不是模型验证误差。"
            )

        nx_cur, ny_cur = float(res["nx"]), float(res["ny"])
        ncr_ritz = float(res["ncr"])
        scale = max(nx_cur, ny_cur)
        D_cur = ad.d_of_angles(res["stack"])
        _d11 = max(abs(float(D_cur[0][0])), 1e-12)
        is_ortho = (abs(float(D_cur[0][2])) < 1e-6 * _d11
                    and abs(float(D_cur[1][2])) < 1e-6 * _d11)

        rows = [{
            "分析方法": "Ritz 法（CLT + 双向正弦级数，M = 16）",
            "Ncr (N/m)": f"{ncr_ritz:,.1f}",
            "模态": f"m = {mode['m']}, n = {mode['n']}",
            "相对差异": "—（基准）",
            "结论": "当前设计采用的解",
        }]

        if is_ortho:
            lam_nav, m_nav, n_nav = navier_orthotropic(
                D_cur, A_PLATE, B_PLATE, nx_cur, ny_cur
            )
            ncr_nav = float(lam_nav) * scale
            rel_nav = (ncr_nav - ncr_ritz) / ncr_ritz if ncr_ritz else 0.0
            rows.append({
                "分析方法": "Navier 解析解（特殊正交异性板）",
                "Ncr (N/m)": f"{ncr_nav:,.1f}",
                "模态": f"m = {m_nav}, n = {n_nav}",
                "相对差异": "< 1e-12" if abs(rel_nav) < 1e-12 else f"{rel_nav:+.2e}",
                "结论": "D16 = D26 = 0，Navier 解与 Ritz 应严格一致",
            })
        else:
            rows.append({
                "分析方法": "Navier 解析解（特殊正交异性板）",
                "Ncr (N/m)": "—",
                "模态": "—",
                "相对差异": "—",
                "结论": "不适用：D16 / D26 ≠ 0，特殊正交异性假设不成立",
            })

        cmp_ab = aba_compare(res, nx_cur, ny_cur)
        ref = cmp_ab["ref"]

        if cmp_ab["state"] == "no_case":
            rows.append({
                "分析方法": "ABAQUS 线性摄动（S4R, Buckle）",
                "Ncr (N/m)": "—",
                "模态": "—",
                "相对差异": "—",
                "结论": ABA_STATE["no_case"],
            })
        elif cmp_ab["state"] == "stack_mismatch":
            rows.append({
                "分析方法": "ABAQUS 线性摄动（S4R, Buckle）",
                "Ncr (N/m)": f"{ref['ncr']:,.1f}（参考铺层）",
                "模态": "—",
                "相对差异": "—",
                "结论": ABA_STATE["stack_mismatch"],
            })
        else:
            delta = cmp_ab["delta_pct"]
            rows.append({
                "分析方法": "ABAQUS 线性摄动（S4R, Buckle）",
                "Ncr (N/m)": f"{ref['ncr']:,.1f}",
                "模态": "m = 1, n = 1（云图中心单波）",
                "相对差异": ("—" if delta is None
                             else f"{delta:+.2f}%（非验证误差）"),
                "结论": ABA_STATE["conditional"],
            })

        _spec_table(
            ["分析方法", "Ncr (N/m)", "模态", "相对差异", "结论"],
            [[row["分析方法"], row["Ncr (N/m)"], row["模态"],
              row["相对差异"], row["结论"]] for row in rows],
        )

        if cmp_ab["state"] == "no_case":
            st.warning(
                "该工况没有离线参考算例：当前 (Nx, Ny) = "
                f"({nx_cur:.0f}, {ny_cur:.0f}) N/m 不在参考算例集合内，"
                "只有 Ritz 解与适用时的 Navier 解可以对照。"
            )
        elif cmp_ab["state"] == "stack_mismatch":
            st.caption(
                f"离线算例 {ref['odb']} 的参考铺层为 "
                f"{pretty_layup(cmp_ab['ref_layup'])}（{ref['n_plies']} 层），"
                f"参考 Ncr = {ref['ncr']:,.1f} N/m。"
                "当前铺层与之不同，两者不是同一个模型，因此不计算也不显示差异；"
                "仅载荷相同不足以称为验证。"
            )
        else:
            st.caption(
                f"离线算例 {ref['odb']}：参考铺层 {pretty_layup(cmp_ab['ref_layup'])}"
                f"（{ref['n_plies']} 层），一阶特征值 {ref['eigen']:.4f}，"
                f"按参考载荷换算 Ncr = {ref['ncr']:,.1f} N/m。"
                "铺层一致只说明这一项相同；材料定义、几何、网格、边界条件与载荷施加方式"
                "没有随包的 .cae / .odb 可核对，表中百分比是与离线参考值的数值差异，"
                "不是模型验证误差。"
            )

        img_name = ABAQUS_IMAGES.get((int(round(nx_cur)), int(round(ny_cur))))
        if img_name is None:
            st.caption("*当前为自定义工况，未生成 ABAQUS 云图。*")
        else:
            img_path = _figure_path(img_name)
            if img_path is not None:
                st.image(
                    str(img_path),
                    caption=f"离线算例 {ref['odb']}：一阶屈曲模态"
                            f"（Nx = {int(round(nx_cur))} / "
                            f"Ny = {int(round(ny_cur))} N/m，"
                            "U 云图，变形放大系数 50）",
                    width=760,
                )
            else:
                st.caption(f"未找到 figures/{img_name}，请确认图片随项目一起部署。")

        st.caption("证据等级：条件性参考对比，不构成验证结论。")
        with st.expander("查看证据等级与覆盖范围"):
            st.caption(
                "包内只能核对离线算例的工况、铺层、特征值与云图，"
                "无法复核材料定义、几何、网格、边界条件与加载方式，"
                "因此即便铺层一致，也只能算条件性参考对比，不构成验证结论。"
                "参考算例覆盖 "
                + "、".join(f"{a}/{b} N/m" for a, b in sorted(ABAQUS_PRESETS))
                + "，只比对一阶临界载荷，不提供实时求解。"
            )

# 页脚说明
st.divider()
_foot_l, _foot_r = st.columns([3, 1], gap="small")
with _foot_l:
    st.caption("完成一个工况后可直接开始新的设计；当前结果不会保存到文件。")
with _foot_r:
    if st.button("开始新的设计", key="restart_bottom", width="stretch"):
        _restart()

with st.expander("方法与说明"):
    st.markdown(
        f"""
**物理模型**　经典层合板理论（CLT）求 D 矩阵，双正弦级数 Rayleigh–Ritz 求屈曲载荷；
四边简支（SS1），板长宽 a = b = {A_PLATE:.2f} m，保留 D16 / D26 弯扭耦合。
临界倍数 λ 满足 λ·(Nx, Ny) 恰好使板屈曲，取 Ncr = λ·max(Nx, Ny)。

**优化流程**（与 `Auto_Design.py` 一致）
1. Ncr_target = max(Nx, Ny)·SF，扫描 N = 4…20 的纯 [0]N 铺层求最小可行层数
   （双向受压时取偶数层做对称铺层）；
2. 双向受压时固定层数，用 pymoo NSGA-II（种群 80，迭代 60 代）搜索 [θ1/θ2/θ3/θ4]s
   （目标含均衡性罚分），再用 SLSQP + Nelder–Mead 局部精修；
   单向受压时纯 0° 已是最优，不再搜索；
3. 用 Tsai-Hill 准则按设计载荷（工作载荷 × SF）做面内强度校核，FI < 1 才通过。

**快速模式**　仅把 GA 规模降为 40×30、精修级数降为 M = 10；最终 Ncr 统一用 M = 16
重新校核。取消勾选即与 `Auto_Design.py` 完全一致（约 1 分钟）。

**缓存**　结果按 (Nx, Ny, SF, 快速模式) 缓存，调整图表选项或重复点击不会重算。
"""
    )

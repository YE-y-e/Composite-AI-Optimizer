# -*- coding: utf-8 -*-
"""app.py —— 复合材料层合板屈曲自动设计 · Streamlit 交互界面。

运行
    streamlit run app.py

与 Auto_Design.py 的关系
    本界面不重复实现任何力学求解，全部直接调用 Auto_Design.py 中的计算逻辑：

        scan_pure_zero(nx, ny, target)   纯 [0]N 铺层 N = 4…20 层扫描
        optimize_layup(N, nx, ny, target) NSGA-II 角度搜索 + SLSQP / Nelder–Mead 精修
        ncr_equiv(theta, N, nx, ny)      指定铺层的临界载荷 Ncr 与临界倍数 λ
        strength_check(stack, nx, ny, sf) Tsai-Hill 面内强度校核
        half_stack / symmetric_stack / stacking_text / d_of_angles / areal_weight

    屈曲求解器来自 angle_optimize.py：经典层合板理论（CLT）求 D 矩阵，
    双正弦级数 Rayleigh–Ritz 求屈曲载荷，四边简支（SS1），保留 D16 / D26 弯扭耦合。

界面
    左侧：Nx、Ny、SF 滑块 + 「开始优化」按钮
    主区：① 推荐铺层与 Ncr  ② m = 1, n = 1 屈曲波形 3D 曲面  ③ 与纯 0° 的性能对比

部署模式（自动检测）
    Streamlit Community Cloud 的容器 HOME 通常为 /home/adminuser，应用会自动识别并
    进入"云端极速演示"：跳过 GA，直接返回预设案例 / 纯 0° 估算，保证限流下秒开。
    本地运行（Windows / macOS / 普通 Linux）自动进入"完整模式"，调用 Auto_Design.py
    运行真实 NSGA-II 寻优。也可用环境变量 APP_MODE=demo|local 强制指定。
"""

import contextlib
import os
import re
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import Auto_Design as ad
from angle_optimize import (A_PLATE, B_PLATE, DENSITY, M_VERIFY, T_PLY,
                            BucklingSolver)

# =============================================================================
# 0. 环境自动检测：云端演示 or 本地完整计算
# =============================================================================
def _detect_cloud_demo() -> bool:
    """自动判断是否运行在 Streamlit Community Cloud。

    判定顺序：
        1. 环境变量 APP_MODE 显式覆盖（demo/cloud 或 local/full），便于本地调试；
        2. Streamlit Cloud 容器的 HOME 通常为 /home/adminuser；
        3. 兼容常见的云端环境变量（STREAMLIT_CLOUD / IS_STREAMLIT_CLOUD /
           STREAMLIT_SHARING_MODE）。
    本地运行（Windows / macOS / 普通 Linux）默认返回 False，跑真实 GA。
    """
    mode = os.environ.get("APP_MODE", "").strip().lower()
    if mode in {"demo", "cloud", "1", "true", "yes"}:
        return True
    if mode in {"local", "full", "0", "false", "no"}:
        return False

    if os.environ.get("HOME", "").rstrip("/") == "/home/adminuser":
        return True

    for key in ("STREAMLIT_CLOUD", "IS_STREAMLIT_CLOUD", "STREAMLIT_SHARING_MODE"):
        value = os.environ.get(key, "").strip().lower()
        if value in {"1", "true", "yes", "cloud", "streamlit"}:
            return True
    return False


IS_CLOUD_DEMO = _detect_cloud_demo()

# =============================================================================
# 1. 页面与常量
# =============================================================================
st.set_page_config(
    page_title="复合材料层合板屈曲自动设计",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT = "Microsoft YaHei, SimHei, Segoe UI, Arial, sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}
N_GRID = 41          # 3D 曲面网格点数（每边）
CHART_H = 470        # 第二、三列图表高度

# 演示模式预设案例：键为 (Nx, Ny, SF)，值为事先用完整流程算好的结果。
# 其中 500/500 为 12 层等双轴经典解；1040/1040 与 1070/870 为 14 层结果。
DEMO_PRESETS = {
    (500, 500, 1.5): {
        "label": "500/500 经典等双轴",
        "n_plies": 12,
        "theta4": [-44.9, 44.9, 45.2, -45.2],
        "ncr": 2128.6,
        "base_ncr": 1077.6,
        "thinner": {"n": 10, "ncr": 1230.2},
    },
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

# 快速模式：临时降低 GA 规模与精修级数。退出时一定恢复全局参数。
_FAST_LOCK = threading.Lock()


@contextlib.contextmanager
def _lower_accuracy():
    """把 Auto_Design 的 GA 规模与精修求解器临时降档（快速模式）。

    只影响 GA 与局部精修；最终 Ncr 仍用还原后的 M = 16 求解器重新校核。
    """
    with _FAST_LOCK:
        saved = (ad.GA_POP, ad.GA_GEN, ad._SOLVER_CHK)
        ad.GA_POP, ad.GA_GEN = 40, 30
        ad._SOLVER_CHK = BucklingSolver(A_PLATE, B_PLATE, 10)
        try:
            yield
        finally:
            ad.GA_POP, ad.GA_GEN, ad._SOLVER_CHK = saved


# =============================================================================
# 2. 计算层：调用 Auto_Design.py
# =============================================================================
@st.cache_data(show_spinner=False, max_entries=24)
def auto_design(nx: int, ny: int, sf: float, fast: bool) -> dict:
    """复现 Auto_Design.main() 的设计流程，返回结构化结果（不打印终端）。"""
    nx, ny, sf = float(nx), float(ny), float(sf)
    target = max(nx, ny) * sf
    biaxial = nx > 0 and ny > 0

    # ① 纯 0° 层数扫描
    rows = ad.scan_pure_zero(nx, ny, target)
    feasible = [r for r in rows if r["ok"]]

    if not feasible:
        ncr_top = rows[-1]["ncr"]
        need = ad.N_MAX * (target / ncr_top) ** (1.0 / 3.0) if ncr_top > 0 else np.inf
        return {"status": "infeasible", "nx": nx, "ny": ny, "sf": sf,
                "target": target, "biaxial": biaxial, "rows": rows,
                "need_plies": float(need)}

    n_min = feasible[0]["N"]

    # ② 双向受压 → 角度优化；单向受压 → 纯 0° 即最优
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

        # 减薄潜力（与 Auto_Design 相同的工程提示）
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

    # ③ Tsai-Hill 强度校核
    st_res = ad.strength_check(stack, nx, ny, sf)
    d_opt = ad.d_of_angles(stack)
    d_zero = ad.d_of_angles([0.0] * n_plies)

    return {
        "status": "ok", "nx": nx, "ny": ny, "sf": sf, "fast": fast,
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
    """云端演示模式：只查预设表或做纯 0° 快速估算，绝不运行 GA。"""
    nx, ny, sf = float(nx), float(ny), float(sf)
    target = max(nx, ny) * sf
    biaxial = nx > 0 and ny > 0

    # 纯 0° 扫描本身很轻（约 0.2 s），保留它是为了让扫描表和"层数怎么定"的
    # 说明仍然真实，而不是凭空编一个层数。
    rows = ad.scan_pure_zero(nx, ny, target)
    feasible = [r for r in rows if r["ok"]]
    if not feasible:
        ncr_top = rows[-1]["ncr"]
        need = ad.N_MAX * (target / ncr_top) ** (1.0 / 3.0) if ncr_top > 0 else np.inf
        return {"status": "infeasible", "nx": nx, "ny": ny, "sf": sf,
                "target": target, "biaxial": biaxial, "rows": rows,
                "need_plies": float(need), "demo": True,
                "demo_source": "estimate"}

    n_min = feasible[0]["N"]
    preset = DEMO_PRESETS.get((int(round(nx)), int(round(ny)), round(sf, 2)))

    if preset is not None:
        n_plies = preset["n_plies"]
        theta4 = [float(t) for t in preset["theta4"]]
        ncr = float(preset["ncr"])
        base_ncr = float(preset["base_ncr"])
        thinner = preset["thinner"]
        source = "preset"
        label = preset.get("label", "预设案例")
    else:
        # 未命中预设：不跑 GA，直接用"纯 0° 最小可行层数"给出快速估算。
        n_plies = n_min if (not biaxial or n_min % 2 == 0) else n_min + 1
        base_ncr = float(next(r for r in rows if r["N"] == n_plies)["ncr"])
        ncr = base_ncr
        theta4 = [0.0, 0.0, 0.0, 0.0]
        thinner = None
        source = "estimate"
        label = "纯 0° 快速估算（未运行 GA）"

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
        "demo": True, "demo_source": source, "demo_label": label,
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
    """用 Ritz 求解器求推荐铺层的临界模态（含 m、n 与归一化模态云图）。"""
    D = ad.d_of_angles(stack)
    solver = BucklingSolver(A_PLATE, B_PLATE, M_VERIFY)
    lam, c = solver.ncr_mode(D, nx, ny)

    # ① 实际临界模态：w(x,y) = ΣΣ c_mn sin(mπx/a) sin(nπy/b)
    x = np.linspace(0.0, A_PLATE, N_GRID)
    y = np.linspace(0.0, B_PLATE, N_GRID)
    m_idx = np.arange(1, M_VERIFY + 1)
    phi = np.sin(np.pi * np.outer(x, m_idx) / A_PLATE)
    psi = np.sin(np.pi * np.outer(y, m_idx) / B_PLATE)
    w = phi @ c @ psi.T                      # 形状 (nx, ny)
    w = w / np.max(np.abs(w))                # 归一化
    m_dom, n_dom = np.unravel_index(int(np.argmax(np.abs(c))), c.shape)

    # ② 题目要求的 m = 1, n = 1 单半波
    X, Y = np.meshgrid(x, y)
    w11 = np.sin(np.pi * X / A_PLATE) * np.sin(np.pi * Y / B_PLATE)

    return {"lam": float(lam), "x": x, "y": y,
            "w_actual": w.T, "w11": w11,
            "m": int(m_dom) + 1, "n": int(n_dom) + 1}


# =============================================================================
# 3. 绘图层：Plotly
# =============================================================================
def _surface_trace(X, Y, Z):
    """统一风格的曲面 trace（动画帧复用，保证颜色标尺固定）。"""
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
    """屈曲波形 3D 曲面 + 幅值演化动画（0 → 1），鼠标拖动可旋转。"""
    X, Y = np.meshgrid(x, y)
    amps = np.linspace(0.06, 1.0, 14)
    frames = [go.Frame(data=[_surface_trace(X, Y, Z * a)], name=f"{a:.3f}")
              for a in amps]

    fig = go.Figure(data=[_surface_trace(X, Y, Z * amps[-1])], frames=frames)

    play = dict(label="▶ 播放", method="animate",
                args=[None, {"frame": {"duration": 90, "redraw": True},
                             "fromcurrent": False,
                             "transition": {"duration": 0}}])
    pause = dict(label="⏸ 暂停", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}}])

    steps = [dict(method="animate", label=f"{a:.0%}",
                  args=[[f.name], {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}])
             for a, f in zip(amps, frames)]

    fig.update_layout(
        height=CHART_H,
        margin=dict(l=0, r=0, t=34, b=46),
        paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        updatemenus=[dict(type="buttons", direction="right", showactive=False,
                          x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                          pad=dict(l=0, r=4, t=0, b=0),
                          bgcolor="#ffffff", bordercolor="#d6e4ff",
                          font=dict(size=11, color="#0b2e6b"),
                          buttons=[play, pause])],
        sliders=[dict(active=len(amps) - 1, x=0.02, y=-0.02, len=0.96,
                      currentvalue=dict(prefix="变形幅值 ", font=dict(size=11)),
                      font=dict(size=10),
                      bgcolor="#f1f5f9", bordercolor="#d6e4ff",
                      tickcolor="#94a3b8", steps=steps)],
        scene=dict(
            xaxis=dict(title="x (m)", range=[0.0, A_PLATE],
                       showbackground=True, backgroundcolor="#f6f8fb",
                       gridcolor="#e2e8f0", zeroline=False),
            yaxis=dict(title="y (m)", range=[0.0, B_PLATE],
                       showbackground=True, backgroundcolor="#f6f8fb",
                       gridcolor="#e2e8f0", zeroline=False),
            zaxis=dict(title="w / w_max", range=[-1.2, 1.2],
                       showbackground=True, backgroundcolor="#fbfcfe",
                       gridcolor="#e8edf5", zeroline=False),
            aspectmode="manual", aspectratio=dict(x=1.15, y=1.15, z=0.52),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.05)),
        ),
    )
    return fig


def figure_compare(base_ncr: float, ncr: float, target: float) -> go.Figure:
    """柱状图：优化推荐 vs 纯 0° 同层数铺层，附目标载荷参考线。"""
    labels = ["纯 0° 铺层", "优化推荐"]
    values = [base_ncr, ncr]
    fig = go.Figure(go.Bar(
        x=labels, y=values, width=0.55,
        marker=dict(color=["#94a3b8", "#16a34a"],
                    line=dict(color="#1f2937", width=1)),
        text=[f"{base_ncr:,.0f}", f"{ncr:,.0f}"],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=14, color="#111827"),
        hovertemplate="%{x}<br>Ncr = %{y:,.0f} N/m<extra></extra>",
    ))
    fig.add_hline(
        y=target, line=dict(color="#dc2626", dash="dash", width=2),
    )
    fig.add_annotation(
        x=0.015, y=target, xref="paper", yref="y", showarrow=False,
        text=f"目标 Ncr = {target:,.0f} N/m", xanchor="left", yanchor="bottom",
        font=dict(color="#dc2626", size=11),
        bgcolor="rgba(255,255,255,0.88)", borderpad=3,
    )
    top = max(values + [target]) * 1.20
    fig.update_layout(
        height=CHART_H,
        margin=dict(l=64, r=14, t=46, b=10),
        paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        title=dict(text="同层数下的屈曲承载能力", x=0.02, y=0.98,
                   xanchor="left", yanchor="top",
                   font=dict(size=13, color="#334155")),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=dict(title="临界屈曲载荷 Ncr (N/m)", range=[0, top]),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


# =============================================================================
# 4. 界面小组件
# =============================================================================
def pretty_layup(layup: str) -> str:
    """显示用铺层字符串：去掉无意义的小数尾零（-45.0 → -45，44.9 保留）。"""
    return re.sub(r"\.0(?=[/\]])", "", layup)


def layup_card(layup: str) -> None:
    """推荐铺层的高亮卡片（字号自适应，只在斜杠处换行）。"""
    shown = pretty_layup(layup)
    size = 22 if len(shown) <= 34 else 18 if len(shown) <= 44 else 15
    shown = shown.replace("/", "/<wbr>")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#eef4ff 0%,#f8fbff 100%);
                    border:1px solid #d6e4ff;border-radius:14px;
                    padding:16px 18px;margin-bottom:12px;text-align:center;">
          <div style="font-size:11px;letter-spacing:2px;color:#64748b;
                      font-weight:600;margin-bottom:6px;">FINAL LAYUP</div>
          <div style="font-family:Consolas,'Cascadia Mono',monospace;
                      font-size:{size}px;font-weight:700;color:#0b2e6b;
                      line-height:1.4;word-break:normal;
                      overflow-wrap:normal;">{shown}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scan_table(rows, target: float) -> pd.DataFrame:
    """纯 0° 扫描结果表。"""
    return pd.DataFrame({
        "N": [r["N"] for r in rows],
        "t (mm)": [round(r["t"], 3) for r in rows],
        "Ncr (N/m)": [round(r["ncr"], 1) for r in rows],
        "λ": [round(r["lam"], 2) for r in rows],
        "Ncr / Ncr_target": [round(r["ncr"] / target, 2) for r in rows],
        "判定": ["✓ 可行" if r["ok"] else "✗ 不足" for r in rows],
    })


def welcome() -> None:
    st.markdown("### 复合材料层合板屈曲自动设计")
    if IS_CLOUD_DEMO:
        st.markdown(
            "设定左侧的 **Nx / Ny / SF** 后点击 **开始优化**，"
            "云端演示会直接给出预设案例的推荐铺层、屈曲波形与性能对比，页面秒开；"
            "需要运行完整 GA 时，请在本地环境启动本项目（应用会自动检测为本地模式）。"
        )
    else:
        st.markdown(
            "设定左侧的 **Nx / Ny / SF** 后点击 **开始优化**，"
            "程序会调用 `Auto_Design.py` 完成层数扫描、铺层角度优化与强度校核，"
            "并给出推荐铺层、屈曲波形与性能对比。"
        )
    c1, c2, c3 = st.columns(3)
    c1.info("**① 推荐铺层**\n\n最终铺层 `[θ1/θ2/θ3/θ4]s`、Ncr 与安全裕度")
    c2.info("**② 屈曲波形**\n\nm = 1, n = 1 的 3D 曲面，可旋转、可播放幅值演化")
    c3.info("**③ 性能对比**\n\n优化结果与同层数纯 0° 铺层的 Ncr 柱状对比")
    st.caption(
        f"结构：a = b = {A_PLATE:.2f} m，四边简支（SS1）；"
        f"材料：T300/5208，单层 {T_PLY} mm，ρ = {DENSITY:.0f} kg/m³。"
    )


# =============================================================================
# 5. 侧边栏
# =============================================================================
with st.sidebar:
    st.markdown(
        "<style>[data-testid='stAppDeployButton']{display:none;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown("## ⚙️ 设计输入")
    st.caption("面内载荷（N/m）与目标安全系数，回车或拖动后点击「开始优化」。")

    nx_in = st.slider("纵向载荷 Nx (N/m)", 0, 2000, 500, 10)
    ny_in = st.slider("横向载荷 Ny (N/m)", 0, 2000, 500, 10)
    sf_in = st.slider("安全系数 SF", 1.0, 2.0, 1.5, 0.05, format="%.2f")

    if IS_CLOUD_DEMO:
        fast_in = True
        st.caption("☁️ 云端环境：已自动切换极速演示，GA 已跳过")
    else:
        fast_in = st.checkbox(
            "快速模式（数秒）", value=True,
            help="GA 40×30 + 精修级数 M=10；最终 Ncr 仍按 M=16 重新校核。"
                 "取消勾选则与 Auto_Design.py 完全一致（约 1 分钟）。",
        )
        st.caption("💻 本地环境：真实 NSGA-II 寻优已启用")
    run = st.button("🚀 开始优化", type="primary", width="stretch")

    st.divider()
    st.caption(f"结构：a = b = {A_PLATE:.2f} m，四边简支 (SS1)")
    st.caption(f"材料：T300/5208，单层 {T_PLY} mm，ρ = {DENSITY:.0f} kg/m³")
    st.caption("层数扫描：N = 4…20；双向受压取偶数层对称铺层")

if IS_CLOUD_DEMO:
    st.info("当前为云端极速演示模式，全量 GA 寻优请参考视频或本地运行。")

# =============================================================================
# 6. 计算触发（结果按参数缓存，切换图表不会重算）
# =============================================================================
if run:
    if nx_in == 0 and ny_in == 0:
        st.session_state["input_error"] = "Nx 与 Ny 不能同时为 0，请至少施加一个方向的载荷。"
    else:
        st.session_state.pop("input_error", None)
        st.session_state["design_job"] = {
            "nx": nx_in, "ny": ny_in, "sf": sf_in, "fast": fast_in,
        }

err = st.session_state.get("input_error")
job = st.session_state.get("design_job")

if err:
    st.error(err)

if job is None:
    welcome()
    st.stop()

current = {"nx": nx_in, "ny": ny_in, "sf": sf_in, "fast": fast_in}
if job != current:
    st.info(
        "⚠️ 侧边栏参数已修改。点击「开始优化」按新参数重新计算；"
        "下方仍显示上一次的设计结果。"
    )

if IS_CLOUD_DEMO:
    with st.spinner("演示模式：正在生成预设方案（已跳过 GA）…"):
        res = demo_design(job["nx"], job["ny"], job["sf"])
else:
    with st.spinner("正在调用 Auto_Design.py 的计算逻辑（快速模式数秒，标准模式约 1 min）…"):
        res = auto_design(job["nx"], job["ny"], job["sf"], job["fast"])

st.caption(
    f"**设计条件**：Nx = {res['nx']:.0f} N/m ｜ Ny = {res['ny']:.0f} N/m ｜ "
    f"SF = {res['sf']:.2f} ｜ "
    f"**Ncr_target = max(Nx, Ny)·SF = {res['target']:,.0f} N/m** ｜ "
    f"{'双向受压' if res['biaxial'] else '单向受压'} ｜ "
    f"{'演示模式' if res.get('demo') else ('快速模式' if res['fast'] else '标准模式')}"
)

# =============================================================================
# 7. 不可行分支
# =============================================================================
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
    st.stop()

# =============================================================================
# 8. 主界面三列
# =============================================================================
mode = buckling_mode(tuple(res["stack"]), res["nx"], res["ny"])

col1, col2, col3 = st.columns([1.0, 1.1, 1.0], gap="large")

# ---------------------------------------------------------------- 第一列
with col1:
    st.subheader("① 推荐铺层")
    layup_card(res["layup"])
    if res.get("demo"):
        if res.get("demo_source") == "preset":
            st.caption(f"演示模式 · 预设案例：{res.get('demo_label', '')}（已跳过 GA）")
        else:
            st.caption("演示模式 · 纯 0° 快速估算（未运行 GA，结果不代表角度最优）")

    ncr, target, lam = res["ncr"], res["target"], res["lam"]
    st.metric(
        "临界屈曲载荷 Ncr", f"{ncr:,.0f} N/m",
        f"{(ncr / target - 1) * 100:+.1f}% 相对目标",
        help="Ncr = λ·max(Nx, Ny)，λ 来自 CLT + Rayleigh–Ritz 屈曲求解。",
    )

    a, b = st.columns(2)
    a.metric("层数 N", f"{res['n_plies']}")
    b.metric("总厚度 t", f"{res['thickness']:.2f} mm")
    a, b = st.columns(2)
    a.metric("面密度", f"{res['areal']:.2f} kg/m²")
    b.metric("Tsai-Hill FI", f"{res['fi']:.2e}")

    st.caption(f"临界倍数 λ = {lam:.2f}，" +
               ("≥ " if lam >= res["sf"] else "< ") + f"SF = {res['sf']:.2f}")

    if res["biaxial"]:
        balanced = ad.balance_penalty(ad.half_stack(res["theta4"], res["n_plies"])) < 0.02
        st.caption(
            f"均衡性：ΣQ16 = {res['s16']:+.3f}，ΣQ26 = {res['s26']:+.3f} GPa → "
            f"{'均衡铺层 ✓（A16 = A26 = 0）' if balanced else '非均衡 ✗'}"
        )
        st.caption(
            "设计角度 θ = [" + ", ".join(f"{t:.1f}" for t in res["theta4"]) + "]°"
        )

    if ncr >= target and res["fi"] < 1.0:
        st.success(f"✓ 屈曲与强度双重校核通过（Ncr ≥ {target:,.0f} N/m，FI < 1）")
    elif ncr < target:
        st.warning(
            f"⚠ 屈曲承载不足：Ncr = {ncr:,.0f} N/m < 目标 {target:,.0f} N/m，"
            f"建议增加层数。"
        )
    if res["fi"] >= 1.0:
        st.error(
            f"⚠ Tsai-Hill 强度失效：FI = {res['fi']:.2e} ≥ 1"
            f"（最危险层为第 {res['fi_ply']} 层，θ = {res['fi_theta']:.1f}°），"
            f"必须增加厚度。"
        )

    if res["thinner"]:
        st.info(
            f"工程提示：若允许以角度优化替代纯 0° 基准来定层数，"
            f"N = {res['thinner']['n']} 层（{res['thinner']['n'] * T_PLY:.2f} mm）"
            f"采用同一角度族即可满足（Ncr ≈ {res['thinner']['ncr']:,.0f} N/m），"
            f"仍有减薄空间。"
        )

    with st.expander("纯 0° 层数扫描明细（N = 4…20）"):
        if res["n_min"] != res["n_plies"]:
            st.caption(
                f"纯 0° 最小可行层数 N_min = {res['n_min']}，"
                f"对称铺层需偶数层，取 N = {res['n_plies']}。"
            )
        st.dataframe(scan_table(res["rows"], target),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- 第二列
with col2:
    st.subheader("② 屈曲波形（3D）")

    choice = st.radio(
        "显示模态",
        ["m = 1, n = 1", "Ritz 实际临界模态"],
        horizontal=True, key="mode_choice",
    )
    if choice.startswith("m = 1"):
        Z, mode_label = mode["w11"], "m = 1, n = 1（单半波）"
    else:
        Z, mode_label = mode["w_actual"], f"Ritz 临界模态 m = {mode['m']}, n = {mode['n']}"

    st.caption(
        f"当前显示：**{mode_label}** ｜ 板 a = b = {A_PLATE:.2f} m，四边简支 (SS1)"
    )
    st.plotly_chart(
        figure_mode_3d(mode["x"], mode["y"], Z),
        width="stretch", config=PLOTLY_CONFIG,
    )
    st.caption(
        f"颜色 = w / w_max；鼠标拖动旋转、滚轮缩放，▶ 播放屈曲幅值演化。"
        f"该铺层的 Ritz 临界模态为 m = {mode['m']}, n = {mode['n']}。"
    )

# ---------------------------------------------------------------- 第三列
with col3:
    st.subheader("③ 性能对比")
    st.plotly_chart(
        figure_compare(res["base_ncr"], ncr, target),
        width="stretch", config={"displaylogo": False},
    )

    gain = (ncr / res["base_ncr"] - 1.0) * 100.0 if res["base_ncr"] > 0 else 0.0
    st.metric(
        "相对同层数纯 0° 的提升", f"{gain:+.1f} %",
        help="纯 0° 基准取同一层数 N 下的 [0]N 铺层，与优化铺层严格同厚度。",
    )
    if not res["biaxial"]:
        st.caption("单向受压时纯 0° 铺层的 D11 最大，已是最优解，角度优化没有提升空间。")

    d0, d1 = res["d_zero"], res["d_opt"]
    cmp_df = pd.DataFrame({
        "指标": ["Ncr (N/m)", "D11 (GPa·mm³)", "D22 (GPa·mm³)",
                 "D66 (GPa·mm³)", "D16 (GPa·mm³)", "厚度 (mm)"],
        f"纯 0°（N={res['n_plies']}）": [
            f"{res['base_ncr']:,.0f}", f"{d0[0][0]:.1f}", f"{d0[1][1]:.1f}",
            f"{d0[2][2]:.1f}", f"{d0[0][2]:.2f}", f"{res['thickness']:.2f}",
        ],
        f"优化（N={res['n_plies']}）": [
            f"{ncr:,.0f}", f"{d1[0][0]:.1f}", f"{d1[1][1]:.1f}",
            f"{d1[2][2]:.1f}", f"{d1[0][2]:.2f}", f"{res['thickness']:.2f}",
        ],
    })
    st.caption("D 矩阵对比（D16 由铺层次序决定；角度配对只保证 A16 = A26 = 0）")
    st.dataframe(cmp_df, hide_index=True, width="stretch")

# =============================================================================
# 9. 页脚说明
# =============================================================================
st.divider()
with st.expander("方法与说明"):
    st.markdown(
        f"""
**物理模型**　经典层合板理论（CLT）求 D 矩阵，双正弦级数 Rayleigh–Ritz 求屈曲载荷；
四边简支（SS1），板长宽 a = b = {A_PLATE:.2f} m，保留 D16 / D26 弯扭耦合。
临界倍数 λ 满足 λ·(Nx, Ny) 恰好使板屈曲，取 Ncr = λ·max(Nx, Ny)。

**优化流程**（与 `Auto_Design.py` 一致）
1. Ncr_target = max(Nx, Ny)·SF，扫描 N = 4…20 的纯 [0]N 铺层求最小可行层数
   （双向受压时取偶数层做对称铺层）；
2. 双向受压时固定层数，用 pymoo NSGA-II 搜索 [θ1/θ2/θ3/θ4]s（目标含均衡性罚分），
   再用 SLSQP + Nelder–Mead 局部精修；单向受压时纯 0° 已是最优，不再搜索；
3. 用 Tsai-Hill 准则按设计载荷（工作载荷 × SF）做面内强度校核，FI < 1 才通过。

**快速模式**　仅把 GA 规模降为 40×30、精修级数降为 M = 10；最终 Ncr 统一用 M = 16
重新校核。取消勾选即与 `Auto_Design.py` 完全一致（约 1 分钟）。

**缓存**　结果按 (Nx, Ny, SF, 快速模式) 缓存，调整图表选项或重复点击不会重算。
"""
    )

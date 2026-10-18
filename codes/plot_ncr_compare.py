# -*- coding: utf-8 -*-
"""纯 [0]8 vs 角度优化铺层的屈曲载荷对比柱状图（双向受压 Nx=Ny=1000 N/m）。

数值全部由 angle_optimize.py 的同一套 CLT + Rayleigh–Ritz 求解器算出，
不手抄数字，保证图与脚本一致。

输出：
    figures/ncr_comparison.png
终端打印 8 层与 12 层两组对比的数值，供实验报告引用。
"""

import os
import sys

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import angle_optimize as ao

OUT_PNG = os.path.join("figures", "ncr_comparison.png")
TARGET = ao.TARGET_NCR


# ---------------------------------------------------------------- 通用 CLT
def d_of_stack(angles):
    """任意层数、任意角度的 D 矩阵（GPa·mm³），逐层按 z³ 积分。"""
    n = len(angles)
    z = np.linspace(-n * ao.T_PLY / 2.0, n * ao.T_PLY / 2.0, n + 1)
    D = np.zeros((3, 3))
    for th, w in zip(angles, np.diff(z ** 3) / 3.0):
        D += ao.qbar(th) * w
    return D


def ncr(angles, solver):
    return float(solver.ncr(d_of_stack(angles)))


def main():
    fine = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, ao.M_VERIFY)

    # ---------------- 8 层（1.0 mm）----------------
    base8 = [0.0] * 8
    opt8 = ao.stack_of([45.0, -45.0, -45.0, 45.0])
    n_base8 = ncr(base8, fine)
    n_opt8 = ncr(opt8, fine)

    # ---------------- 12 层（1.5 mm）----------------
    base12 = [0.0] * 12
    half12 = [45.0, -45.0, -45.0, 45.0, 45.0, -45.0]
    opt12 = half12 + half12[::-1]
    n_base12 = ncr(base12, fine)
    n_opt12 = ncr(opt12, fine)

    print("=" * 78)
    print("屈曲载荷对比（双向受压 Nx = Ny = 1000 N/m，四边简支，a = b = 0.5 m）")
    print("=" * 78)
    print(f"  N= 8（1.00 mm） 纯 [0]8            : Ncr = {n_base8:7.1f} N/m"
          f"   ({n_base8 / TARGET * 100:5.1f}% 需求)")
    print(f"  N= 8（1.00 mm） [45/-45/-45/45]s   : Ncr = {n_opt8:7.1f} N/m"
          f"   ({n_opt8 / TARGET * 100:5.1f}% 需求)"
          f"   提升 {(n_opt8 / n_base8 - 1) * 100:+.1f}%")
    print(f"  N=12（1.50 mm） 纯 [0]12           : Ncr = {n_base12:7.1f} N/m"
          f"   ({n_base12 / TARGET * 100:5.1f}% 需求)")
    print(f"  N=12（1.50 mm） [±45 均衡 12 层]    : Ncr = {n_opt12:7.1f} N/m"
          f"   ({n_opt12 / TARGET * 100:5.1f}% 需求)"
          f"   提升 {(n_opt12 / n_base12 - 1) * 100:+.1f}%")
    print(f"\n  8 层两方案均 < {TARGET:.0f} N/m：厚度不足，角度优化无法弥补。")
    print(f"  12 层两方案均 ≥ {TARGET:.0f} N/m：但纯 0° 仅余 "
          f"{(n_base12 / TARGET - 1) * 100:.1f}% 裕度，均衡铺层余 "
          f"{(n_opt12 / TARGET - 1) * 100:.1f}%。")

    # ---------------- 绘图（两张图：8 层主对比 + 12 层参考）----------------
    available = {f.name for f in font_manager.fontManager.ttflist}
    has_cn = False
    for name in ("Microsoft YaHei", "SimHei", "SimSun",
                 "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            has_cn = True
            break
    L = (lambda cn, en: cn) if has_cn else (lambda cn, en: en)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # ---------- 左：8 层对比（主图）----------
    ax = axes[0]
    labels = [L("[0]8\n（纯 0°）", "[0]8"), L("[45/-45/-45/45]s\n（均衡角度优化）", "[45/-45/-45/45]s")]
    vals = [n_base8, n_opt8]
    bars = ax.bar(labels, vals, color=["#d62728", "#2ca02c"], width=0.55,
                  edgecolor="black", linewidth=0.9)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f} N/m", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(TARGET, color="crimson", linestyle="--", linewidth=1.8, zorder=1)
    ax.text(-0.46, TARGET * 1.015, L(f"设计要求 {TARGET:.0f} N/m",
                                    f"target {TARGET:.0f} N/m"),
            ha="left", va="bottom", fontsize=10, color="crimson", zorder=5)
    ax.annotate("", xy=(0.28, n_base8), xytext=(0.28, n_opt8),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
    ax.annotate(L(f"+{(n_opt8 / n_base8 - 1) * 100:.0f}%", f"+{(n_opt8 / n_base8 - 1) * 100:.0f}%"),
                xy=(0.30, 0.5 * (n_base8 + n_opt8)), xytext=(8, 0),
                textcoords="offset points", fontsize=12, fontweight="bold",
                color="0.2", va="center")
    ax.set_ylim(0, TARGET * 1.12)
    ax.set_ylabel(L("临界屈曲载荷 Ncr (N/m)", "buckling load Ncr (N/m)"), fontsize=11)
    ax.set_title(L("N = 8（1.0 mm）：均衡角度优化把屈曲载荷提高 96%\n"
                   "但两者都远低于 1000 N/m —— 厚度不足",
                   "N = 8 (1.0 mm): +36% by angle optimisation, still far below target"),
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)

    # ---------- 右：N 与铺层的组合（12 层参考）----------
    ax = axes[1]
    labels2 = [L("纯 [0]12", "[0]12"), L("均衡 ±45 铺层 12 层", "balanced 12-ply")]
    vals2 = [n_base12, n_opt12]
    bars2 = ax.bar(labels2, vals2, color=["#ff9896", "#98df8a"], width=0.55,
                   edgecolor="black", linewidth=0.9)
    for b, v in zip(bars2, vals2):
        ax.annotate(f"{v:.0f}\n({v / TARGET * 100:.0f}%)",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.axhline(TARGET, color="crimson", linestyle="--", linewidth=1.8)
    ax.set_ylim(0, max(vals2) * 1.22)
    ax.set_ylabel(L("Ncr (N/m)", "Ncr (N/m)"), fontsize=11)
    ax.set_title(L("N = 12（1.5 mm）：增厚后才达标\n"
                   "均衡 ±45 铺层裕度远高于纯 0°",
                   "N = 12 (1.5 mm): only thicker laminates meet the target"),
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)

    fig.suptitle(L("双向受压（Nx = Ny = 1000 N/m）下 8/12 层层合板的临界屈曲载荷",
                   "Buckling load under biaxial compression"), fontsize=12.5)
    fig.text(0.01, -0.02,
             L("注：Ncr 由 CLT + Rayleigh–Ritz（四边简支，M=16）计算；"
               "[45/-45/-45/45]s 为均衡铺层（A16 = A26 = 0），Ncr = 624.7 N/m。",
               "Note: CLT + Rayleigh-Ritz (SSSS, M=16)."),
             fontsize=8.5, color="0.35")

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n已保存: {OUT_PNG}")


if __name__ == "__main__":
    main()

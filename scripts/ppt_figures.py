# -*- coding: utf-8 -*-
"""生成答辩 PPT 的五张核心图（F1–F5）。

用法：
    python scripts/ppt_figures.py          # 全部生成
    python scripts/ppt_figures.py f3,f4    # 只生成指定几张

输出：figures/ppt/ppt_f{1..5}_*.png（2400 px 宽、300 dpi、16:9）
每张图只讲一个结论，图标题即该页幻灯片的标题。
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "codes"))
import mpl_fonts  # noqa: E402
mpl_fonts.install_cjk_fallback()   # mathtext 混排里的中文回退到宋体
import Auto_Design as ad      # noqa: E402
import angle_optimize as ao   # noqa: E402

OUT = ROOT / "figures" / "ppt"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"

INK, BLUE, RED, GREEN, GOLD = "#0f172a", "#2563eb", "#b91c1c", "#059669", "#b45309"

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.edgecolor": "#94a3b8",
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
})
BASE_W, BASE_H = 8.0, 4.5


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p.name, f"{p.stat().st_size / 1024:.0f} KB")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- F1 角度—层数
def f1():
    """提升幅度只取决于角度；绝对承载随层数立方增长。"""
    solver = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)
    thetas = np.arange(-90, 91, 5.0)
    n_list = list(ad.N_ALLOWED)
    Z = np.zeros((len(n_list), len(thetas)))
    for i, n in enumerate(n_list):
        for j, th in enumerate(thetas):
            Z[i, j] = ad.evaluate_candidate([th, -th, -th, th], n, 500, 500, 1.5,
                                            solver=solver)["ncr"]
    zero = int(np.argmin(np.abs(thetas)))
    gain = (Z / Z[:, [zero]] - 1.0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(BASE_W, BASE_H),
                             gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.46))
    ax = axes[0]
    im = ax.imshow(gain, aspect="auto", origin="lower", cmap="viridis",
                   extent=[thetas[0], thetas[-1], n_list[0], n_list[-1]],
                   vmin=0, vmax=float(np.percentile(gain, 99.5)))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("提升（%）", fontsize=10)
    for t in (-45.0, 45.0):
        ax.axvline(t, color="white", lw=1.0, ls="--", alpha=0.9)
    ax.text(0, 17.2, "±45° 族", color="white", fontsize=11.5,
            ha="center", va="bottom")
    ax.set_xlabel("设计角 θ（平衡族 [θ/−θ/−θ/θ]s）")
    ax.set_ylabel("层数 N")
    ax.set_title("① 提升幅度只取决于角度：±45° 约 +90%，与层数几乎无关", fontsize=11)

    ax = axes[1]
    j45 = int(np.argmin(np.abs(thetas + 45)))
    ax.plot(n_list, Z[:, j45], "o-", color=BLUE, ms=5, lw=1.5, label="θ = −45°（±45° 族）")
    ax.plot(n_list, Z[:, zero], "s--", color=RED, ms=4.5, lw=1.3, label="θ = 0°（纯 0°）")
    ax.set_yscale("log")
    ax.set_xlabel("层数 N")
    ax.set_ylabel("$N_{cr}$ (N/m，对数轴)")
    ax.set_title("② 绝对承载随层数立方增长（两线平行）", fontsize=11)
    ax.grid(alpha=0.22, ls=":", which="both")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ppt_f1_angle_layer.png")


# ---------------------------------------------------------------- F2 同预算搜索
def f2():
    """5^10 → 4 角 625：能枚举，且枚举优于 GA。"""
    d = load("d3_compare.json")
    gb = np.array(d["grid_budgets"], float)
    gm = np.array(d["ga_mean"], float)
    gs = np.array(d["ga_std"], float)
    rb = np.array(d["random_budgets"], float)
    rm = np.array(d["random_mean"], float)
    best = max(v["best_ncr"] for v in d["enumeration"].values())
    target = d["case"]["target"]

    fig, ax = plt.subplots(figsize=(BASE_W, BASE_H))
    ax.fill_between(gb, np.maximum(gm - gs, 1), gm + gs, color=BLUE, alpha=0.15,
                    label="GA ±1σ（10 seed）")
    ax.plot(gb, gm, "o-", color=BLUE, ms=3.2, lw=1.5, label="GA（NSGA-II，逐代最优）")
    ax.plot(rb, rm, "s--", color=RED, ms=5, lw=1.5, label="离散网格随机抽样")
    ax.axhline(best, color=GREEN, ls=":", lw=1.4)
    ax.text(26, best * 1.012, f"离散枚举 625 次 = {best:.1f} N/m（全局最优）",
            color=GREEN, fontsize=9.5)
    ax.axhline(target, color=GOLD, ls="-.", lw=1.1)
    ax.text(26, target * 1.03, f"设计载荷 {target:.0f} N/m", color=GOLD, fontsize=9.5)
    ax.annotate(f"GA 2,400 次仍差 {(1 - gm[-1] / best) * 100:.1f}%",
                xy=(gb[-1], gm[-1]), xytext=(300, gm[-1] * 0.70),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate(f"随机 2,400 次仅 {rm[-1]:.0f} N/m\n（低于设计载荷）",
                xy=(rb[-1], rm[-1]), xytext=(80, 430),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("真实评价次数（一次 = 一次 Rayleigh–Ritz 求解）")
    ax.set_ylabel("找到的最优 $N_{cr}$ (N/m)")
    ax.set_ylim(0, best * 1.12)
    ax.grid(alpha=0.22, ls=":", which="both")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title(r"把 $5^{10}$ ≈ 976 万压到 4 角 625 组：所以能枚举，而且枚举比 GA 好", pad=10)
    save(fig, "ppt_f2_budget.png")


# ---------------------------------------------------------------- F3 逐层数阶梯
def f3():
    """厚度定生死：第一个越过判据的层数就是最轻方案。"""
    ps = load("preset_scan.json")
    key = "500|500|1.5"
    rows = ps[key]["rows"]
    N = np.array([r["n_plies"] for r in rows], float)
    ncr = np.array([r["ncr"] for r in rows], float)
    feas = np.array([r["feasible"] for r in rows])
    target = 1.2 * ps[key]["SF"] * max(ps[key]["Nx"], ps[key]["Ny"])

    fig, ax = plt.subplots(figsize=(BASE_W, BASE_H))
    ax.plot(N, ncr, "-", color=BLUE, lw=1.7, zorder=2)
    ax.plot(N[~feas], ncr[~feas], "o", ms=8, mfc="white", mec=RED, mew=1.7,
            label="未达判据（λ < 1.2·SF）", zorder=3)
    ax.plot(N[feas], ncr[feas], "o", ms=7.5, color=BLUE, mec="white", mew=1.2,
            label="可行（λ ≥ 1.2·SF）", zorder=3)
    ax.axhline(target, color=RED, ls="--", lw=1.2)
    ax.text(3.45, target * 0.80, f"判据 = 1.2·SF·max(Nx,Ny) = {target:.0f} N/m",
            color=RED, fontsize=9.2)

    p = ps[key]["primary"]
    ax.plot([p["n_plies"]], [p["ncr"]], marker="*", ms=21, color=GREEN,
            mec="white", mew=1.2, zorder=6, label="主推荐")
    ax.annotate(f"主推荐 {p['n_plies']} 层 · {p['ncr']:.1f} N/m\n"
                f"λ/SF = {p['lam'] / ps[key]['SF']:.2f}",
                xy=(p["n_plies"], p["ncr"]), xytext=(11.2, 300),
                fontsize=10, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                connectionstyle="arc3,rad=-0.15"))

    pr = ps["1000|200|1.2"]["primary"]
    ax.plot([pr["n_plies"]], [pr["ncr"]], marker="D", ms=9, color=GOLD,
            mec="white", mew=1.0, zorder=5)
    ax.annotate(f"电池包上盖 {pr['n_plies']} 层 · {pr['ncr']:.0f} N/m",
                xy=(pr["n_plies"], pr["ncr"]), xytext=(10.7, 2500),
                fontsize=9, color=GOLD,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))

    pr2 = ps["1500|800|2.0"]["primary"]
    ax.plot([pr2["n_plies"]], [pr2["ncr"]], marker="s", ms=9, color=GREEN,
            mec="white", mew=1.0, zorder=5)
    ax.annotate(f"太阳能帆板 {pr2['n_plies']} 层 · {pr2['ncr']:.0f} N/m",
                xy=(pr2["n_plies"], pr2["ncr"]), xytext=(11.0, 5000),
                fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))

    ax.annotate("8 层最优 627 < 900\n→ 8 层不可行", xy=(8, 627), xytext=(4.6, 1500),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlim(3.3, 15.4)
    ax.set_ylim(50, 7200)
    ax.set_xticks(list(N) + [14])
    ax.set_xlabel("层数 N")
    ax.set_ylabel("临界屈曲载荷 $N_{cr}$ (N/m，对数轴)")
    ax.grid(alpha=0.22, ls=":", which="both")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title("厚度定生死：第一个越过判据的层数，就是最轻方案", pad=9)
    save(fig, "ppt_f3_ladder.png")


# ---------------------------------------------------------------- F4 解的交叉验证
def f4():
    """四种方法算同一件事：偏差量级与数值一致。"""
    fig, axes = plt.subplots(1, 2, figsize=(BASE_W, BASE_H * 0.88),
                             gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.34))

    ax = axes[0]
    items = [("Navier 闭式\n(纯 0° 基准)", 1e-10, GREEN),
             ("级数外推\n(M = 40)", 0.007, GOLD),
             ("ABAQUS 实算\n(10 层当前设计)", 0.56, BLUE)]
    y = np.arange(3)[::-1]
    for (lab, v, c), yy in zip(items, y):
        ax.barh(yy, v, height=0.5, color=c, alpha=0.9)
        ax.text(v * 1.5, yy, ("< 1e-9%" if v < 1e-9 else f"{v:.3f}%"),
                va="center", fontsize=10.5, fontweight="bold",
                fontfamily="Times New Roman", color=c)
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 30)
    for xv, lab in ((0.1, "0.1%"), (1.0, "1%")):
        ax.axvline(xv, color=RED, ls="--", lw=0.9, alpha=0.8)
        ax.annotate(lab, xy=(xv, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points",
                    fontsize=8.5, color=RED, rotation=90, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _ in items], fontsize=9.5)
    ax.set_xlabel("相对 Ritz 的偏差（%，对数轴）")
    ax.set_title("① 偏差量级：从 1e-12 到 0.56%", fontsize=11.5)
    ax.grid(axis="x", alpha=0.22, ls=":", which="both")

    ax = axes[1]
    methods = ["Ritz\n(M = 16)", "级数外推\n(M = 40)", "ABAQUS\n实算"]
    vals = [1205.4, 1205.3, 1198.6]
    devs = [0.0, 0.007, 0.56]
    colors = [BLUE, GOLD, GREEN]
    x = np.arange(3)
    ax.bar(x, vals, width=0.55, color=colors, alpha=0.9, edgecolor="white")
    for i, (v, dv) in enumerate(zip(vals, devs)):
        ax.text(i, v + 10, f"{v:,.1f}", ha="center", fontsize=11,
                fontweight="bold", fontfamily="Times New Roman")
        if i:
            ax.text(i, v * 0.5, f"+{dv:.3f}%" if dv < 0.1 else f"+{dv:.2f}%",
                    ha="center", color="white", fontsize=10.5,
                    fontweight="bold", fontfamily="Times New Roman")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 1400)
    ax.set_ylabel("$N_{cr}$ (N/m)")
    ax.set_title("② 同一设计、三条独立算法：1,198.6–1,205.4", fontsize=11.5)
    ax.grid(axis="y", alpha=0.22, ls=":")
    ax.text(0.02, 0.04, "数值互校，不作验证误差；ABAQUS 为同一模板现场求解",
            transform=ax.transAxes, fontsize=8.5, color="#475569")
    save(fig, "ppt_f4_crosscheck.png")


# ---------------------------------------------------------------- F5 稳健性
def f5():
    """制造偏差：先控厚度，再谈角度。"""
    d = load("mc_deviation.json")
    tol = list(d["tolerances"])[0]
    pri = d["tolerances"][tol]["primary_10"]
    both = pri["both"]
    nom = pri["nominal_ncr"]
    target = d["target_ncr"]
    samples = np.load(DATA / "mc_deviation_samples.npz")["primary_10_both"]

    fig, axes = plt.subplots(1, 2, figsize=(BASE_W, BASE_H),
                             gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.3))
    ax = axes[0]
    ax.hist(samples, bins=45, color=BLUE, alpha=0.5, edgecolor="white",
            density=True, label="蒙特卡洛（5000 样本）")
    ax.axvline(nom, color=INK, ls="--", lw=1.2, label=f"标称 {nom:,.0f}")
    ax.axvline(both["p05"], color=GOLD, ls=":", lw=1.5, label=f"5% 分位 {both['p05']:,.0f}")
    ax.axvline(target, color=RED, lw=1.5, label=f"判据 {target:,.0f}")
    ax.set_xlabel("$N_{cr}$ (N/m)")
    ax.set_ylabel("样本密度")
    ax.set_ylim(0, 0.0072)
    ax.set_title("① 10 层主推荐：均值 1,208，失效概率 0.00%", fontsize=11)
    ax.grid(alpha=0.2, ls=":")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.text(0.98, 0.55, f"公差：{tol}", transform=ax.transAxes,
            ha="right", fontsize=9, color="#475569")

    ax = axes[1]
    labels = ["角度 + 厚度", "仅厚度", "仅角度"]
    vals = [both["cv_pct"], pri["thickness_only"]["cv_pct"], pri["theta_only"]["cv_pct"]]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, height=0.55, color=[BLUE, BLUE, GREEN], alpha=0.9)
    for yy, v in zip(y, vals):
        ax.text(v + 0.12, yy, f"{v:.2f}%", va="center", fontsize=10.5,
                fontweight="bold", fontfamily="Times New Roman")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("$N_{cr}$ 变异系数 CV（%）")
    ax.set_xlim(0, max(vals) * 1.4)
    ax.set_title("② 方差来源：厚度 5.97% vs 角度 0.04%", fontsize=11)
    ax.grid(axis="x", alpha=0.2, ls=":")
    ax.text(0.0, -0.30, r"$N_{cr} \propto D \propto t^3$：2% 厚度误差 → 约 6% 载荷分散",
            transform=ax.transAxes, fontsize=9, color="#475569")
    save(fig, "ppt_f5_robust.png")


FNS = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(FNS) if arg == "all" else [s.strip() for s in arg.split(",")]
    for n in names:
        FNS[n]()

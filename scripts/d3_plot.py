"""D3 配图（v2）：预算—质量曲线，重排标注避免重叠。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
})

ROOT = Path(r"F:\AI+力学")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpl_fonts  # noqa: E402
mpl_fonts.install_cjk_fallback()   # mathtext 混排里的中文回退到宋体
d = json.loads((ROOT / "Final_Submission" / "data" / "d3_compare.json")
               .read_text(encoding="utf-8"))

ga_x, ga_m, ga_s = d["grid_budgets"], d["ga_mean"], d["ga_std"]
rd_x, rd_m = d["random_budgets"], d["random_mean"]
grid_best = d["enumeration"]["n10"]["best_ncr"]
n8_best = d["enumeration"]["n8"]["best_ncr"]
target = d["case"]["target"]

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)

ax.plot(rd_x, rd_m, color="#dc2626", lw=2, marker="s", ms=3,
        label="随机搜索（只统计满足均衡的候选）")
ax.plot(ga_x, ga_m, color="#2563eb", lw=2, marker="o", ms=3,
        label="GA（NSGA-II，逐代最优，10 种子均值）")
ax.fill_between(ga_x,
                [m - s for m, s in zip(ga_m, ga_s)],
                [m + s for m, s in zip(ga_m, ga_s)],
                color="#2563eb", alpha=0.15, lw=0, label="GA ±1σ")

x_right = ga_x[-1]
for y, txt, color, va, off in [
        (grid_best, f"离散枚举最优 {grid_best:.0f} N/m（626 次评价，全局最优）",
         "#0d9488", "bottom", 4),
]:
    ax.axhline(y, color=color, ls="--" if color == "#0d9488" else ":", lw=1.3)
    xa, ha = (ga_x[0], "left") if color == "#64748b" else (x_right, "right")
    ax.annotate(txt, xy=(xa, y), xytext=(4 if ha == "left" else 0, off),
                textcoords="offset points", ha=ha, va=va,
                color=color, fontsize=8.2)

ax.axhline(target, color="#f59e0b", ls="-.", lw=1.2)
ax.annotate(f"目标 {target:.0f} N/m", xy=(ga_x[0], target), xytext=(4, -12),
            textcoords="offset points", color="#f59e0b", fontsize=8.2)

ax.set_xscale("log")
ax.set_xlabel("评价次数（一次 = 一次 Rayleigh–Ritz 求解）", fontsize=9.5)
ax.set_ylabel("该层数下最优临界载荷 $N_{cr}$ (N/m)", fontsize=9.5)
ax.set_title("同预算对照：随机搜索 vs GA（500/500 N/m，SF 1.5，10 层）",
             fontsize=10.5)
ax.set_ylim(0, max(grid_best, max(ga_m)) * 1.14)
ax.tick_params(labelsize=8.5)
ax.grid(alpha=0.25, ls=":")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42),
          fontsize=7.8, framealpha=0.95)

note = (f"随机抽样 24,000 次，满足均衡约束的候选仅 "
        f"{d['random_balanced_rate']*100:.1f}%\n"
        f"8 层离散域完全枚举最优 {n8_best:.0f} N/m < 目标 {target:.0f} N/m → 8 层无解\n"
        f"GA 需 5 倍预算（≈5,500 次）才追平离散最优；D4 起主推荐直接取离散枚举结果")
ax.text(0.015, 0.30, note, transform=ax.transAxes, fontsize=7.4,
        va="top", color="#334155",
        bbox=dict(boxstyle="round,pad=0.32", fc="#f8fafc", ec="#cbd5e1", lw=0.7))

fig.tight_layout()
out = ROOT / "Final_Submission" / "figures" / "budget_quality.png"
fig.savefig(out)
print("已保存", out, out.stat().st_size, "B")

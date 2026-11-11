# -*- coding: utf-8 -*-
"""画代理模型筛选器的两张图：精度（真值-预测）与筛选曲线。"""
import json
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "surrogate_screen.json"
NPZ = ROOT / "data" / "surrogate_screen_data.npz"
OUT = ROOT / "figures" / "surrogate_screen.png"

sys.path.insert(0, str(ROOT / "scripts"))
import mpl_fonts  # noqa: E402
mpl_fonts.install_cjk_fallback()   # mathtext 混排里的中文回退到宋体

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.unicode_minus": False,
})


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    z = np.load(NPZ)
    y_true, y_hat = z["y_true"], z["y_hat"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # 左：精度
    ax = axes[0]
    ax.scatter(y_true, y_hat, s=6, alpha=0.35, color="#2563eb", edgecolors="none")
    lo = min(y_true.min(), y_hat.min())
    hi = max(y_true.max(), y_hat.max())
    ax.plot([lo, hi], [lo, hi], color="#0f172a", ls="--", lw=0.9, label="理想 y = x")
    ax.set_xlabel("Ritz 真值 Ncr (N/m)", fontsize=9)
    ax.set_ylabel("代理预测 Ncr (N/m)", fontsize=9)
    ax.set_title(f"代理精度：R² = {d['model']['r2_test']:.4f}，"
                 f"MAE = {d['model']['mae']:.1f} N/m", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.15, lw=0.6)
    ax.tick_params(labelsize=8)

    # 右：筛选曲线（复核次数 vs 找到的最优）
    ax = axes[1]
    ks = [int(k) for k in d["screening"].keys()]
    best = [d["screening"][str(k)]["best_verified"] for k in ks]
    ax.plot(ks, best, "o-", color="#2563eb", lw=1.4, ms=4, label="筛选后复核的最优")
    ax.axhline(d["full_scan"]["best_true"], color="#b91c1c", ls="--", lw=1.0,
               label=f"全扫最优 {d['full_scan']['best_true']:,.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("送去 Ritz 复核的候选数（从 5000 个里挑）", fontsize=9)
    ax.set_ylabel("找到的最优 Ncr (N/m)", fontsize=9)
    ax.set_title("筛选曲线：50 次复核即可捞回全局最优", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.15, lw=0.6)
    ax.tick_params(labelsize=8)

    fig.suptitle("临界载荷代理模型作为筛选器（10 层、500/500 N/m，训练 2000 组 Ritz 标注）",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()

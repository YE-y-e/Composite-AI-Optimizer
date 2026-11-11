# -*- coding: utf-8 -*-
"""画制造偏差的蒙特卡洛分布图（读 data/mc_deviation.json + 样本 npz）。"""
import json
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mc_deviation.json"
SAMPLES = ROOT / "data" / "mc_deviation_samples.npz"
OUT = ROOT / "figures" / "mc_deviation.png"

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
    tol = next(iter(d["tolerances"]))
    cases = d["tolerances"][tol]
    npz = np.load(SAMPLES)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, c) in zip(axes, cases.items()):
        vals = npz[f"{key}_both"]
        ax.hist(vals, bins=60, density=True, color="#2563eb", alpha=0.25,
                edgecolor="#1e3a8a", linewidth=0.4)
        ax.axvline(c["nominal_ncr"], color="#0f172a", ls="--", lw=1.0,
                   label=f"标称 {c['nominal_ncr']:,.0f}")
        ax.axvline(c["both"]["p05"], color="#b45309", ls=":", lw=1.3,
                   label=f"5% 分位 {c['both']['p05']:,.0f}")
        ax.axvline(d["target_ncr"], color="#b91c1c", ls="-", lw=1.0,
                   label=f"判据 {d['target_ncr']:,.0f}")
        ax.set_title(c["label"], fontsize=10)
        ax.set_xlabel("临界载荷 Ncr (N/m)", fontsize=9)
        ax.set_ylabel("样本密度", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        ax.grid(alpha=0.15, lw=0.6)
        ax.set_xlim(min(vals.min(), d["target_ncr"]) * 0.985, vals.max() * 1.02)

    fig.suptitle("制造偏差下的临界载荷分布（5000 次蒙特卡洛；角度 σ = 1.0°，厚度 σ = 2%）",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("saved", OUT)


if __name__ == "__main__":
    main()

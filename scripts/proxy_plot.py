"""把 proxy_experiment.py 的 v2 结果画成一张对照图（图 8）。

左：小域（工程 5 档角，折角去重 + 均衡过滤后 44 组）
右：大域（15° 网格，同法处理后 248 组）
横轴：真实复核次数 K；纵轴：跨 5 个 seed 的平均得分（未命中记 0）。
两侧统计口径完全一致：都先在各 seed 内聚合，再跨 seed 取均值。
"""
import json
import pathlib
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "proxy_experiment.png"

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

ARMS = [("data/proxy_v2_small.json", "小域：工程 5 档角"),
        ("data/proxy_v2_large.json", "大域：15° 网格（含 5 档）")]


def load(path):
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    ks, rf, rd, hits = [], [], [], []
    refs = []
    for k, s in sorted(d["summary"].items(), key=lambda kv: int(kv[0])):
        ks.append(int(k))
        rf.append(s["rf_mean_score"])
        rd.append(s["random_mean_score"])
        hits.append(s["rf_hit_rate"])
        seed_refs = [r["by_budget"][k]["reference_full_sweep"]["best_lam"]
                     for r in d["runs"] if k in r["by_budget"]
                     and r["by_budget"][k]["reference_full_sweep"]["best_lam"] is not None]
        if seed_refs:
            refs.append(float(np.mean(seed_refs)))
    ts = [r["timing_s"] for r in d["runs"]]
    n_pool = d["runs"][0]["counts"]["screen_pool"]
    n_train = d["runs"][0]["counts"]["train"]
    total = d["runs"][0]["counts"]["pool_generated"]
    per_eval = st.median([t["full_sweep"] for t in ts]) / max(n_pool, 1)
    return {"k": ks, "rf": rf, "rd": rd, "ref": refs, "hit": hits,
            "per_eval": per_eval, "n_train": n_train, "pool": n_pool,
            "total": total, "cfg": d["config"]}


def _usable(rec):
    """没有有效 run（全部 insufficient/failed）时不要画图，也不要访问缺失的计时字段。"""
    runs = [r for r in rec.get("runs", []) if r.get("status") in (None, "ok")]
    return runs


def main():
    arms = []
    for path, label in ARMS:
        if (ROOT / path).exists():
            arms.append((label, load(path)))
        else:
            print("skip (missing)", path)
    if not arms:
        return 1

    fig, axes = plt.subplots(1, len(arms), figsize=(6.8 * len(arms), 4.4), sharey=True)
    if len(arms) == 1:
        axes = [axes]
    for ax, (label, d) in zip(axes, arms):
        k = np.array(d["k"], float)
        ax.plot(k, d["rf"], "o-", color="#2563eb", lw=1.8, ms=5,
                label="代理筛选（RF 排序取前 K 复核）")
        ax.plot(k, d["rd"], "s--", color="#b91c1c", lw=1.5, ms=4,
                label="随机挑选（同复核次数）")
        ax.plot(k, d["ref"], ":", color="#059669", lw=1.4, label="该池内可行最优（均值）")
        ax.set_xscale("log")
        ax.set_xlabel("真实复核次数 K（对数轴）")
        ax.set_title(f"{label}\n候选池 {d['total']} 组 → 训练 {d['n_train']} / 复核池 {d['pool']}",
                     fontsize=10.5)
        ax.grid(alpha=0.25, ls=":")
        lo = min(min(d["rd"]), min(d["rf"])) * 0.94
        ax.set_ylim(lo, max(d["ref"]) * 1.03)
        if all(h >= 0.999 for h in d["hit"]):
            ax.annotate("全部 K 下均命中可行解（本域 N=12 的可行率接近 100%）",
                        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8,
                        color="#334155")
    axes[0].set_ylabel("平均得分 λ（未命中记 0，跨 5 seed）")
    axes[0].legend(fontsize=8.5, loc="lower right", frameon=True)
    fig.suptitle("代理筛选 vs 随机挑选：同样复核次数下拿到的方案质量（修正口径后）",
                 fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=220)
    print("saved", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E1/E2：把"评价次数 × 单次耗时"的估算换成分项实测的端到端耗时。

数据来源：proxy_experiment.py 已记录的 timing_s（generate / label_train / fit /
eval_test / full_sweep / predict，单位秒，按 seed 记录）。
本脚本把它们按 seed 汇总，再把"复核 K 次"的成本按实测单次评价耗时折算，
输出全扫与代理两条路径的端到端耗时对比。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = [
    ("15° 网格·池 248", ROOT / "data/proxy_v2_large.json"),
    ("5 档工艺网格·池 44", ROOT / "data/proxy_v2_small.json"),
]
OUT = ROOT / "data/proxy_timing.json"


def load(path: Path):
    rec = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for run in rec["runs"]:
        t = run["timing_s"]
        n_pool = run["counts"]["screen_pool"]
        per_eval = t["full_sweep"] / max(n_pool, 1)
        rows.append(
            {
                "seed": run["seed"],
                "n_pool": n_pool,
                "per_eval": per_eval,
                "generate": t["generate"],
                "label": t["label_train"],
                "fit": t["fit"],
                "predict": t["predict"],
                "full_sweep": t["full_sweep"],
            }
        )
    return rec, rows


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def main() -> int:
    out = {}
    print(f"{'实验组':<22}{'池':>5}{'单次评价':>10}{'全扫':>9}"
          f"{'代理-固定开销':>14}{'K=4':>8}{'K=8':>8}{'K=32':>8}")
    for label, path in ARMS:
        if not path.exists():
            print(f"{label:<22} 缺 {path.name}")
            continue
        _, rows = load(path)
        per_eval = med(rows, "per_eval")
        gen = med(rows, "generate")
        lab = med(rows, "label")
        fit = med(rows, "fit")
        pred = med(rows, "predict")
        sweep = med(rows, "full_sweep")
        fixed_proxy = lab + fit + pred          # 代理的固定开销（不含复核）
        k_costs = {k: k * per_eval for k in (4, 8, 16, 32)}
        print(f"{label:<22}{rows[0]['n_pool']:>5}{per_eval*1000:>9.1f}ms"
              f"{gen + sweep:>8.1f}s{fixed_proxy:>13.1f}s"
              f"{fixed_proxy + k_costs[4]:>7.1f}s"
              f"{fixed_proxy + k_costs[8]:>7.1f}s"
              f"{fixed_proxy + k_costs[32]:>7.1f}s")
        out[label] = {
            "pool": rows[0]["n_pool"],
            "per_eval_s": per_eval,
            "generate_s": gen,
            "full_sweep_s": sweep,
            "baseline_total_s": gen + sweep,
            "proxy_label_s": lab,
            "proxy_fit_s": fit,
            "proxy_predict_s": pred,
            "proxy_fixed_s": gen + fixed_proxy,
            "proxy_total_s": {str(k): gen + fixed_proxy + v for k, v in k_costs.items()},
            "speedup": {str(k): (gen + sweep) / (gen + fixed_proxy + v)
                        for k, v in k_costs.items()},
            "per_seed": rows,
        }
        for k in (4, 8, 32):
            r = out[label]["speedup"][str(k)]
            print(f"    K={k:<3} 端到端 {out[label]['proxy_total_s'][str(k)]:.1f}s"
                  f"（全扫 {gen + sweep:.1f}s）→ {r:.1f}×")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写出", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

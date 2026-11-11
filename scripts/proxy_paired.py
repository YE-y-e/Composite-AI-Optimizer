"""按 seed 报配对差值：同一 seed、同一复核次数下 RF 与随机的得分差（E3）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("SF=1.5（高可行率）", ROOT / "data/proxy_v2_large.json"),
    ("SF=2.4（中）", ROOT / "data/proxy_v2_sf2p4.json"),
    ("SF=3.0（低）", ROOT / "data/proxy_v2_sf3p0.json"),
]


def load(path: Path):
    rec = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for run in rec["runs"]:
        seed = run["seed"]
        for k, block in run["by_budget"].items():
            rows.setdefault(int(k), {})[seed] = (
                block["rf"]["best_verified_lam"],
                block["random"]["expected_lam"],
                block["reference_full_sweep"]["n_feasible"],
                block["rf"]["found_feasible"],
            )
    return rec, rows


def main() -> int:
    for label, path in FILES:
        if not path.exists():
            print(f"{label}: 缺 {path.name}")
            continue
        rec, rows = load(path)
        seeds = sorted({s for v in rows.values() for s in v})
        feas = sorted({v[2] for v in rows.values() for v in v.values()})
        pool = rec["runs"][0]["counts"]["pool_generated"]
        print(f"\n=== {label}｜池 {pool}｜可行 {feas}")
        print("K    " + "  ".join(f"s{s:<6}" for s in seeds) + "  均值  ≥0 的 seed 数")
        for k in sorted(rows):
            cells, diffs = [], []
            for s in seeds:
                if s not in rows[k]:
                    continue
                rf = 0.0 if rows[k][s][0] is None else float(rows[k][s][0])
                rnd = 0.0 if rows[k][s][1] is None else float(rows[k][s][1])
                d = rf - rnd
                diffs.append(d)
                cells.append(f"{d:+.3f}")
            n_pos = sum(1 for d in diffs if d > 0)
            print(f"{k:<4} " + "  ".join(f"{c:<5}" for c in cells)
                  + f"  {np.mean(diffs):+.3f}      {n_pos}/{len(diffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

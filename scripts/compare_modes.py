# -*- coding: utf-8 -*-
"""快速模式（GA 40×30 / 枚举）与完整模式（GA 80×60 / 枚举）的对照。

对三条预设各跑两种模式，记录层数、λ、铺层、耗时；快速模式额外跑两遍以观察是否可重复。
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(r"F:\AI+力学\Final_Submission")
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad  # noqa: E402

CASES = [
    ("500/500 SF1.5", 500, 500, 1.5),
    ("1000/200 SF1.2", 1000, 200, 1.2),
    ("1500/800 SF2.0", 1500, 800, 2.0),
]

out = {}
for label, nx, ny, sf in CASES:
    row = {}
    for tag, fast, reps in (("fast", True, 2), ("full", False, 1)):
        runs = []
        for i in range(reps):
            t0 = time.perf_counter()
            sel = ad.select_design(nx, ny, sf, fast=fast)
            dt = time.perf_counter() - t0
            p = sel["primary"]
            runs.append({
                "n_plies": p["n_plies"], "lam": round(p["lam"], 5),
                "ncr": round(p["ncr"], 1), "layup": p["layup"],
                "seconds": round(dt, 2), "evals": sel["evals"],
            })
            print(f"{label} {tag}#{i+1}: N={p['n_plies']} λ={p['lam']:.4f} "
                  f"Ncr={p['ncr']:.1f} {p['layup']} ({dt:.1f}s)", flush=True)
        row[tag] = runs
    f = row["fast"][0]
    u = row["full"][0]
    row["delta_pct"] = round((u["ncr"] - f["ncr"]) / f["ncr"] * 100, 2)
    row["same_layup"] = (f["layup"] == u["layup"])
    print(f"  -> 完整比快速 {row['delta_pct']:+.2f}%，铺层{'相同' if row['same_layup'] else '不同'}", flush=True)
    out[label] = row

path = ROOT / "data" / "mode_compare.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", path)

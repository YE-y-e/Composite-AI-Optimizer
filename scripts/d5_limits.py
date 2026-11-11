"""D5-② 边界工况行为表：载荷/安全系数取到 UI 上限时，要几层、耗时多少、是否触顶。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import Auto_Design as ad  # noqa: E402

DATA = ROOT / "data"

CASES = [
    ("等双轴上界", 2000, 2000, 2.0),
    ("等双轴上界", 2000, 2000, 1.5),
    ("单轴受压（纵向）", 2000, 0, 2.0),
    ("单轴受压（纵向）", 2000, 0, 1.5),
    ("单轴受压（横向）", 0, 2000, 2.0),
    ("等双轴中值", 1000, 1000, 2.0),
]


def main():
    rows = []
    for label, nx, ny, sf in CASES:
        ad.reset_eval_count()
        t0 = time.time()
        try:
            sel = ad.select_design(nx, ny, sf, fast=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"label": label, "nx": nx, "ny": ny, "sf": sf,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"{label} {nx}/{ny} SF{sf}: 异常 {exc}")
            continue
        dt = time.time() - t0
        p = sel["primary"]
        rows.append({
            "label": label, "nx": nx, "ny": ny, "sf": sf,
            "n_plies": p["n_plies"],
            "ncr": round(p["ncr_m16"] if "ncr_m16" in p else p["ncr"], 2),
            "lam": round(p["lam_m16"] if "lam_m16" in p else p["lam"], 4),
            "lam_over_sf": round((p["lam_m16"] if "lam_m16" in p else p["lam"]) / sf, 3),
            "fi": (p.get("fi_m16") or p.get("fi")),
            "areal": round(ad.DENSITY * p["n_plies"] * ad.T_PLY / 1000.0, 2),
            "hit_nmax": p["n_plies"] >= max(ad.N_ALLOWED),
            "seconds": round(dt, 2),
            "evals": ad.eval_count(),
        })
        r = rows[-1]
        print(f"{label:14s} {nx:4d}/{ny:4d} SF{sf:.1f} → {r['n_plies']:2d} 层 "
              f"Ncr={r['ncr']:.1f} λ={r['lam']:.3f} λ/SF={r['lam_over_sf']:.2f} "
              f"{r['seconds']:.1f}s {r['evals']} 次"
              + ("  ← 触顶" if r["hit_nmax"] else ""))

    (DATA / "d5_limits.json").write_text(
        json.dumps({"n_max": max(ad.N_ALLOWED), "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写入 data/d5_limits.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""以完整模式（GA 80×60、不降级）重算三条预设的逐层数表，并覆盖 data/preset_scan.json。

结构与旧文件保持一致：
  { "nx|ny|sf": {case_id, Nx, Ny, SF, target, label, primary{...}, backup{...}, rows[...], seconds, evals} }
每个 primary/backup/row 里的 theta4 / n_plies / lam / ncr / fi / feasible / layup / rules 都来自
evaluate_candidate（M = 16 复核），因此与界面、PDF 用的是同一套数。
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(r"F:\AI+力学\Final_Submission")
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad  # noqa: E402

CASES = [
    ("500|500|1.5", 500, 500, 1.5, "wing_500_500_sf1.5", "无人机机翼蒙皮"),
    ("1000|200|1.2", 1000, 200, 1.2, "battery_1000_200_sf1.2", "新能源电池包上盖"),
    ("1500|800|2.0", 1500, 800, 2.0, "solar_1500_800_sf2.0", "航天器太阳能帆板"),
]


def slim(c):
    if c is None:
        return None
    return {
        "n_plies": c["n_plies"], "areal": c["areal"],
        "thickness": c["thickness"], "lam": c["lam"], "ncr": c["ncr"],
        "fi": c["fi"], "feasible": c["feasible"],
        "theta4": c["theta4"], "layup": c["layup"],
        "rules": c["rules"], "method": c["method"],
        "lam_m16": c["lam"], "ncr_m16": c["ncr"], "fi_m16": c["fi"],
        "feasible_m16": c["feasible"],
    }


out = {}
for key, nx, ny, sf, case_id, label in CASES:
    t0 = time.perf_counter()
    sel = ad.select_design(nx, ny, sf, fast=False)
    dt = time.perf_counter() - t0
    rows = []
    for r in sel["rows"]:
        rows.append({
            "n_plies": r["n_plies"], "areal": r["areal"],
            "thickness": r["n_plies"] * ad.T_PLY,
            "lam": r["lam"], "ncr": r["ncr"], "fi": r["fi"],
            "feasible": r["feasible"], "theta4": r["theta4"],
            "layup": r["layup"], "method": r["method"],
        })
    out[key] = {
        "case_id": case_id, "Nx": float(nx), "Ny": float(ny), "SF": float(sf),
        "target": max(nx, ny) * sf, "label": label,
        "primary": slim(sel["primary"]), "backup": slim(sel["backup"]),
        "rows": rows, "seconds": round(dt, 1), "evals": sel["evals"],
        "mode": "full",
    }
    p = sel["primary"]
    print(f"{label}: 主推荐 N={p['n_plies']} λ={p['lam']:.4f} Ncr={p['ncr']:.1f} "
          f"{p['layup']}  （{dt:.1f} s, {sel['evals']} 次评价）")
    if sel["backup"]:
        b = sel["backup"]
        print(f"          备选 N={b['n_plies']} λ={b['lam']:.4f} Ncr={b['ncr']:.1f} {b['layup']}")

path = ROOT / "data" / "preset_scan.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("written", path)

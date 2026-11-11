# -*- coding: utf-8 -*-
"""按修正后的（压缩取负）强度公式，重算数据文件里的全部 FI。

只改 fi / fi_m16 字段，其余数值（lam、ncr、耗时等）一个字节都不动，
因此不会引入"重跑优化器导致的漂移"。改动前后打印对照，便于写进过程日志。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(r"F:\AI+力学\Final_Submission")
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad  # noqa: E402


def fi_of(theta4, n_plies, nx, ny, sf):
    stack = ad.symmetric_stack([float(t) for t in theta4], int(n_plies))
    return float(ad.strength_check(stack, float(nx), float(ny), float(sf))["fi"])


def patch_record(rec, nx, ny, sf, log, tag):
    if not isinstance(rec, dict):
        return
    theta4 = rec.get("theta4")
    n = rec.get("n_plies")
    if theta4 is None or n is None:
        return
    for key in ("fi", "fi_m16"):
        if key in rec:
            old = rec[key]
            new = fi_of(theta4, n, nx, ny, sf)
            rec[key] = new
            log.append(f"{tag:28s} N={n:<3d} {key:6s} {old:.4e} → {new:.4e}")


def main():
    log = []

    p = ROOT / "data" / "preset_scan.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for key, case in d.items():
        nx, ny, sf = case["Nx"], case["Ny"], case["SF"]
        for name in ("primary", "backup"):
            if case.get(name):
                patch_record(case[name], nx, ny, sf, log, f"preset/{key}/{name}")
        for row in case.get("rows", []):
            patch_record(row, nx, ny, sf, log, f"preset/{key}/row")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    p2 = ROOT / "data" / "cases_d1.json"
    d2 = json.loads(p2.read_text(encoding="utf-8"))
    for case in d2:
        nx, ny, sf = case["Nx"], case["Ny"], case["SF"]
        for name in ("primary", "backup"):
            if case.get(name):
                patch_record(case[name], nx, ny, sf, log, f"d1/{case['case_id']}/{name}")
        for row in case.get("rows", []):
            patch_record(row, nx, ny, sf, log, f"d1/{case['case_id']}/row")
    p2.write_text(json.dumps(d2, ensure_ascii=False, indent=1), encoding="utf-8")

    p3 = ROOT / "data" / "d5_limits.json"
    d3 = json.loads(p3.read_text(encoding="utf-8"))
    for row in d3.get("rows", []):
        patch_record(row, row["nx"], row["ny"], row["sf"], log,
                     f"d5/{row['label']}")
    p3.write_text(json.dumps(d3, ensure_ascii=False, indent=1), encoding="utf-8")

    for line in log:
        print(line)
    print(f"patched {len(log)} records")


if __name__ == "__main__":
    main()

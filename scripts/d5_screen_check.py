"""D5-① 粗筛是否掉解：把 625 组全部用 M = 16 复算，对比 M = 10 粗筛的前 5 名。

同时给出 M = 10 与 M = 16 的排序一致性，用来支撑"粗筛加速不掉解"这句话。
"""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import angle_optimize as ao  # noqa: E402
import Auto_Design as ad  # noqa: E402

DATA = ROOT / "data"

ANGLES = ad.ANGLE_GRID            # (-90, -45, 0, 45, 90)
SUPER_HALF = list(itertools.product(ANGLES, repeat=4))   # 625 组半铺层

CASES = [
    ("500|500|1.5", 500, 500, 1.5, 10),
    ("1000|200|1.2", 1000, 200, 1.2, 10),
    ("1500|800|2.0", 1500, 800, 2.0, 14),
]


def eval_stack(theta4, n, nx, ny, solver, target, sf):
    stack = ad.symmetric_stack(list(theta4), n)
    rule = ad.check_rules(list(theta4), n)
    lam = solver.ncr(ad.d_of_angles(stack), nx, ny)
    fi = ad.strength_check(stack, nx, ny, sf)["fi"]
    ok = bool(rule["symmetric"] and rule["balanced"]
              and lam >= ad.MIN_MARGIN * sf and fi < 1.0)
    return {"theta4": list(theta4), "lam": float(lam),
            "ncr": float(lam) * max(nx, ny), "fi": float(fi), "ok": ok}


def main():
    s10 = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 10)
    s16 = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)
    out = {"grid_size": len(SUPER_HALF), "topk": ad.GRID_TOPK, "cases": []}

    for key, nx, ny, sf, n in CASES:
        t0 = time.time()
        target = max(nx, ny) * sf
        rows = []
        for theta4 in SUPER_HALF:
            stack = ad.symmetric_stack(list(theta4), n)
            if not ad.check_rules(list(theta4), n)["balanced"]:
                continue
            r10 = eval_stack(theta4, n, nx, ny, s10, target, sf)
            r = eval_stack(theta4, n, nx, ny, s16, target, sf)
            r["ncr10"] = r10["ncr"]
            rows.append(r)

        rank10 = sorted(rows, key=lambda r: -r["ncr10"])
        topk_theta = [tuple(r["theta4"]) for r in rank10[:ad.GRID_TOPK]]
        best16 = max(rows, key=lambda r: r["ncr"])
        hit = tuple(best16["theta4"]) in topk_theta
        pos = ([tuple(r["theta4"]) for r in rank10].index(tuple(best16["theta4"]))
               if hit else None)
        best10 = max(rows, key=lambda r: r["ncr10"])

        out["cases"].append({
            "case": key, "n_plies": n,
            "candidates_checked": len(rows),
            "best_m16_ncr": round(best16["ncr"], 2),
            "best_m16_sol": [round(t, 1) for t in best16["theta4"]],
            "best_m10_ncr": round(best10["ncr"], 2),
            "best_m10_sol": [round(t, 1) for t in best10["theta4"]],
            "m16_best_in_topk": hit,
            "m16_best_rank_by_m10": pos,
            "gap_topk_vs_all_pct": (None if hit else
                                    round((best16["ncr"] / best10["ncr"] - 1) * 100, 3)),
            "seconds": round(time.time() - t0, 1),
        })
        c = out["cases"][-1]
        print(f"{key}  N={n}  检查 {c['candidates_checked']} 组（{c['seconds']} s）")
        print(f"   M=16 真最优 {c['best_m16_ncr']:.2f} @ {c['best_m16_sol']}")
        print(f"   M=10 粗筛最优 {c['best_m10_ncr']:.2f} @ {c['best_m10_sol']}")
        print(f"   真最优是否落在前 {ad.GRID_TOPK}：" + (
            f"是（粗筛排名第 {pos + 1}）" if hit else "否 ← 掉解"))

    DATA.mkdir(exist_ok=True)
    (DATA / "d5_screen_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写入 data/d5_screen_check.json")


if __name__ == "__main__":
    main()

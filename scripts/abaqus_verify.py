"""用本机 Abaqus 复核某个铺层方案（在项目自带的 venv 里运行）。

    python scripts/abaqus_verify.py --preset "500|500|1.5"
    python scripts/abaqus_verify.py --theta4 -45,45,45,-45 --plies 8 --nx 500 --ny 500

做法：以 `abaqus/Job_1_Drone.inp` 为模板（50×50 S4R、四边简支、1 N/mm 面载荷），
替换铺层行、按载荷比缩放面载荷，调 `abaqus job=... interactive`，从 .dat 读一阶特征值
→ Ncr = λ × 1000 N/m → 与 Ritz（M = 16）比较 → 追加写入 data/abaqus_runs.json。

找不到 abaqus 时明确提示「未安装，本次跳过」并以退出码 2 结束，
不会伪装成已校验；调用方（网页按钮 / CI）据此区分「未跑」与「跑失败」。
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
sys.path.insert(0, str(ROOT / "scripts"))

import abaqus_inp  # noqa: E402

RUNS_PATH = ROOT / "data" / "abaqus_runs.json"


def abaqus_cmd():
    for cand in (os.environ.get("ABAQUS_CMD"), "abaqus",
                 r"C:\SIMULIA\Commands\abaqus.bat",
                 r"D:\SIMULIA\Commands\abaqus.BAT"):
        if not cand:
            continue
        found = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if found:
            return found
    return None


def preset_case(key):
    """从 data/preset_scan.json 取主推荐，并解析键里的 Nx|Ny|SF。"""
    path = ROOT / "data" / "preset_scan.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data.get(key)
    if not row:
        raise SystemExit("预设不存在：%s（可用：%s）"
                         % (key, ", ".join(sorted(data))))
    prim = row.get("primary", {})
    parts = [float(x) for x in key.split("|")]
    nx, ny = parts[0], parts[1]
    sf = parts[2] if len(parts) > 2 else 1.0
    return prim.get("theta4"), int(prim.get("n_plies") or row.get("n_plies")), nx, ny, sf


def ritz_ncr(theta4, n_plies, nx, ny):
    import angle_optimize as ao
    import Auto_Design as ad
    stack = ad.symmetric_stack([float(x) for x in theta4], n_plies)
    solver = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, ao.M_VERIFY)
    lam = solver.ncr(ad.d_of_angles(stack), float(nx), float(ny))
    return lam * max(float(nx), float(ny))


def main():
    ap = argparse.ArgumentParser(description="Abaqus 复核铺层方案")
    ap.add_argument("--theta4", help="设计角，逗号分隔，如 -45,45,45,-90")
    ap.add_argument("--plies", type=int, help="层数（偶数、对称）")
    ap.add_argument("--nx", type=float, default=500.0)
    ap.add_argument("--ny", type=float, default=500.0)
    ap.add_argument("--preset", help='改用预设主推荐，如 "500|500|1.5"')
    ap.add_argument("--t-ply", type=float, default=0.125, help="单层厚度 mm")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", action="store_true", help="保留临时目录便于排错")
    ap.add_argument("--no-record", action="store_true", help="不写入 abaqus_runs.json")
    args = ap.parse_args()

    if args.preset:
        theta4, plies, nx_p, ny_p, sf_p = preset_case(args.preset)
        args.nx, args.ny = nx_p, ny_p
        print("预设 %s：Nx=%g、Ny=%g N/m、SF=%g" % (args.preset, args.nx, args.ny, sf_p))
    else:
        if not args.theta4 or not args.plies:
            ap.error("需要 --theta4 与 --plies，或改用 --preset")
        theta4 = [float(x) for x in args.theta4.split(",")]
        plies = args.plies

    cmd = abaqus_cmd()
    if not cmd:
        print("未检测到 abaqus 命令，本次跳过有限元复核（不计为通过）。")
        print("可设 ABAQUS_CMD 指向 abaqus.bat，或把 Abaqus 的 Commands 目录加入 PATH。")
        return 2

    ritz = ritz_ncr(theta4, plies, args.nx, args.ny)
    print("铺层：%s ｜ N=%d ｜ Nx=%g、Ny=%g N/m" % (theta4, plies, args.nx, args.ny))
    print("Ritz（M = 16）：%.1f N/m" % ritz)
    print("调用：%s" % cmd)

    inp_text = abaqus_inp.build_inp(theta4, plies, args.nx, args.ny,
                                    t_ply_mm=args.t_ply)
    eigen, workdir = abaqus_inp.run_inp(cmd, inp_text, timeout=args.timeout,
                                        keep_dir=args.keep)
    ncr_fe = eigen * abaqus_inp.EDNOR_N_PER_MM
    delta = (ritz - ncr_fe) / ncr_fe * 100.0

    print("Abaqus：%.1f N/m（一阶特征值 %.4f）" % (ncr_fe, eigen))
    print("Ritz 相对 Abaqus：%+.2f%%" % delta)

    if not args.no_record:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "theta4": theta4, "n_plies": plies,
            "nx": args.nx, "ny": args.ny,
            "ncr_ritz": round(ritz, 6),
            "ncr_fe": round(ncr_fe, 6),
            "delta_pct": round(delta, 6),
            "eigen": eigen,
            "abaqus": cmd,
        }
        RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        hist = []
        if RUNS_PATH.exists():
            try:
                hist = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                hist = []
        hist.append(record)
        RUNS_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        print("已追加到 %s" % RUNS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

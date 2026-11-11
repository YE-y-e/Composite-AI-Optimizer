# -*- coding: utf-8 -*-
"""独立复核 Tsai-Hill 强度校核（FI）—— 与 codes/ 完全解耦的第二套实现。

目的
    不复用 `codes/angle_optimize.py` 与 `codes/Auto_Design.py` 的任何函数，
    从零实现一条独立的计算链：

        Q(材料主方向) --张量变换--> Qbar(θ) --厚度积分--> A
        N(设计载荷) --> ε0 = A^-1 N --> σ_global = Qbar·ε0
        σ_global --应力张量变换--> (σ1, σ2, τ12) --> Tsai-Hill FI

    然后回答三个问题：
      1) 三条工程预设（data/preset_scan.json 的 primary）的 FI 能否被复现？
      2) 压缩载荷的符号是否处理正确（该用 Xc/Yc 而不是 Xt/Yt）？
      3) 若不一致，差在单位 / 角度 / 层序 / σ 变换方向 / 强度选择的哪一步？

实现上的独立性（关键）
    - Qbar 用 **应力/应变张量变换矩阵** Qbar = Tσ^-1 · Q · Tε 推导，
      与工程代码里手写的 Qbar 分项公式是两条不同的代数路径；
      脚本内部再用 Tsai-Wu 不变量 U1..U5 的闭式解做第三路交叉验证。
    - 材料主方向应力用矩阵乘法 σ_mat = Tσ(θ)·σ_glob，而不是手写分项公式，
      这样"σ 变换方向"被独立检验一次。
    - 铺层序列按 Auto_Design 的规则独立重建（半叠层 4 角度循环 + 镜像对称），
      并打印出来与文件里的 `layup` 字符串对照。

单位约定（这是最容易出错的地方，脚本内全部显式标注）
    Qbar   : GPa
    t      : mm
    A      : GPa·mm = 1000 N/mm            （1 GPa = 1000 N/mm²）
    N      : N/m（工程输入）→ N/mm = N/m / 1000
    ε0     : 无量纲
    σ      : MPa（GPa × 1000）

符号约定
    工程输入的 Nx、Ny 是**压缩载荷的大小（正数）**。CLT 的 N 以拉为正，
    所以内部必须取 N = -[Nx, Ny, 0]·SF/1000。若漏掉这个负号，σ1、σ2 全部变号，
    压缩侧强度 Yc=246 就会被错换成拉伸侧 Yt=40（这就是历史上的那个 bug）。

用法
    python scripts/verify_tsai_hill.py            # 复核三条预设 + 三个解析对照
    python scripts/verify_tsai_hill.py --all      # 额外复核 preset_scan 里的每一行
    python scripts/verify_tsai_hill.py --json out.json

退出码：0 = 全部通过；1 = 有不一致（便于 CI 直接判定）。
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# 0. 输入常数（硬编码，不读取 codes/，保证复核对象是"数字"而不是"代码调用"）
# =============================================================================
E1, E2, G12, V12 = 181.0, 10.3, 7.17, 0.28      # GPa / -
T_PLY = 0.125                                    # mm
XT = XC = 1500.0                                 # MPa
YT, YC, S12 = 40.0, 246.0, 68.0                  # MPa

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRESET_JSON = ROOT / "data" / "preset_scan.json"
PRESETS = ("500|500|1.5", "1000|200|1.2", "1500|800|2.0")

REL_TOL = 1e-9          # 预设复现判据（相对差）
ANALYTIC_TOL = 1e-10    # 解析对照判据


# =============================================================================
# 1. 单层刚度：张量变换路径（与工程代码的分项公式相互独立）
# =============================================================================
def _q_material():
    """材料主方向平面应力折减刚度 Q（GPa）。"""
    nu21 = V12 * E2 / E1
    den = 1.0 - V12 * nu21
    return np.array([
        [E1 / den, V12 * E2 / den, 0.0],
        [V12 * E2 / den, E2 / den, 0.0],
        [0.0, 0.0, G12],
    ])


Q_MAT = _q_material()


def t_stress(a):
    """应力变换矩阵：σ_mat = Tσ(a) · σ_glob（a 为弧度，材料 1 轴相对 x 的转角）。"""
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c * c, s * s, 2.0 * s * c],
        [s * s, c * c, -2.0 * s * c],
        [-s * c, s * c, c * c - s * s],
    ])


def t_strain(a):
    """应变变换矩阵（工程剪应变）：ε_mat = Tε(a) · ε_glob。"""
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c * c, s * s, s * c],
        [s * s, c * c, -s * c],
        [-2.0 * s * c, 2.0 * s * c, c * c - s * s],
    ])


def qbar(theta_deg):
    """θ 度铺层的折算刚度 Qbar（GPa）。

    由 σ_glob = Tσ^-1 · Q · ε_mat = Tσ^-1 · Q · Tε · ε_glob 得
        Qbar = Tσ^-1 · Q · Tε
    """
    a = math.radians(theta_deg)
    return np.linalg.inv(t_stress(a)) @ Q_MAT @ t_strain(a)


def qbar_invariants(theta_deg):
    """Tsai-Wu 不变量闭式解（第三路独立校验 Qbar）。"""
    q11, q22, q12, q66 = Q_MAT[0, 0], Q_MAT[1, 1], Q_MAT[0, 1], Q_MAT[2, 2]
    u1 = (3.0 * q11 + 3.0 * q22 + 2.0 * q12 + 4.0 * q66) / 8.0
    u2 = (q11 - q22) / 2.0
    u3 = (q11 + q22 - 2.0 * q12 - 4.0 * q66) / 8.0
    u4 = (q11 + q22 + 6.0 * q12 - 4.0 * q66) / 8.0
    u5 = (q11 + q22 - 2.0 * q12 + 4.0 * q66) / 8.0
    a = math.radians(theta_deg)
    c2, s2 = math.cos(2.0 * a), math.sin(2.0 * a)
    c4, s4 = math.cos(4.0 * a), math.sin(4.0 * a)
    return np.array([
        [u1 + u2 * c2 + u3 * c4, u4 - u3 * c4, 0.5 * u2 * s2 + u3 * s4],
        [u4 - u3 * c4, u1 - u2 * c2 + u3 * c4, 0.5 * u2 * s2 - u3 * s4],
        [0.5 * u2 * s2 + u3 * s4, 0.5 * u2 * s2 - u3 * s4, u5 - u3 * c4],
    ])


# =============================================================================
# 2. 铺层序列与 A 矩阵
# =============================================================================
def half_stack(theta4, n_plies):
    """半叠层：4 个设计角循环填充到 N/2 层（与 Auto_Design 的构造规则一致）。"""
    return [float(theta4[i % 4]) for i in range(int(n_plies) // 2)]


def symmetric_stack(theta4, n_plies):
    """[θ1/θ2/θ3/θ4]s 的完整 N 层序列（半叠层 + 镜像）。"""
    half = half_stack(theta4, n_plies)
    return half + half[::-1]


def a_matrix(stack, t=T_PLY):
    """面内刚度 A = Σ Qbar_k · t_k，单位 GPa·mm（= 1000 N/mm）。"""
    A = np.zeros((3, 3))
    for theta in stack:
        A += qbar(theta) * t
    return A


# =============================================================================
# 3. 逐层应变/应力与 Tsai-Hill
# =============================================================================
def analyse(stack, nx, ny, sf, load_sign=-1.0):
    """给定铺层与压缩载荷，返回逐层应力与最大 FI。

    load_sign = -1 : 压缩（正确；内部 N 以拉为正）
    load_sign = +1 : 当作拉伸（复现历史 bug 用）
    """
    n = len(stack)
    # 工程输入 N/m → N/mm；压缩取负
    N = load_sign * np.array([nx, ny, 0.0]) * sf / 1000.0
    A_nmm = a_matrix(stack) * 1000.0                 # GPa·mm → N/mm
    eps = np.linalg.solve(A_nmm, N)                  # (εx, εy, γxy)

    plies = []
    for k, theta in enumerate(stack, start=1):
        sig_glob = qbar(theta) @ eps * 1000.0        # GPa → MPa
        a = math.radians(theta)
        sig_mat = t_stress(a) @ sig_glob             # MPa，材料主方向
        s1, s2, t12 = float(sig_mat[0]), float(sig_mat[1]), float(sig_mat[2])
        X = XT if s1 >= 0.0 else XC
        Y = YT if s2 >= 0.0 else YC
        fi = (s1 ** 2 / X ** 2 - s1 * s2 / X ** 2
              + s2 ** 2 / Y ** 2 + t12 ** 2 / S12 ** 2)
        plies.append({
            "ply": k, "theta": theta, "sig_glob": sig_glob,
            "s1": s1, "s2": s2, "t12": t12,
            "X": X, "Y": Y, "fi": fi,
        })

    worst = max(plies, key=lambda p: p["fi"])
    # 内力平衡自检：Σ t·σ_global 必须复现施加的 N（N/mm）
    n_check = sum(p["sig_glob"] * T_PLY for p in plies)   # MPa·mm == N/mm
    return {"eps": eps, "plies": plies, "worst": worst,
            "fi": worst["fi"], "N": N, "N_check": n_check,
            "stack": list(stack), "n": n}


def tsai_hill_scalar(s1, s2, t12, use="auto"):
    """FI 标量函数；use='tension' 强制用拉伸强度（复现历史 bug）。"""
    if use == "auto":
        X = XT if s1 >= 0.0 else XC
        Y = YT if s2 >= 0.0 else YC
    elif use == "tension":
        X, Y = XT, YT
    else:
        raise ValueError(use)
    return (s1 ** 2 / X ** 2 - s1 * s2 / X ** 2
            + s2 ** 2 / Y ** 2 + t12 ** 2 / S12 ** 2)


# =============================================================================
# 4. 自检：Qbar 的两条独立路径
# =============================================================================
def self_check_qbar(verbose=True):
    worst = 0.0
    for th in (0.0, 15.0, 30.0, 45.0, 60.0, -45.0, 90.0):
        worst = max(worst, float(np.max(np.abs(qbar(th) - qbar_invariants(th)))))
    nu21 = V12 * E2 / E1
    den = 1.0 - V12 * nu21
    checks = [
        ("Qbar(0) == Q", float(np.max(np.abs(qbar(0.0) - Q_MAT)))),
        ("Qbar(90)[0,0] == Q22", abs(qbar(90.0)[0, 0] - E2 / den)),
        ("Qbar(45)[2,2] == (Q11+Q22-2Q12)/4",
         abs(qbar(45.0)[2, 2] - (Q_MAT[0, 0] + Q_MAT[1, 1] - 2 * Q_MAT[0, 1]) / 4)),
        ("Qbar(90)[0,2] == 0（无拉剪耦合）", abs(qbar(90.0)[0, 2])),
    ]
    if verbose:
        print("  动力学/刚度路径自检")
        print(f"    张量变换 vs U 不变量闭式解：7 个角度最大互差 {worst:.2e} GPa")
        for name, val in checks:
            print(f"    {name:<42s} 偏差 {val:.2e} GPa")
        print(f"    Q11={Q_MAT[0,0]:.4f}  Q22={Q_MAT[1,1]:.4f}  "
              f"Q12={Q_MAT[0,1]:.4f}  Q66={Q_MAT[2,2]:.4f} GPa")
        print(f"    Qbar(30)[0,2]={qbar(30.0)[0,2]:.4f} GPa（拉剪耦合，量级应 ≈ +54）")
    return worst


# =============================================================================
# 5. 三条预设的复现
# =============================================================================
def check_presets(all_rows=False, verbose=True):
    data = json.loads(PRESET_JSON.read_text(encoding="utf-8"))
    results = []

    def one(key, tag, entry, nx, ny, sf):
        stack = symmetric_stack(entry["theta4"], entry["n_plies"])
        res = analyse(stack, nx, ny, sf)
        fi_file = float(entry["fi"])
        rel = (res["fi"] - fi_file) / fi_file
        w = res["worst"]
        layup = "[" + "/".join(f"{t:.1f}" for t in half_stack(
            entry["theta4"], entry["n_plies"])) + "]s"
        lin = float(np.max(np.abs(res["N_check"] - res["N"]))) / float(
            np.max(np.abs(res["N"])) if np.max(np.abs(res["N"])) else 1.0)
        row = {
            "key": key, "tag": tag, "n_plies": entry["n_plies"],
            "nx": nx, "ny": ny, "sf": sf, "layup": layup,
            "layup_file": entry.get("layup", ""),
            "fi_mine": res["fi"], "fi_file": fi_file, "rel": rel,
            "worst_ply": w["ply"], "worst_theta": w["theta"],
            "s1": w["s1"], "s2": w["s2"], "t12": w["t12"],
            "X_used": w["X"], "Y_used": w["Y"],
            "eps": res["eps"].tolist(), "balance_rel": lin,
        }
        results.append(row)
        return row

    for key in PRESETS:
        case = data[key]
        one(key, "primary", case["primary"],
            case["Nx"], case["Ny"], case["SF"])
        if all_rows:
            one(key, "backup", case["backup"],
                case["Nx"], case["Ny"], case["SF"])
            for i, r in enumerate(case["rows"], start=1):
                one(key, f"row{i}", r, case["Nx"], case["Ny"], case["SF"])

    if verbose:
        print("\n" + "=" * 96)
        print("1) 三条工程预设：独立实现 vs data/preset_scan.json")
        print("=" * 96)
        print(f"  {'工况':<14s}{'层数':>4s}  {'铺层':<34s}{'独立 FI':>13s}"
              f"{'文件 FI':>13s}{'相对差':>11s}")
        print("  " + "-" * 90)
        for r in results:
            if r["tag"] != "primary":
                continue
            print(f"  {r['key']:<14s}{r['n_plies']:>4d}  {r['layup']:<34s}"
                  f"{r['fi_mine']:>13.4e}{r['fi_file']:>13.4e}{r['rel']:>+11.2e}")
        print()
        for r in results:
            if r["tag"] != "primary":
                continue
            print(f"  [{r['key']}] 控制层 #{r['worst_ply']} θ={r['worst_theta']:7.2f}°  "
                  f"σ1={r['s1']:+9.4f}  σ2={r['s2']:+9.4f}  τ12={r['t12']:+9.4f} MPa  "
                  f"-> X={r['X_used']:.0f}, Y={r['Y_used']:.0f}")
            print(f"    ε0 = [{r['eps'][0]:+.6e}, {r['eps'][1]:+.6e}, "
                  f"{r['eps'][2]:+.6e}]   内力平衡残差 {r['balance_rel']:.1e}")

    return results


# =============================================================================
# 6. 解析可判定的对照
# =============================================================================
def analytic_cases(verbose=True):
    """三个有闭式解的算例，专门用来抓符号/强度选择错误。"""
    out = {}

    # ---- 对照 A：[0]10，仅压载 Nx ------------------------------------------
    # A = t·Q，ε = A^-1·N = Q^-1·(N/t)  =>  σ_glob = Q·ε = N/t（均匀）
    # 故 σ1 = -Nx·SF/1000/(N·t)，σ2 = 0，τ12 = 0
    n_a, nx_a, sf_a = 10, 500.0, 1.5
    stack_a = symmetric_stack([0.0, 0.0, 0.0, 0.0], n_a)
    res_a = analyse(stack_a, nx_a, 0.0, sf_a)
    s1_exp = -nx_a * sf_a / 1000.0 / (n_a * T_PLY)
    fi_exp = (abs(s1_exp) / XC) ** 2
    out["A"] = {
        "desc": "[0]10 仅压载 Nx",
        "s1_exp": s1_exp, "s1": res_a["worst"]["s1"],
        "s2": res_a["worst"]["s2"], "t12": res_a["worst"]["t12"],
        "fi_exp": fi_exp, "fi": res_a["fi"],
        "rel": (res_a["fi"] - fi_exp) / fi_exp,
    }

    # ---- 对照 B：[90]10，仅压载 Nx（Yc vs Yt 的判别点）--------------------
    n_b, nx_b, sf_b = 10, 500.0, 1.5
    stack_b = symmetric_stack([90.0, 90.0, 90.0, 90.0], n_b)
    res_b = analyse(stack_b, nx_b, 0.0, sf_b)
    s2_exp = -nx_b * sf_b / 1000.0 / (n_b * T_PLY)      # = -0.6 MPa
    fi_yc = (abs(s2_exp) / YC) ** 2
    fi_yt = (abs(s2_exp) / YT) ** 2
    # 历史 bug（漏负号 → 拉伸）：直接翻载荷符号再算一遍
    res_bug = analyse(stack_b, nx_b, 0.0, sf_b, load_sign=+1.0)
    out["B"] = {
        "desc": "[90]10 仅压载 Nx",
        "s2_exp": s2_exp, "s2": res_b["worst"]["s2"],
        "s1": res_b["worst"]["s1"], "t12": res_b["worst"]["t12"],
        "fi_yc_exp": fi_yc, "fi_yc": res_b["fi"],
        "fi_yt_exp": fi_yt, "fi_yt_forced": res_bug["fi"],
        "fi_yt_scalar": tsai_hill_scalar(0.0, s2_exp, 0.0, use="tension"),
        "ratio_exp": (YC / YT) ** 2,
        "ratio": res_bug["fi"] / res_b["fi"],
        "Y_used": res_b["worst"]["Y"],
    }

    # ---- 对照 C：[45]10，仅压载 Nx（检验 σ 变换与 τ12 量级）----------------
    n_c, nx_c, sf_c = 10, 500.0, 1.5
    stack_c = symmetric_stack([45.0, 45.0, 45.0, 45.0], n_c)
    res_c = analyse(stack_c, nx_c, 0.0, sf_c)
    sx = -nx_c * sf_c / 1000.0 / (n_c * T_PLY)          # 全局 σx = -0.6 MPa
    s1_exp = s2_exp_c = 0.5 * sx
    t12_exp = -0.5 * sx
    fi_c_exp = tsai_hill_scalar(s1_exp, s2_exp_c, t12_exp)
    out["C"] = {
        "desc": "[45]10 仅压载 Nx",
        "s1_exp": s1_exp, "s1": res_c["worst"]["s1"],
        "s2_exp": s2_exp_c, "s2": res_c["worst"]["s2"],
        "t12_exp": t12_exp, "t12": res_c["worst"]["t12"],
        "fi_exp": fi_c_exp, "fi": res_c["fi"],
        "rel": (res_c["fi"] - fi_c_exp) / fi_c_exp,
    }

    if verbose:
        print("\n" + "=" * 96)
        print("2) 解析对照（闭式解 vs 脚本）")
        print("=" * 96)
        a = out["A"]
        print(f"  对照 A  {a['desc']}（t = {n_a * T_PLY:.3f} mm）")
        print(f"    手算  σ1 = -Nx·SF/1000/(N·t) = {a['s1_exp']:+.6f} MPa，σ2 = 0，τ12 = 0")
        print(f"    脚本  σ1 = {a['s1']:+.6f} MPa，σ2 = {a['s2']:+.2e}，τ12 = {a['t12']:+.2e} MPa")
        print(f"    手算  FI = (|σ1|/Xc)^2 = {a['fi_exp']:.6e}")
        print(f"    脚本  FI = {a['fi']:.6e}   相对差 {a['rel']:+.2e}")
        print(f"    -> 符号正确：压载下 σ1 < 0，选 Xc=Xt=1500（本例 Xc=Xt 无法区分，"
              f"见对照 B）")

        b = out["B"]
        print(f"\n  对照 B  {b['desc']}（t = {n_b * T_PLY:.3f} mm）—— Yc/Yt 判别点")
        print(f"    手算  σ2 = {b['s2_exp']:+.6f} MPa（压缩），σ1 = 0，τ12 = 0")
        print(f"    脚本  σ1 = {b['s1']:+.2e}，σ2 = {b['s2']:+.6f}，τ12 = {b['t12']:+.2e} MPa")
        print(f"    脚本选用的横向强度 Y = {b['Y_used']:.0f} MPa  "
              f"-> {'Yc ✓' if b['Y_used'] == YC else 'Yt ✗'}")
        print(f"    正确  FI(Yc=246) = (|σ2|/246)^2 = {b['fi_yc_exp']:.6e}   "
              f"脚本 {b['fi_yc']:.6e}")
        print(f"    错误  FI(Yt=40)  = (|σ2|/40)^2  = {b['fi_yt_exp']:.6e}   "
              f"漏负号(拉伸)时脚本 {b['fi_yt_forced']:.6e}   直接按 Yt 算 "
              f"{b['fi_yt_scalar']:.6e}")
        print(f"    比值  FI(Yt)/FI(Yc) = (246/40)^2 = {b['ratio_exp']:.4f}，"
              f"实测 {b['ratio']:.4f}")

        c = out["C"]
        print(f"\n  对照 C  {c['desc']}（附加：检验 σ 变换与 τ12）")
        print(f"    手算  σ1 = σ2 = σx/2 = {c['s1_exp']:+.6f}，τ12 = -σx/2 = {c['t12_exp']:+.6f} MPa")
        print(f"    脚本  σ1 = {c['s1']:+.6f}，σ2 = {c['s2']:+.6f}，τ12 = {c['t12']:+.6f} MPa")
        print(f"    手算  FI = {c['fi_exp']:.6e}   脚本 FI = {c['fi']:.6e}   "
              f"相对差 {c['rel']:+.2e}")

    return out


# =============================================================================
# 7. 主流程
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="独立复核 Tsai-Hill 强度指标 FI")
    ap.add_argument("--all", action="store_true",
                    help="复核 preset_scan.json 中的 backup 与全部 rows")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="把复核结果写成 JSON 文件")
    args = ap.parse_args(argv)

    print("=" * 96)
    print("Tsai-Hill 强度校核独立复核（不依赖 codes/ 的任何实现）")
    print("=" * 96)
    print(f"  材料 : E1={E1} E2={E2} G12={G12} GPa，ν12={V12}，单层 {T_PLY} mm")
    print(f"  许用 : Xt={XT:.0f} Xc={XC:.0f} Yt={YT:.0f} Yc={YC:.0f} S12={S12:.0f} MPa")
    print(f"  数据 : {PRESET_JSON}")

    q_worst = self_check_qbar()
    presets = check_presets(all_rows=args.all)
    analytic = analytic_cases()

    # ---------------- 判定 ----------------
    fails = []
    for r in presets:
        if not (abs(r["rel"]) <= REL_TOL):
            fails.append(f"{r['key']}/{r['tag']} 相对差 {r['rel']:+.2e}")
    for tag in ("A", "C"):
        if abs(analytic[tag]["rel"]) > ANALYTIC_TOL:
            fails.append(f"对照 {tag} 相对差 {analytic[tag]['rel']:+.2e}")
    b = analytic["B"]
    if abs(b["s2"] - b["s2_exp"]) > 1e-12:
        fails.append(f"对照 B σ2 偏差 {b['s2'] - b['s2_exp']:+.2e}")
    if b["Y_used"] != YC:
        fails.append("对照 B 选用了 Yt 而非 Yc")
    b_ok = (abs(b["s2"] - b["s2_exp"]) <= 1e-12) and (b["Y_used"] == YC)

    primary = [r for r in presets if r["tag"] == "primary"]
    worst_rel = max(abs(r["rel"]) for r in primary)

    print("\n" + "=" * 96)
    print("3) 结论")
    print("=" * 96)
    print(f"  三条预设：独立实现的 FI 与文件值最大相对差 {worst_rel:.2e}  "
          f"({'复现一致' if worst_rel <= REL_TOL else '不一致'})")
    if len(presets) > len(primary):
        all_rel = max(abs(r["rel"]) for r in presets)
        n_other = len(presets) - len(primary)
        print(f"  附加复核：backup + 全部 rows 共 {n_other} 条，"
              f"最大相对差 {all_rel:.2e}  "
              f"({'全部复现' if all_rel <= REL_TOL else '有不一致'})")
    print(f"  压缩符号：σ1、σ2 均为负，X/Y 取 Xc/Yc "
          f"({'正确' if b_ok else '有误'})")
    print(f"  [90] 判别：Yc 给出 {b['fi_yc']:.4e}，误用 Yt 给出 {b['fi_yt_forced']:.4e}，"
          f"比值 {b['ratio']:.2f}（理论 37.82）")
    if fails:
        print("  判定：✗ 存在不一致")
        for f in fails:
            print(f"    - {f}")
    else:
        print("  判定：✓ 强度校核可信（单位/角度/层序/σ 变换/强度选择全部复现）")
    print("=" * 96)

    if args.json:
        payload = {
            "qbar_cross_check_max_abs_GPa": q_worst,
            "presets": presets,
            "analytic": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in analytic.items()},
            "ok": not fails,
            "fails": fails,
        }
        pathlib.Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  结果已写入 {args.json}")

    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())

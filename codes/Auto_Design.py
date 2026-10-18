# -*- coding: utf-8 -*-
"""Auto_Design.py —— 面向工程应用的复合材料层合板屈曲自动设计工具。

功能
    输入纵向载荷 Nx、横向载荷 Ny（N/m）与目标安全系数 SF，自动给出最轻的可行铺层：

      1. 目标临界载荷  Ncr_target = max(Nx, Ny) · SF；
      2. 遍历层数 N = 4…20，找出**纯 0° 铺层**下满足要求的最小层数；
      3. 若为双向受压（Nx、Ny 同为正），固定该层数，用 pymoo 遗传算法搜索
         最佳的 4 角度对称铺层 [θ1/θ2/θ3/θ4]s（N > 8 时按 4 角度循环填充半叠层，
         保持"对称 + 4 个设计变量"不变）；角度范围 [-90°, 90°]（含 -45°），
         目标函数带"均衡铺层"罚分——+θ 与 -θ 不成对时重罚（A16/A26 ≠ 0）；
      4. 用 Tsai-Hill 准则做**面内强度校核**（T300/5208 典型许用值），
         与屈曲共同构成"双重校核"：FI < 1 才算通过；
      5. 终端打印可直接写进工程文件的推荐方案。

物理模型
    与 Angle_Optimize 同一套求解器：经典层合板理论（CLT，composipy 校核）+ 双正弦级数
    Rayleigh–Ritz 屈曲解，完整保留 D16/D26 弯扭耦合，四边简支（SS1）。

判定准则
    对载荷对 (Nx, Ny) 求临界倍数 λ：λ·(Nx, Ny) 使板屈曲。
    等价地，主方向的临界载荷 Ncr_eq = λ · max(Nx, Ny)，要求 Ncr_eq ≥ Ncr_target，
    即 λ ≥ SF。当 Nx = Ny 时 Ncr_eq 就是双向临界载荷本身，与常规写法一致。

用法
    python Auto_Design.py                 # 交互式输入（回车采用默认值）
    python Auto_Design.py 500 500 1.5     # 命令行直接给定，便于脚本化与复现

依赖
    同目录下的 angle_optimize.py（提供 qbar / BucklingSolver 等共用实现）。
"""

import os
import sys
import time

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import minimize as scipy_minimize

from angle_optimize import (A_PLATE, B_PLATE, BALANCE_WEIGHT, DENSITY, T_PLY,
                            BucklingSolver, balance_penalty, balance_residual,
                            M_VERIFY, qbar)

# =============================================================================
# 1. 设计参数
# =============================================================================
N_MIN, N_MAX = 4, 20          # 层数扫描范围
M_GA = 10                     # GA 阶段级数项数（求快）
GA_POP, GA_GEN = 80, 60       # GA 规模
SEED = 1
DEFAULT_NX, DEFAULT_NY, DEFAULT_SF = 500.0, 500.0, 1.5

_SOLVER_GA = BucklingSolver(A_PLATE, B_PLATE, M_GA)
_SOLVER_CHK = BucklingSolver(A_PLATE, B_PLATE, M_VERIFY)


# =============================================================================
# 2. 铺层构造与刚度
# =============================================================================
def half_stack(theta4, n_plies):
    """把 4 个设计角度循环填充成半叠层（长度 N/2）。

    N = 4  -> [θ1, θ2]                    （即 [θ1/θ2]s）
    N = 8  -> [θ1, θ2, θ3, θ4]            （即 [θ1/θ2/θ3/θ4]s）
    N = 12 -> [θ1, θ2, θ3, θ4, θ1, θ2]
    N 更大时同理循环，保证始终只有 4 个独立变量。
    """
    return [float(theta4[i % 4]) for i in range(n_plies // 2)]


def symmetric_stack(theta4, n_plies):
    """由半叠层镜像得到对称铺层（共 N 层）。"""
    half = half_stack(theta4, n_plies)
    return half + half[::-1]


def d_of_angles(angles):
    """任意角度序列的 D 矩阵（GPa·mm³）。"""
    n = len(angles)
    z = np.linspace(-n * T_PLY / 2.0, n * T_PLY / 2.0, n + 1)
    D = np.zeros((3, 3))
    for theta, w in zip(angles, np.diff(z ** 3) / 3.0):
        D += qbar(theta) * w
    return D


def ncr_equiv(theta4, n_plies, nx, ny, solver=None):
    """给定铺层与载荷对，返回 (主方向临界载荷 N/m, 临界倍数 λ)。

    solver.ncr(D, nx, ny) 返回 λ：λ·(Nx, Ny) 恰好使板屈曲。
    主方向临界载荷 Ncr_eq = λ · max(Nx, Ny)。
    """
    solver = solver or _SOLVER_GA
    D = d_of_angles(symmetric_stack(theta4, n_plies))
    lam = solver.ncr(D, nx, ny)
    return lam * max(nx, ny), lam


def stacking_text(theta4, n_plies):
    """铺层字符串，如 [45.2/90.0/0.0/0.0/45.2/90.0]s。"""
    return "[" + "/".join(f"{a:.1f}" for a in half_stack(theta4, n_plies)) + "]s"


def areal_weight(n_plies):
    """面密度 kg/m²。"""
    return n_plies * T_PLY * 1e-3 * DENSITY


# =============================================================================
# 2b. 强度校核：Tsai-Hill 准则（T300/5208 典型许用值，MPa）
# =============================================================================
XT, XC = 1500.0, 1500.0     # 纵向拉伸 / 压缩强度
YT, YC = 40.0, 246.0        # 横向拉伸 / 压缩强度
S12 = 68.0                  # 面内剪切强度


def strength_check(stack_angles, nx, ny, sf):
    """层合板平面应力下的 Tsai-Hill 强度校核。

    步骤
        1. 由 A 矩阵反解面内应变：ε = A⁻¹·N（A 单位 GPa·mm = 1000 N/mm，
           载荷取 SF 放大的设计值 N/mm）；
        2. 逐层用 Qbar(θ) 求全局应力 (σx, σy, τxy)，再转到材料主轴 (σ1, σ2, τ12)；
        3. Tsai-Hill 失效因子
               FI = σ1²/X² − σ1σ2/X² + σ2²/Y² + τ12²/S²
           （σ1 > 0 用 Xt 否则 Xc；σ2 > 0 用 Yt 否则 Yc）；
        4. 返回全板最大 FI 及其所在铺层。
    """
    A = sum(qbar(t) for t in stack_angles) * T_PLY        # GPa·mm
    A_n = A * 1000.0                                      # N/mm
    N = np.array([nx * sf, ny * sf, 0.0]) / 1000.0        # N/m → N/mm
    eps = np.linalg.solve(A_n, N)                         # 面内应变 (εx, εy, γxy)

    worst = {"fi": 0.0, "ply": 0, "theta": 0.0,
             "s1": 0.0, "s2": 0.0, "t12": 0.0}
    for k, theta in enumerate(stack_angles, start=1):
        sig = qbar(theta) @ eps * 1000.0                  # GPa → MPa
        sx, sy, txy = sig[0], sig[1], sig[2]
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        s1 = sx * c * c + sy * s * s + 2.0 * txy * s * c
        s2 = sx * s * s + sy * c * c - 2.0 * txy * s * c
        t12 = -(sx - sy) * s * c + txy * (c * c - s * s)
        X = XT if s1 >= 0 else XC
        Y = YT if s2 >= 0 else YC
        fi = (s1 ** 2 / X ** 2 - s1 * s2 / X ** 2
              + s2 ** 2 / Y ** 2 + t12 ** 2 / S12 ** 2)
        if fi > worst["fi"]:
            worst = {"fi": fi, "ply": k, "theta": theta,
                     "s1": s1, "s2": s2, "t12": t12}
    worst["eps"] = eps
    return worst


# =============================================================================
# 3. 输入
# =============================================================================
def _read_float(prompt, default):
    """交互读取一个正数；空输入或非法输入时用默认值。"""
    try:
        raw = input(prompt).strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        print(f"   输入无法解析为数值，采用默认值 {default}")
        return default
    return val


def read_inputs(argv):
    """命令行参数优先，否则交互输入。"""
    if len(argv) >= 4:
        try:
            nx, ny, sf = float(argv[1]), float(argv[2]), float(argv[3])
        except ValueError:
            raise SystemExit("命令行参数须为数字：Auto_Design.py Nx Ny SF")
        return nx, ny, sf, "命令行参数"

    print("请输入设计条件（直接回车采用方括号中的默认值）：")
    nx = _read_float(f"  纵向载荷 Nx (N/m)  [{DEFAULT_NX:g}]: ", DEFAULT_NX)
    ny = _read_float(f"  横向载荷 Ny (N/m)  [{DEFAULT_NY:g}]: ", DEFAULT_NY)
    sf = _read_float(f"  目标安全系数 SF    [{DEFAULT_SF:g}]: ", DEFAULT_SF)
    return nx, ny, sf, "交互输入"


# =============================================================================
# 4. 纯 [0]N 层数扫描
# =============================================================================
def scan_pure_zero(nx, ny, target):
    """扫描 N = 4…20 的纯 [0]N 铺层，返回结果列表。"""
    rows = []
    for n in range(N_MIN, N_MAX + 1):
        D = d_of_angles([0.0] * n)
        lam = _SOLVER_CHK.ncr(D, nx, ny)
        ncr = lam * max(nx, ny)
        rows.append({"N": n, "t": n * T_PLY, "ncr": ncr, "lam": lam,
                     "ok": ncr >= target})
    return rows


# =============================================================================
# 5. 角度优化（GA + 局部精修）
# =============================================================================
class LayupProblem(Problem):
    """最大化 Ncr（pymoo 最小化 -Ncr），约束 Ncr ≥ target。"""

    def __init__(self, n_plies, nx, ny, target):
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=1,
                         xl=np.full(4, -90.0), xu=np.full(4, 90.0))
        self.n_plies, self.nx, self.ny, self.target = n_plies, nx, ny, target
        self.cache = {}

    def evaluate_one(self, theta):
        """返回 (真实 Ncr, 均衡加权得分)，带缓存。"""
        key = tuple(np.round(np.asarray(theta, float), 6))
        if key not in self.cache:
            ncr = ncr_equiv(list(key), self.n_plies, self.nx, self.ny, _SOLVER_GA)[0]
            # 均衡性罚分：半叠层中 +θ 与 -θ 不成对则重罚（A16/A26 ≠ 0）
            r = balance_penalty(half_stack(key, self.n_plies))
            self.cache[key] = (ncr, ncr * max(1.0 - BALANCE_WEIGHT * r, 0.05))
        return self.cache[key]

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 1))
        G = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            ncr, score = self.evaluate_one(x)
            F[i, 0] = -score
            G[i, 0] = (self.target - ncr) / max(self.target, 1e-12)
        out["F"] = F
        out["G"] = G


def optimize_layup(n_plies, nx, ny, target):
    """GA 全局搜索 + SLSQP/Nelder–Mead 精修，返回 (theta4, Ncr, 用时)。"""
    problem = LayupProblem(n_plies, nx, ny, target)
    algo = NSGA2(pop_size=GA_POP,
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20, prob=1.0 / 4),
                 eliminate_duplicates=True)
    t0 = time.perf_counter()
    res = pymoo_minimize(problem, algo, ("n_gen", GA_GEN), seed=SEED,
                         verbose=False)
    dt = time.perf_counter() - t0

    pop_x = np.atleast_2d(res.pop.get("X"))
    pop_f = np.atleast_1d(res.pop.get("F")).ravel()
    best = pop_x[int(np.argmax(-pop_f))]
    best_ncr = float(problem.evaluate_one(best)[0])

    # 局部精修（统一在 M_VERIFY 精度下进行，目标同样带均衡性罚分）
    def neg(theta):
        t = np.clip(theta, -90.0, 90.0)
        ncr = ncr_equiv(t, n_plies, nx, ny, _SOLVER_CHK)[0]
        r = balance_penalty(half_stack(t, n_plies))
        return -(ncr * max(1.0 - BALANCE_WEIGHT * r, 0.05))

    x0 = np.clip(best, -90.0, 90.0)
    v0 = -neg(x0)
    for method, opts in (("SLSQP", {"maxiter": 200, "ftol": 1e-8}),
                         ("Nelder-Mead", {"maxiter": 3000,
                                          "xatol": 1e-4, "fatol": 1e-4})):
        r = scipy_minimize(neg, x0, method=method,
                           bounds=[(-90.0, 90.0)] * 4 if method == "SLSQP" else None,
                           options=opts)
        cand = np.clip(r.x, -90.0, 90.0)
        v = -neg(cand)
        if v > v0:
            x0, v0 = cand, v

    return x0, v0, dt


# =============================================================================
# 6. 主流程
# =============================================================================
def main():
    nx, ny, sf, how = read_inputs(sys.argv)

    if nx < 0 or ny < 0:
        raise SystemExit("载荷不能为负。")
    if nx == 0 and ny == 0:
        raise SystemExit("Nx 与 Ny 不能同时为 0。")
    if sf <= 0:
        raise SystemExit("安全系数必须为正数。")

    target = max(nx, ny) * sf
    biaxial = (nx > 0 and ny > 0)

    line = "=" * 74
    print("\n" + line)
    print(" Auto_Design  ·  复合材料层合板屈曲自动设计")
    print(line)
    print(f" 输入来源 : {how}")
    print(f" 载荷     : Nx = {nx:g} N/m，Ny = {ny:g} N/m"
          f"（载荷比 Ny/Nx = {ny / nx if nx else float('inf'):.2f}）")
    print(f" 安全系数 : SF = {sf:g}")
    print(f" 目标载荷 : Ncr_target = max(Nx, Ny)·SF = {target:.1f} N/m")
    print(f" 工况     : {'双向受压 → 需要角度优化' if biaxial else '单向受压 → 纯 0° 即最优'}")
    print(f" 结构     : a = b = {A_PLATE:.2f} m 四边简支；单层 {T_PLY} mm，"
          f"ρ = {DENSITY:.0f} kg/m³")

    # ---------------- ① 纯 0° 层数扫描 ----------------
    print("\n" + "-" * 74)
    print("① 纯 [0]N 层数扫描（N = 4…20）")
    print("-" * 74)
    print(f"  {'N':>3s} {'t(mm)':>7s} {'Ncr(N/m)':>10s} {'λ':>6s} {'裕度':>7s}  判定")
    rows = scan_pure_zero(nx, ny, target)
    first_ok = None
    for r in rows:
        if r["ok"] and first_ok is None:
            first_ok = r
        mark = "✓" if r["ok"] else "✗"
        print(f"  {r['N']:3d} {r['t']:7.3f} {r['ncr']:10.1f} {r['lam']:6.2f} "
              f"{r['ncr'] / target:7.2f}  {mark}")

    if first_ok is None:
        print(f"\n  纯 0° 铺层在 N ≤ {N_MAX} 内无法满足 {target:.0f} N/m，"
              f"需增加层数或降低载荷/安全系数。")
        return

    n_min = first_ok["N"]
    print(f"\n  → 最小纯 0° 层数 N = {n_min}（{n_min * T_PLY:.3f} mm），"
          f"Ncr = {first_ok['ncr']:.1f} N/m，λ = {first_ok['lam']:.2f}")

    # ---------------- ② 双向受压：角度优化 ----------------
    if biaxial:
        n_design = n_min if n_min % 2 == 0 else n_min + 1
        if n_design != n_min:
            print(f"  → 对称铺层要求偶数层，取 N = {n_design}（{n_design * T_PLY:.3f} mm）")

        # 该层数下纯 0° 的基准值
        base = next(r for r in rows if r["N"] == n_design)

        print("\n" + "-" * 74)
        print(f"② 角度优化（N = {n_design} 固定，4 角度对称铺层，GA {GA_POP}×{GA_GEN}）")
        print("-" * 74)
        theta, ncr_opt, dt = optimize_layup(n_design, nx, ny, target)
        ncr_opt, lam_opt = ncr_equiv(theta, n_design, nx, ny, _SOLVER_CHK)

        print(f"  GA 用时 {dt:.1f} s")
        print(f"  最优角度 : θ = [{theta[0]:.2f}, {theta[1]:.2f}, "
              f"{theta[2]:.2f}, {theta[3]:.2f}]°")
        print(f"  铺层     : {stacking_text(theta, n_design)}")
        s16, s26 = balance_residual(half_stack(theta, n_design))
        print(f"  均衡性   : ΣQ̄16 = {s16:+.3f}，ΣQ̄26 = {s26:+.3f} GPa"
              f"  -> {'均衡 ✓（A16 = A26 = 0）' if balance_penalty(half_stack(theta, n_design)) < 0.02 else '不均衡 ✗'}")
        print(f"  屈曲载荷 : {ncr_opt:.1f} N/m（λ = {lam_opt:.2f}）")
        print(f"  相对同层数纯 0°（{base['ncr']:.1f} N/m）："
              f"{(ncr_opt / base['ncr'] - 1) * 100:+.1f}%")

        ncr_final = ncr_opt
        lam_final = lam_opt
        layup = stacking_text(theta, n_design)

        # 减薄潜力：用同一角度族在更薄的偶数层数上复算
        thinner = None
        for n in range(N_MIN, n_design, 2):
            v, _ = ncr_equiv(theta, n, nx, ny, _SOLVER_CHK)
            if v >= target:
                thinner = (n, v)
                break

        # 是否满足
        ok_final = ncr_final >= target
        verdict = "满足" if ok_final else "未满足"
        print("\n" + "-" * 74)
        print("③ 方案校验")
        print("-" * 74)
        print(f"  临界倍数 λ = {lam_final:.2f}  {'≥' if ok_final else '<'} SF = {sf:.2f}"
              f"  → {verdict}")
        print(f"  厚度 t = {n_design * T_PLY:.3f} mm，面密度 {areal_weight(n_design):.2f} kg/m²")
        if thinner and n_design > N_MIN:
            print(f"\n  工程提示：若允许以角度优化替代纯 0° 基准来定层数，"
                  f"N = {thinner[0]} 层（{thinner[0] * T_PLY:.3f} mm）")
            print(f"            采用同一角度族即可满足（Ncr ≈ {thinner[1]:.0f} N/m），"
                  f"仍有减薄空间；本工具按需求以纯 0° 基准确定层数。")
    else:
        n_design = n_min
        ncr_final = first_ok["ncr"]
        lam_final = first_ok["lam"]
        layup = f"[0]_{n_min}"
        theta = [0.0, 0.0, 0.0, 0.0]
        print("\n  单向受压时纯 0° 铺层本身即为最优（D11 最大），无需角度优化。")

    # ---------------- ④ 强度校核（Tsai-Hill）----------------
    stack_final = (symmetric_stack(theta, n_design) if biaxial
                   else [0.0] * n_design)
    st = strength_check(stack_final, nx, ny, sf)
    safe_strength = st["fi"] < 1.0

    print("\n" + "-" * 74)
    print("④ 强度校核（Tsai-Hill，设计载荷已乘安全系数）")
    print("-" * 74)
    print(f"  设计载荷 : Nx = {nx * sf:.0f} N/m，Ny = {ny * sf:.0f} N/m"
          f"（工作载荷 × SF = {sf:g}）")
    print(f"  面内应变 : εx = {st['eps'][0]:+.3e}，εy = {st['eps'][1]:+.3e}，"
          f"γxy = {st['eps'][2]:+.3e}")
    print(f"  最危险层 : 第 {st['ply']} 层（θ = {st['theta']:.1f}°）")
    print(f"             σ1 = {st['s1']:+8.3f} MPa，σ2 = {st['s2']:+8.3f} MPa，"
          f"τ12 = {st['t12']:+7.3f} MPa")
    print(f"  Tsai-Hill: FI = {st['fi']:.3e}"
          f"（许用值 Xt={XT:.0f}、Xc={XC:.0f}、Yt={YT:.0f}、Yc={YC:.0f}、S={S12:.0f} MPa）")
    print(f"  强度裕度 : 1/FI = {1.0 / st['fi']:.2e}（FI < 1 即安全）")

    # ---------------- ⑤ 推荐方案 ----------------
    print("\n" + "=" * 74)
    print(" 推荐方案")
    print("=" * 74)
    print(f" 基于给定的载荷与安全系数，推荐采用 {n_design} 层 {layup} 铺层，")
    print(f" 实际屈曲载荷为 {ncr_final:.0f} N/m，总厚度为 {n_design * T_PLY:.2f} mm。")
    print()
    print(f" （Ncr_target = {target:.0f} N/m；实际 λ = {lam_final:.2f} ≥ SF = {sf:.2f}；"
          f"面密度 {areal_weight(n_design):.2f} kg/m²；Tsai-Hill FI = {st['fi']:.2e}）")
    print()
    if safe_strength:
        print(" 【安全】屈曲与强度双重校核通过")
    else:
        print(" 【警告】发生 Tsai-Hill 强度失效，必须增加厚度")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()

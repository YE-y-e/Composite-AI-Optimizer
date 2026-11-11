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

import contextlib
import csv
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
    """由半叠层镜像得到对称铺层（共 N 层）。

    对称构造要求层数为偶数：奇数层必须有中心层（mid-plane ply）才谈得上"对称"，
    而本版的刚度/厚度/质量都按 "整层数 × 单层厚" 计算。历史实现里
    `symmetric_stack(..., 9)` 会静默返回 8 层，而厚度仍按 9 层计，
    导致"刚度属于 8 层、重量属于 9 层"的错配。这里直接拒绝奇数层。
    """
    if int(n_plies) % 2:
        raise DesignInputError(
            f"对称层合板要求偶数层，收到 {n_plies} 层。"
            f"如需奇数层（含中心层），须同时实现中心层铺层、刚度、厚度与质量的定义。")
    half = half_stack(theta4, n_plies)
    return half + half[::-1]


class DesignInputError(ValueError):
    """设计参数不合法。

    单独定义异常类型，让界面/CLI 能给出领域内的说明，
    而不是把 `LinAlgError: leading minor ...` 之类的底层线代异常抛给用户。
    """


def validate_design(theta4, n_plies, nx, ny, sf):
    """校验一次设计评价的全部输入，返回规范化后的四元组。

    校验项（与界面、CLI、脚本共用同一套）：
    - 层数：整数、偶数、落在配置的扫描范围 [N_MIN, N_MAX] 内；
    - 设计角：4 个、有限、落在 −90°…90°；
    - 面内载荷：有限、非负、且不全为零（零载荷时临界载荷无定义）；
    - 安全系数：有限且为正。
    """
    if isinstance(n_plies, bool) or not isinstance(n_plies, (int, np.integer)):
        raise DesignInputError(f"层数必须是整数，收到 {n_plies!r}")
    n = int(n_plies)
    if n % 2:
        raise DesignInputError(
            f"本版只支持对称偶数组铺层，层数必须为偶数，收到 {n}。")
    if not (N_MIN <= n <= N_MAX):
        raise DesignInputError(
            f"层数必须落在扫描范围 {N_MIN}–{N_MAX} 内，收到 {n}。")

    t = list(theta4)
    if len(t) != 4:
        raise DesignInputError(f"需要 4 个设计角，收到 {len(t)} 个。")
    t_arr = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t_arr)):
        raise DesignInputError(f"设计角必须是有限值，收到 {t}。")
    if np.any(np.abs(t_arr) > 90.0 + 1e-9):
        raise DesignInputError(f"设计角必须落在 −90°…90°，收到 {t}。")

    nx = float(nx)
    ny = float(ny)
    if not (np.isfinite(nx) and np.isfinite(ny)):
        raise DesignInputError(f"载荷必须是有限值，收到 Nx={nx}、Ny={ny}。")
    if nx < 0 or ny < 0:
        raise DesignInputError(f"载荷必须非负（本版按压缩工况定义），收到 Nx={nx}、Ny={ny}。")
    if max(nx, ny) <= 0.0:
        raise DesignInputError(
            "载荷为零：此时临界载荷没有定义，无法给出屈曲结论；"
            "请给出非零的面内压缩载荷。")

    sf = float(sf)
    if not np.isfinite(sf) or sf <= 0:
        raise DesignInputError(f"安全系数必须是有限正数，收到 {sf}。")

    return n, [float(v) for v in t_arr], nx, ny, sf


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
    n_plies, theta4, nx, ny, sf = validate_design(theta4, n_plies, nx, ny, 1.0)
    solver = solver or _SOLVER_GA
    D = d_of_angles(symmetric_stack(theta4, n_plies))
    lam = solver.ncr(D, nx, ny)
    _EVAL_COUNT["n"] += 1
    return lam * max(nx, ny), lam


def stacking_text(theta4, n_plies):
    """铺层字符串，如 [45.2/90.0/0.0/0.0/45.2/90.0]s。"""
    def _fmt(a):
        a = 0.0 if abs(a) < 5e-2 else float(a)
        return f"{a:.1f}"
    return "[" + "/".join(_fmt(a) for a in half_stack(theta4, n_plies)) + "]s"


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

    符号约定：外部输入 Nx、Ny 为**压缩载荷的大小**（正值），与屈曲求解器一致；
    但 CLT 的 A⁻¹·N 用的是"拉为正"的内力，所以这里必须先把压缩转成负号，
    否则会算出拉伸应力、进而错选 Yt（40 MPa）而不是 Yc（246 MPa）。

    步骤
        1. 由 A 矩阵反解面内应变：ε = A⁻¹·N（A 单位 GPa·mm = 1000 N/mm，
           载荷取 SF 放大的设计值 N/mm，压缩取负）；
        2. 逐层用 Qbar(θ) 求全局应力 (σx, σy, τxy)，再转到材料主轴 (σ1, σ2, τ12)；
        3. Tsai-Hill 失效因子
               FI = σ1²/X² − σ1σ2/X² + σ2²/Y² + τ12²/S²
           （σ1 > 0 用 Xt 否则 Xc；σ2 > 0 用 Yt 否则 Yc）；
        4. 返回全板最大 FI 及其所在铺层。
    """
    values = np.asarray([nx, ny, sf], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("载荷与安全系数必须是有限实数。")
    if nx < 0 or ny < 0:
        raise ValueError("压缩载荷以正数输入，不接受负值。")
    if sf <= 0:
        raise ValueError("安全系数必须为正数。")
    A = sum(qbar(t) for t in stack_angles) * T_PLY        # GPa·mm
    A_n = A * 1000.0                                      # N/mm
    # 外部"压缩为正" → 内部"拉为正"：取负号
    N = -np.array([nx, ny, 0.0]) * sf / 1000.0            # N/m → N/mm
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


def optimize_layup(n_plies, nx, ny, target, seed=None):
    """GA 全局搜索 + SLSQP/Nelder–Mead 精修，返回 (theta4, Ncr, 用时)。

    seed 缺省时用模块常量 SEED（保持历史结果可复现）；显式传入则真正用于
    pymoo 的随机数发生器——用于验证"换 seed 结果会不会变"和多 seed 稳定性。
    """
    problem = LayupProblem(n_plies, nx, ny, target)
    algo = NSGA2(pop_size=GA_POP,
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20, prob=1.0 / 4),
                 eliminate_duplicates=True)
    ga_seed = SEED if seed is None else int(seed)
    t0 = time.perf_counter()
    res = pymoo_minimize(problem, algo, ("n_gen", GA_GEN), seed=ga_seed,
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
# 5b. 逐层数优化与最轻候选选择（国赛主流程）
# =============================================================================
# 设计域（本版明确写死的范围，界面与报告须一致引用）
N_ALLOWED = tuple(n for n in range(N_MIN, N_MAX + 1) if n % 2 == 0)
ANGLE_GRID = (-90.0, -45.0, 0.0, 45.0, 90.0)   # 可制造离散角度，仅用于取整与对照
BAL_TOL = 1.0                                   # 均衡残差阈值（GPa，约 1% Q11）
MIN_MARGIN = 1.2                               # 可行性要求的 λ/SF 下限
GRID_N_MAX = 12                              # 层数不超过该值时用离散枚举求全局最优
GRID_TOPK = 5                                # 粗筛后按 M=10 取前几名，再用 M=16 复核
MARGIN_FACTOR = 1.5                             # 备选方案要求的 λ/SF 下限

_EVAL_COUNT = {"n": 0}


def reset_eval_count():
    """清零物理评价计数（用于预算统计与同预算对照）。"""
    _EVAL_COUNT["n"] = 0


def eval_count():
    """返回自上次清零以来调用屈曲求解器的次数（由 ncr_equiv 记账）。"""
    return _EVAL_COUNT["n"]


@contextlib.contextmanager
def quick_ga(enabled=True):
    """临时降低 GA 规模与校核阶次（快速模式）。退出时严格还原。"""
    if not enabled:
        yield
        return
    global GA_POP, GA_GEN, _SOLVER_CHK
    old = (GA_POP, GA_GEN, _SOLVER_CHK)
    try:
        GA_POP, GA_GEN = 40, 30
        _SOLVER_CHK = BucklingSolver(A_PLATE, B_PLATE, 10)
        yield
    finally:
        GA_POP, GA_GEN, _SOLVER_CHK = old


def snap_theta4(theta4, grid=None):
    """把连续角度取整到最近的可制造角度。"""
    grid = grid or ANGLE_GRID
    return [float(min(grid, key=lambda g: abs(g - float(a))))
            for a in np.asarray(theta4, float)]


def check_rules(theta4, n_plies):
    """检查本版实现的两条制造规则：对称（构造保证）与面内均衡。"""
    half = half_stack(theta4, n_plies)
    stack = symmetric_stack(theta4, n_plies)
    s16, s26 = balance_residual(half)
    symmetric = all(abs(stack[i] - stack[-1 - i]) < 1e-9
                    for i in range(len(stack)))
    return {
        "symmetric": bool(symmetric),
        "balanced": bool(abs(s16) <= BAL_TOL and abs(s26) <= BAL_TOL),
        "s16": float(s16), "s26": float(s26),
    }


def evaluate_candidate(theta4, n_plies, nx, ny, sf, solver=None):
    """对一个候选（给定层数与 4 个设计角）做完整评价。

    可行 = λ ≥ MIN_MARGIN·SF（默认 1.2）且逐层最大 FI < 1，
    且满足对称与均衡两条规则。
    """
    n_plies, theta4, nx, ny, sf = validate_design(theta4, n_plies, nx, ny, sf)
    solver = solver or _SOLVER_CHK
    stack = symmetric_stack(theta4, n_plies)
    lam = solver.ncr(d_of_angles(stack), nx, ny)
    _EVAL_COUNT["n"] += 1
    ncr = lam * max(nx, ny)
    st = strength_check(stack, nx, ny, sf)
    rules = check_rules(theta4, n_plies)
    feasible = bool(lam >= MIN_MARGIN * sf and st["fi"] < 1.0
                    and rules["symmetric"] and rules["balanced"])
    return {
        "n_plies": int(n_plies),
        "theta4": theta4,
        "layup": stacking_text(theta4, n_plies),
        "stack": [float(a) for a in stack],
        "thickness": float(n_plies * T_PLY),
        "areal": float(areal_weight(n_plies)),
        "lam": float(lam), "ncr": float(ncr),
        "fi": float(st["fi"]), "fi_ply": int(st["ply"]),
        "fi_theta": float(st["theta"]),
        "rules": rules, "feasible": feasible,
    }


def optimize_at_n(n_plies, nx, ny, sf, seed=None, method="ga"):
    """固定层数搜索角度，返回该层数的最优候选（含评价字段）。

    method="ga"     : NSGA-II + SLSQP/Nelder–Mead（默认，主流程）
    method="random" : 同评价预算的随机搜索（基线对照）
    method="grid"   : 离散角度网格枚举（4 角 × 5 网格 = 625 组，最优性证据）
    """
    target = max(nx, ny) * sf
    seed = SEED if seed is None else int(seed)
    grid_checked = None
    grid_evaluated = None
    t0 = time.perf_counter()

    if method == "auto":
        method = "grid" if int(n_plies) <= GRID_N_MAX else "ga"

    if method == "ga":
        theta, _score, _dt = optimize_layup(n_plies, nx, ny, target, seed=seed)
    elif method == "random":
        rng = np.random.default_rng(seed)
        best, best_score = None, -np.inf
        for _ in range(GA_POP * GA_GEN):          # 与 GA 同量级的评价预算
            cand = rng.uniform(-90.0, 90.0, 4)
            lam = ncr_equiv(cand, n_plies, nx, ny, _SOLVER_GA)[1]
            pen = balance_penalty(half_stack(cand, n_plies))
            score = lam * max(1.0 - BALANCE_WEIGHT * pen, 0.05)
            if score > best_score:
                best, best_score = cand, score
        theta = np.asarray(best, float)
    elif method in ("grid", "auto"):
        # 离散枚举：先用 M=10 求解器把 5^4 组全部粗筛，再对靠前的候选做完整评价
        # （λ + Tsai-Hill FI + 制造规则），只在"可行"里取最大 λ。
        # 注意：这只保证"复核集合内的最优"，不等于整个角度设计域的全局最优。
        scored = []
        for a in ANGLE_GRID:
            for b in ANGLE_GRID:
                for c in ANGLE_GRID:
                    for d in ANGLE_GRID:
                        cand = [a, b, c, d]
                        rules = check_rules(cand, n_plies)
                        if not (rules["symmetric"] and rules["balanced"]):
                            continue
                        lam = ncr_equiv(cand, n_plies, nx, ny, _SOLVER_GA)[1]
                        scored.append((lam, cand))
        if not scored:
            raise RuntimeError("离散角度网格内没有满足均衡约束的候选")
        scored.sort(key=lambda t: -t[0])

        # 逐级扩大复核范围：5 → 25 → 125 → 全部（每轮只评价新增区间，已评价的走缓存）
        topk = max(1, int(GRID_TOPK))
        eval_cache = {}

        def _ev(cand):
            key = tuple(float(x) for x in cand)
            if key not in eval_cache:
                eval_cache[key] = evaluate_candidate(cand, n_plies, nx, ny, sf)
            return eval_cache[key]

        best_ev = None
        while True:
            for _lam, cand in scored[:topk]:
                ev = _ev(cand)
                if not ev["feasible"]:
                    continue
                if best_ev is None or ev["ncr"] > best_ev["ncr"]:
                    best_ev = ev
            if best_ev is not None or topk >= len(scored):
                break
            topk = min(len(scored), max(topk * 5, topk + 20))

        if best_ev is None:
            # 复核范围内没有可行解：退回"最好的那个不可行候选"，由调用方
            # （select_design）按 feasible 决定跳过还是继续，而不是在这里抛错中断整个层数扫描。
            best_fallback = None
            for _lam, cand in scored[:topk]:
                ev = _ev(cand)
                if best_fallback is None or ev["ncr"] > best_fallback["ncr"]:
                    best_fallback = ev
            if best_fallback is None:
                raise RuntimeError(
                    f"离散角度网格内没有可评价的候选（层数 {n_plies}，"
                    f"载荷 {nx:.0f}/{ny:.0f} N/m，SF {sf:g}）"
                )
            best_ev = best_fallback
            best_ev["grid_no_feasible"] = True
        theta = np.asarray(best_ev["theta4"], float)
        grid_checked = topk
        grid_evaluated = len(eval_cache)
    else:
        raise ValueError(f"未知的优化方法：{method}")

    # 网格路径已对最终候选做过完整评价，直接复用，避免再算一次（也保住 grid_no_feasible 等状态字段）
    if method in ("grid", "auto") and grid_checked is not None:
        cand = dict(best_ev)
    else:
        cand = evaluate_candidate(theta, n_plies, nx, ny, sf)
    cand["method"] = method
    cand["seed"] = seed
    cand["seconds"] = round(time.perf_counter() - t0, 2)
    if grid_checked is not None:
        cand["grid_checked"] = grid_checked
        cand["grid_evaluated"] = grid_evaluated
    return cand


def select_design(nx, ny, sf, fast=True, methods=("auto",), n_max=None,
                  margin_factor=MARGIN_FACTOR, seed=None):
    """主流程：层数从小到大逐个优化角度，取最小面密度的可行方案。

    返回
        rows     每个层数一行（含不可行），供"层数—承载"曲线
        primary  最轻可行候选
        backup   主推荐之上、满足更大裕度（λ ≥ margin·SF）的备选
        status   ok / infeasible
    """
    nx, ny, sf = float(nx), float(ny), float(sf)
    if nx == 0 and ny == 0:
        raise ValueError("Nx 与 Ny 不能同时为 0。")
    if not methods:
        raise DesignInputError(
            "methods 不能为空：至少指定一种寻优方法（auto / grid / ga / random）。")
    top = n_max or N_MAX
    if not (N_MIN <= int(top) <= N_MAX):
        raise DesignInputError(
            f"n_max 必须落在 {N_MIN}–{N_MAX} 内，收到 {top}。")
    top = int(top)
    rows, primary, backup = [], None, None

    def rank(cand):
        """候选排序键：先看是否可行，再看承载。

        只比 lam 会出事——一个 lam 更高但违反均衡/强度/裕度的候选，
        会把同层数里真正可用的候选挤掉，最终 primary 变成不可行方案。
        """
        return (1 if cand["feasible"] else 0, cand["lam"])

    with quick_ga(fast):
        for n in N_ALLOWED:
            if n > top:
                break
            best = None
            for method in methods:
                cand = optimize_at_n(n, nx, ny, sf, seed=seed, method=method)
                if best is None or rank(cand) > rank(best):
                    best = cand
            rows.append({
                "n_plies": n, "areal": best["areal"],
                "lam": best["lam"], "ncr": best["ncr"],
                "fi": best["fi"], "feasible": best["feasible"],
                "theta4": best["theta4"], "layup": best["layup"],
                "method": best["method"], "seconds": best["seconds"],
            })
            if primary is not None and best["feasible"] \
                    and best["lam"] >= margin_factor * sf:
                backup = best
                break
            if primary is None and best["feasible"]:
                primary = best

    return {
        "status": "ok" if primary is not None else "infeasible",
        "nx": nx, "ny": ny, "sf": sf, "target": max(nx, ny) * sf,
        "fast": bool(fast), "rows": rows,
        "primary": primary, "backup": backup,
        "evals": eval_count(),
        "domain": {"n_allowed": list(N_ALLOWED),
                   "angle_grid": list(ANGLE_GRID),
                   "balance_tol": BAL_TOL, "min_margin": MIN_MARGIN,
                   "margin_factor": margin_factor},
    }


def append_run_row(path, row):
    """把一次运行记录追加到 runs.csv（首行写表头）。"""
    fields = ["time", "case_id", "Nx", "Ny", "SF", "method", "seed",
              "n_plies", "areal", "lam", "ncr", "fi", "feasible",
              "evals", "seconds", "fast"]
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})

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

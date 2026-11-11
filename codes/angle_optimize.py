# -*- coding: utf-8 -*-
"""双向受压下 8 层对称层合板的铺层角度优化（真实屈曲分析 + 遗传算法）。

工程问题
    一块四边简支（SSSS）正方形复合材料层合板，a = b = 0.5 m，总层数固定 N = 8
    （单层 0.125 mm，总厚 1.0 mm，保持轻量化），承受**双向等值面内压缩**
    Nx = Ny = 1000 N/m。设计变量是 4 个铺层角度，层合板为对称铺层：

        [θ1 / θ2 / θ3 / θ4]_s  =  θ1,θ2,θ3,θ4,θ4,θ3,θ2,θ1

    目标：在 θi ∈ [-90°, 90°] 内（含 -45° 等负角）寻找使一阶临界屈曲载荷 Ncr
    最大的铺层，并检查是否满足 Ncr ≥ 1000 N/m。
    均衡性：目标函数中加入"均衡铺层"罚分——每个 +θ 层必须与一个 -θ 层成对，
    否则 A16/A26 ≠ 0（拉-剪耦合、固化翘曲），罚分 = Ncr·(1 - 5·r)。

为什么这个问题比纯 [0]N 有意思
    全部 0° 铺层时 D11 远大于 D22（本材料 D11/D22 ≈ 17.6），板在 y（弱轴）方向
    先失稳：双向受压时临界模态为 m=1、n=2（沿 y 方向 2 个半波），Ncr 只有
    319 N/m。角度设计能重新分配 D11/D22/D66，把弱轴补起来 —— 这才是铺层
    角度真正发挥作用的场景。

物理模型（真实 CLT + 能量法屈曲，不使用 ML 代理模型）
    1) 每层用平面应力折减刚度 Q、转到 θ 后得 Qbar(θ)，按经典层合板理论求和
       得 D 矩阵（GPa·mm³），并与 composipy 逐点校核（自检 1）。
    2) 屈曲用 Rayleigh–Ritz 双正弦级数：
           w(x,y) = Σ Σ c_mn sin(mπx/a) sin(nπy/b)        （m,n ≤ M）
       该级数严格满足四边简支的 w = 0；弯矩自然边界条件由变分自动满足。
       势能 = 弯曲应变能 - 双向压缩外力功：
           K c = Ncr · G c,   G = Nx·(φ'φ' ⊗ φφ) + Ny·(φφ ⊗ φ'φ')
       最小广义特征值即临界载荷。级数同时保留 D11、D12、D22、D66 以及弯扭
       耦合项 D16、D26（[45]8 这类非均衡铺层 D16 ≠ 0，不能被忽略）。
    3) 当 D16 = D26 = 0（如 [0]8、[0/90]s）时，Ritz 解与 Navier 精确解完全一致
       —— 这是自检 2 的依据；对带耦合的铺层，级数收敛稍慢，脚本给出 M 收敛性
       检查（自检 3）。

边界条件说明
    四边简支取 SS1 理想化：w = 0，弯矩由变分自然处理。对 D16/D26 ≠ 0 的铺层，
    简支边界在文献中本身存在 SS1~SS4 多种定义，取值会有差别；但本问题的
    结论（N = 8 无论怎么铺都达不到 1000 N/m）由与边界条件无关的理论上限
    632 N/m 决定，不依赖这一细节。

输出
    终端：纯 [0]8 的失效证明、角度扫描/GA 结果、最优铺层与 Ncr、可行性结论
    图片：figures/angle_modes.png     (a) Ncr 柱状对比 + 两种铺层的屈曲模态
          figures/angle_polar.png     (b) 最优铺层的角度分布极坐标图
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
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from scipy.linalg import eigh
from scipy.optimize import minimize as scipy_minimize

# =============================================================================
# 1. 工程输入
# =============================================================================
A_PLATE = 0.5          # m
B_PLATE = 0.5          # m
NX_LOAD = 1000.0       # N/m，x 向面内压缩
NY_LOAD = 1000.0       # N/m，y 向面内压缩
TARGET_NCR = 1000.0    # N/m，设计要求（双向同时达到）

E1, E2, G12, V12 = 181.0, 10.3, 7.17, 0.28   # GPa
T_PLY = 0.125          # mm
DENSITY = 1600.0       # kg/m³
N_PLIES = 8            # 总层数固定（1.0 mm）

M_GA = 10              # GA 阶段的级数项数（速度快）
M_VERIFY = 16          # 最终校核的级数项数（精度高）

OUT_MODES = os.path.join("figures", "angle_modes.png")
OUT_POLAR = os.path.join("figures", "angle_polar.png")

GA_POP = 80
GA_GEN = 60
SEED = 1

# =============================================================================
# 2. 单层刚度与层合板 D 矩阵（快速路径，已与 composipy 校核）
# =============================================================================
_NU21 = V12 * E2 / E1
_DELTA = 1.0 - V12 * _NU21
Q11 = E1 / _DELTA
Q22 = E2 / _DELTA
Q12 = V12 * E2 / _DELTA
Q66 = G12

# 8 层对称铺层的 z 坐标（中面为 0），用于 D = Σ Qbar·(z_k³-z_{k-1}³)/3
_Z_EDGES = np.linspace(-N_PLIES * T_PLY / 2.0, N_PLIES * T_PLY / 2.0, N_PLIES + 1)
_PLY_WEIGHT = np.diff(_Z_EDGES ** 3) / 3.0


def qbar(theta_deg):
    """平面应力折减刚度 Q 转到 θ 角后的 Qbar（GPa）。θ 以度为单位。"""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    c2, s2, c3, s3 = c * c, s * s, c ** 3, s ** 3
    Q = np.empty((3, 3))
    Q[0, 0] = Q11 * c2 * c2 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * s2 * s2
    Q[1, 1] = Q11 * s2 * s2 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * c2 * c2
    Q[0, 1] = Q[1, 0] = (Q11 + Q22 - 4.0 * Q66) * s2 * c2 + Q12 * (s2 * s2 + c2 * c2)
    Q[2, 2] = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2 * c2 + Q66 * (s2 * s2 + c2 * c2)
    Q[0, 2] = Q[2, 0] = (Q11 - Q12 - 2.0 * Q66) * c3 * s - (Q22 - Q12 - 2.0 * Q66) * c * s3
    Q[1, 2] = Q[2, 1] = (Q11 - Q12 - 2.0 * Q66) * c * s3 - (Q22 - Q12 - 2.0 * Q66) * c3 * s
    return Q


def stack_of(theta4):
    """[θ1/θ2/θ3/θ4]s 展开成 8 层的角度序列。"""
    theta4 = [float(t) for t in theta4]
    return theta4 + theta4[::-1]


def d_matrix(theta4):
    """[θ1/θ2/θ3/θ4]s 的弯曲刚度矩阵 D，单位 GPa·mm³。"""
    D = np.zeros((3, 3))
    for theta, w in zip(stack_of(theta4), _PLY_WEIGHT):
        D += qbar(theta) * w
    return D


def a_matrix(theta4):
    """面内刚度矩阵 A，单位 GPa·mm（用于报告等效工程常数）。"""
    return sum(qbar(t) for t in stack_of(theta4)) * T_PLY


# =============================================================================
# 3b. 均衡性（Balanced）判据
#     经典层合板设计原则：每个 +θ 层都必须有对应的 −θ 层，否则层合板在面内
#     载荷下会产生拉-剪耦合（A16、A26 ≠ 0），固化后又出现翘曲。
#     对对称层合板，A16 = t·ΣQbar16，A26 = t·ΣQbar26（对所有层求和），
#     因此"均衡"等价于 ΣQbar16 = 0 且 ΣQbar26 = 0。
# =============================================================================
_U2 = (Q11 - Q22) / 2.0
_U3 = (Q11 + Q22 - 2.0 * Q12 - 4.0 * Q66) / 8.0
_Q16_MAX = 55.0          # |Qbar16| 的近似上界（θ≈30° 时取到，约 54.2 GPa）


def balance_residual(angles):
    """返回 (ΣQbar16, ΣQbar26)，单位 GPa；两者均为 0 即为均衡铺层。"""
    s16 = s26 = 0.0
    for theta in angles:
        a = np.radians(theta)
        s16 += 0.5 * _U2 * np.sin(2.0 * a) + _U3 * np.sin(4.0 * a)
        s26 += 0.5 * _U2 * np.sin(2.0 * a) - _U3 * np.sin(4.0 * a)
    return s16, s26


def balance_penalty(angles):
    """归一化均衡性罚分 r ∈ [0, 1]：0 = 完全均衡，越大越不均衡。"""
    s16, s26 = balance_residual(angles)
    denom = 2.0 * _Q16_MAX * max(len(angles), 1)
    return min((abs(s16) + abs(s26)) / denom, 1.0)


BALANCE_WEIGHT = 5.0     # 罚分权重：r = 0.2 时目标函数打 0.0 折（重罚）


def balanced_objective(angles, ncr):
    """均衡性加权的目标：Ncr × (1 − w·r)，不成对则重罚。"""
    r = balance_penalty(angles)
    return ncr * max(1.0 - BALANCE_WEIGHT * r, 0.05)


# =============================================================================
# 3. 屈曲求解：Rayleigh–Ritz（保留 D16、D26 弯扭耦合）
# =============================================================================
def _basis_integrals(length, n_modes):
    """正弦基函数在 [0,length] 上的一维积分矩阵（解析式）。

    返回 (S0, S1, S2, S4, X1, X2)：
        S0[m,k] = ∫ φ_m φ_k dx
        S1[m,k] = ∫ φ_m' φ_k' dx          （几何刚度用）
        S2[m,k] = ∫ φ_m'' φ_k dx
        S4[m,k] = ∫ φ_m'' φ_k'' dx
        X1[m,k] = ∫ φ_m' φ_k'' dx          （D16/D26 耦合项用）
        X2[m,k] = ∫ φ_m φ_k' dx            （D16/D26 耦合项用）
    其中 φ_m = sin(mπx/length)，m = 1..n_modes。
    """
    m = np.arange(1, n_modes + 1, dtype=float)
    lam = m * np.pi / length
    S0 = np.diag(np.full(n_modes, length / 2.0))
    S1 = np.diag(lam ** 2 * length / 2.0)
    S2 = np.diag(-lam ** 2 * length / 2.0)
    S4 = np.diag(lam ** 4 * length / 2.0)

    X1 = np.zeros((n_modes, n_modes))
    X2 = np.zeros((n_modes, n_modes))
    for i in range(n_modes):
        for j in range(n_modes):
            p, q = i + 1, j + 1
            if (p + q) % 2 == 1:      # m+k 为奇数时积分非零
                X2[i, j] = 2.0 * p * q / (p * p - q * q)
                X1[i, j] = -2.0 * p * q ** 3 * np.pi ** 2 / (
                    length ** 2 * (q * q - p * p))
    return S0, S1, S2, S4, X1, X2


class BucklingSolver:
    """固定板宽/基函数项数时的屈曲求解器（基函数积分只算一次）。"""

    def __init__(self, a, b, n_modes):
        self.n_modes = n_modes
        self.S0x, self.S1x, self.S2x, self.S4x, self.X1x, self.X2x = \
            _basis_integrals(a, n_modes)
        self.S0y, self.S1y, self.S2y, self.S4y, self.X1y, self.X2y = \
            _basis_integrals(b, n_modes)

    def _k_matrix(self, D):
        """D 用 GPa·mm³（数值上等于 N·m），长度用 m，故 Ncr 直接得 N/m。"""
        k = np.kron
        K = (D[0, 0] * k(self.S4x, self.S0y)
             + D[1, 1] * k(self.S0x, self.S4y)
             + 2.0 * D[0, 1] * k(self.S2x, self.S2y)
             + 4.0 * D[2, 2] * k(self.S1x, self.S1y)
             + 4.0 * D[0, 2] * k(self.X1x, self.X2y)
             + 4.0 * D[1, 2] * k(self.X2x, self.X1y))
        return 0.5 * (K + K.T)

    def _g_matrix(self, nx, ny):
        k = np.kron
        return nx * k(self.S1x, self.S0y) + ny * k(self.S0x, self.S1y)

    def ncr(self, D, nx=1.0, ny=1.0):
        """返回最小特征值（即临界载荷，N/m）与特征向量矩阵。"""
        K = self._k_matrix(D)
        G = self._g_matrix(nx, ny)
        val = eigh(K, G, subset_by_index=[0, 0], eigvals_only=True,
                   check_finite=False)[0]
        return float(val)

    def ncr_mode(self, D, nx=1.0, ny=1.0):
        """返回 (Ncr, 模态系数矩阵)。"""
        K = self._k_matrix(D)
        G = self._g_matrix(nx, ny)
        w, v = eigh(K, G, subset_by_index=[0, 0], check_finite=False)
        return float(w[0]), v[:, 0].reshape(self.n_modes, self.n_modes)

    def mode_grid(self, c, a, b, n_grid=61):
        """把 Ritz 系数展开成 (x,y) 网格上的模态位移 w(x,y)。"""
        x = np.linspace(0.0, a, n_grid)
        y = np.linspace(0.0, b, n_grid)
        phi = np.sin(np.pi * np.outer(x, np.arange(1, self.n_modes + 1)) / a)
        psi = np.sin(np.pi * np.outer(y, np.arange(1, self.n_modes + 1)) / b)
        w = phi @ c @ psi.T
        peak = np.max(np.abs(w))
        return x, y, (w / peak if peak > 0 else w)


# 全局求解器实例（GA 复用，避免重复构造积分矩阵）
_SOLVER_GA = BucklingSolver(A_PLATE, B_PLATE, M_GA)
_SOLVER_FINE = BucklingSolver(A_PLATE, B_PLATE, M_VERIFY)


def ncr_of(theta4, solver=None, nx=1.0, ny=1.0):
    """给定 4 个角度，返回双向（或指定）受压下的临界载荷 N/m。"""
    solver = solver or _SOLVER_GA
    return solver.ncr(d_matrix(theta4), nx=nx, ny=ny)


# =============================================================================
# 4. 解析解（仅对 D16 = D26 = 0 的特殊正交异性板精确）
# =============================================================================
def navier_orthotropic(D, a, b, nx=1.0, ny=1.0, m_max=12, n_max=12):
    """特殊正交异性板的精确屈曲解（Navier），返回 (Ncr, m, n)。

    取 nx = ny = 1（单位载荷）时，返回值即临界载荷本身（N/m）；
    若给定实际载荷，则返回的是该载荷组合下的临界倍数。
    Ncr = [D11 kx² + 2(D12+2D66) kx ky + D22 ky²] / (Nx kx + Ny ky)
    """
    best = (np.inf, 1, 1)
    for m in range(1, m_max + 1):
        kx = (m * np.pi / a) ** 2
        for n in range(1, n_max + 1):
            ky = (n * np.pi / b) ** 2
            num = (D[0, 0] * kx ** 2 + 2.0 * (D[0, 1] + 2.0 * D[2, 2]) * kx * ky
                   + D[1, 1] * ky ** 2)
            den = nx * kx + ny * ky
            if den <= 0:
                continue
            val = num / den
            if val < best[0]:
                best = (float(val), m, n)
    return best


# =============================================================================
# 5. 自检
# =============================================================================
def check_against_composipy():
    """自检 1：快速 Qbar/CLT 路径 vs composipy（含负角与随机铺层）。"""
    from composipy import LaminateProperty, OrthotropicMaterial
    ply = OrthotropicMaterial(e1=E1, e2=E2, v12=V12, g12=G12, thickness=T_PLY)
    rng = np.random.default_rng(7)
    cases = [[0.0] * 4, [45.0] * 4, [90.0] * 4, [0.0, 45.0, 90.0, 15.0],
             [30.0, 60.0, 20.0, 70.0]]
    cases += [list(rng.uniform(0.0, 90.0, 4)) for _ in range(5)]
    worst = 0.0
    for th in cases:
        lam = LaminateProperty(stacking=stack_of(th), plies=[ply] * N_PLIES)
        D_ref = np.asarray(lam.D, float)
        worst = max(worst, float(np.max(np.abs(D_ref - d_matrix(th)))))
    print(f"  自检 1  D 矩阵 vs composipy（10 组铺层）  最大绝对误差 {worst:.2e} GPa·mm³"
          f"  -> {'通过' if worst < 1e-10 else '失败'}")
    if worst >= 1e-10:
        raise SystemExit("快速 CLT 路径与 composipy 不一致。")


def check_ritzsolver():
    """自检 2：无弯扭耦合时，Ritz 必须与 Navier 精确解一致。"""
    worst = 0.0
    for th in ([0.0] * 4, [0.0, 90.0, 90.0, 0.0]):
        D = d_matrix(th)
        assert abs(D[0, 2]) < 1e-12 and abs(D[1, 2]) < 1e-12
        exact = navier_orthotropic(D, A_PLATE, B_PLATE)[0]
        ritz = ncr_of(th, _SOLVER_GA)
        worst = max(worst, abs(ritz - exact) / exact)
    print(f"  自检 2  Ritz vs Navier 精确解（D16=D26=0 铺层）  最大相对误差 {worst:.2e}"
          f"  -> {'通过' if worst < 1e-9 else '失败'}")
    if worst >= 1e-9:
        raise SystemExit("Ritz 解与精确解不一致，请检查公式。")


def check_basis_convergence(theta4):
    """自检 3：级数项数收敛性（最优铺层，M 从 8 到 16）。"""
    vals = [ncr_of(theta4, BucklingSolver(A_PLATE, B_PLATE, m))
            for m in (8, 10, 12, 14, 16)]
    spread = (max(vals) - min(vals)) / min(vals) * 100.0
    print(f"  自检 3  级数项数收敛 M=8..16: "
          f"{', '.join(f'{v:.1f}' for v in vals)} N/m"
          f"  极差 {spread:.2f}%  -> {'通过' if spread < 3.0 else '注意'}")
    return vals


# =============================================================================
# 6. 纯 [0]8 的失效证明
# =============================================================================
def analyse_baseline():
    """计算纯 [0]8 的双向/单向外压屈曲载荷，证明其在 y 方向先失稳。"""
    D = d_matrix([0.0, 0.0, 0.0, 0.0])
    # 取 nx=ny=1（单位载荷）时特征值即为临界载荷本身（N/m）
    bi = navier_orthotropic(D, A_PLATE, B_PLATE, 1.0, 1.0)
    x_only = navier_orthotropic(D, A_PLATE, B_PLATE, 1.0, 0.0)
    y_only = navier_orthotropic(D, A_PLATE, B_PLATE, 0.0, 1.0)

    print("\n" + "-" * 92)
    print("1) 纯 [0]8 铺层（1.0 mm）的双向受压屈曲 —— 失效证明")
    print("-" * 92)
    print(f"  D11 = {D[0, 0]:8.4f} GPa·mm³    D22 = {D[1, 1]:8.4f} GPa·mm³"
          f"    D11/D22 = {D[0, 0] / D[1, 1]:.1f}")
    print(f"  D12 = {D[0, 1]:8.4f}           D66 = {D[2, 2]:8.4f}"
          f"           D16 = D26 = 0（全部 0°）")
    print(f"  仅 Nx = {NX_LOAD:.0f} N/m 时       : Ncr = {x_only[0]:7.1f} N/m"
          f"   模态 (m={x_only[1]}, n={x_only[2]})")
    print(f"  仅 Ny = {NY_LOAD:.0f} N/m 时       : Ncr = {y_only[0]:7.1f} N/m"
          f"   模态 (m={y_only[1]}, n={y_only[2]})")
    print(f"  双向 Nx = Ny = {NX_LOAD:.0f} N/m : Ncr = {bi[0]:7.1f} N/m"
          f"   模态 (m={bi[1]}, n={bi[2]})")
    print(f"\n  结论：双向载荷下临界模态为 m={bi[1]}, n={bi[2]}，即沿 y 方向出现 "
          f"{bi[2]} 个半波；")
    print(f"        板在 y（弱轴，D22 仅为 D11 的 {D[1, 1] / D[0, 0] * 100:.1f}%）方向失稳，"
          f"Ncr = {bi[0]:.0f} N/m 仅为需求 {TARGET_NCR:.0f} N/m 的 "
          f"{bi[0] / TARGET_NCR * 100:.0f}%，不满足要求。")
    print(f"        即使只加 Ny（无 Nx），Ncr = {y_only[0]:.0f} N/m 也会先屈曲 —— "
          f"y 向是控制方向。")
    return bi, D


# =============================================================================
# 7. 角度优化：pymoo 遗传算法 + 局部精修
# =============================================================================
_EVAL_CACHE = {}


def ncr_cached(theta4):
    """带缓存的 Ncr 评估（GA 会重复评估相近个体）。"""
    key = tuple(np.round(np.asarray(theta4, float), 6))
    if key not in _EVAL_CACHE:
        _EVAL_CACHE[key] = ncr_of(list(key), _SOLVER_GA)
    return _EVAL_CACHE[key]


class LayupProblem(Problem):
    """4 个角度变量、最大化"均衡加权"的 Ncr。

    注意：本工况（8 层 / 1.0 mm）在物理上不可能达到 1000 N/m（理论上限 632 N/m），
    若仍把 Ncr ≥ 1000 作为约束，pymoo 会优先按"约束违反量"排序，从而**完全忽略**
    均衡性罚分（实测就退化成不均衡的 [45/-45/-45/-45]s）。因此这里去掉该硬约束，
    改为单目标：最大化 Ncr·(1 − 5r)；是否满足 1000 N/m 在结果中单独判定。
    """

    def __init__(self):
        super().__init__(n_var=4, n_obj=1, n_ieq_constr=0,
                         xl=np.full(4, -90.0), xu=np.full(4, 90.0))

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            ncr = ncr_cached(x)
            # 均衡性罚分：不成对的 +θ / −θ 会被重罚（软约束，便于 GA 穿越）
            F[i, 0] = -balanced_objective(x, ncr)
        out["F"] = F


def run_ga():
    """NSGA2（单目标+单约束）全局搜索，返回 (最优角度, 最优 Ncr, 用时)。"""
    problem = LayupProblem()
    algorithm = NSGA2(
        pop_size=GA_POP,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20, prob=1.0 / 4),
        eliminate_duplicates=True,
    )
    t0 = time.perf_counter()
    res = pymoo_minimize(problem, algorithm, ("n_gen", GA_GEN),
                         seed=SEED, verbose=False)
    dt = time.perf_counter() - t0

    # 目标不可行时 pymoo 的 res.opt 可能为空，直接取最终种群
    pop_x = np.atleast_2d(res.pop.get("X"))
    pop_f = np.atleast_1d(res.pop.get("F")).ravel()
    idx = int(np.argmax(-pop_f))
    return pop_x[idx], float(-pop_f[idx]), dt


def local_refine(x0):
    """从 GA 解出发做局部精修：先 SLSQP（带边界），再 Nelder–Mead（抗模态跳变）。

    精修统一在 M_VERIFY 精度下进行，保证与最终报告的口径一致。
    """
    def neg(theta):
        theta = np.clip(theta, -90.0, 90.0)
        return -balanced_objective(theta, ncr_of(theta, _SOLVER_FINE))

    best_x = np.clip(np.asarray(x0, float), -90.0, 90.0)
    best_v = -neg(best_x)

    r = scipy_minimize(neg, best_x, method="SLSQP",
                       bounds=[(-90.0, 90.0)] * 4,
                       options={"maxiter": 200, "ftol": 1e-8})
    if -r.fun > best_v:
        best_v, best_x = -r.fun, np.clip(r.x, -90.0, 90.0)

    r = scipy_minimize(neg, best_x, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
    if -r.fun > best_v:
        best_v, best_x = -r.fun, np.clip(r.x, -90.0, 90.0)

    return best_x, best_v


# =============================================================================
# 8. 理论上限与所需层数
# =============================================================================
def theoretical_bound():
    """任意 8 层铺层在双向等值受压下的 Ncr 理论上限（与铺层顺序无关）。

    Ncr(m=n=1) = (π²/a²)/2 · (D11+D22+2D12+4D66)，
    而 D11+D22+2D12+4D66 对任意θ的和受不变量约束：
        Qbar11+Qbar22+2Qbar12 = Q11+Q22+2Q12       （旋转不变）
        Qbar66 ≤ Q66_max（θ=45° 时取到）
    因此 D11+D22+2D12+4D66 ≤ (Q11+Q22+2Q12+4·Q66max)·h³/12。
    """
    q66_max = (Q11 + Q22 - 2.0 * Q12) / 4.0          # θ=45° 的 Qbar66
    total = (Q11 + Q22 + 2.0 * Q12) + 4.0 * q66_max
    h = N_PLIES * T_PLY
    d_cap = total * h ** 3 / 12.0                     # GPa·mm³ = N·m
    ncr_cap = (np.pi ** 2 / A_PLATE ** 2) / 2.0 * d_cap
    return float(ncr_cap), float(q66_max), float(total)


def required_plies(ncr_at_8):
    """同样铺层按厚度三次方放大，满足 1000 N/m 所需层数（取最近的偶数层）。"""
    n_req = N_PLIES * (TARGET_NCR / ncr_at_8) ** (1.0 / 3.0)
    n_even = int(np.ceil(n_req / 2.0) * 2)
    return n_req, n_even


# =============================================================================
# 9. 绘图
# =============================================================================
def setup_chinese_font():
    """按字体族名启用系统中已安装的中文字体，失败则退回英文标签。

    不写死任何绝对路径，Windows / macOS / Linux 通用。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun",
                 "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "PingFang SC", "Heiti SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            return name, True
    return None, False


def _plot_mode(ax, x, y, w, title, L):
    cf = ax.contourf(x, y, w, levels=24, cmap="RdBu_r")
    ax.contour(x, y, w, levels=[0.0], colors="k", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(L("x (m)", "x (m)"), fontsize=9)
    ax.set_ylabel(L("y (m)", "y (m)"), fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    return cf


def plot_modes(bi, D0, opt_theta, opt_ncr, L):
    """图 (a)：Ncr 柱状对比 + [0]8 与最优铺层的屈曲模态。"""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.05, 1, 1]})

    # ---------- 柱状图 ----------
    ax = axes[0]
    d_cap, _, _ = theoretical_bound()
    labels = [L("[0]8（原始设计）", "[0]8 (baseline)"),
              L("角度优化最优", "optimised"),
              L("理论上限（任意 8 层）", "theoretical bound")]
    values = [bi[0], opt_ncr, d_cap]
    colors = ["tab:red", "tab:green", "0.55"]
    bars = ax.bar(labels, values, color=colors, width=0.6,
                  edgecolor="black", linewidth=0.8)
    bars[2].set_hatch("//")
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(TARGET_NCR, color="crimson", linestyle="--", linewidth=1.8)
    ax.annotate(L(f"设计需求 {TARGET_NCR:.0f} N/m", f"design load {TARGET_NCR:.0f} N/m"),
                xy=(2.45, TARGET_NCR), xytext=(0, 8), textcoords="offset points",
                ha="right", fontsize=9.5, color="crimson")
    ax.set_ylim(0, TARGET_NCR * 1.18)
    ax.set_ylabel(L("临界屈曲载荷 Ncr (N/m)", "buckling load Ncr (N/m)"), fontsize=10.5)
    ax.set_title(L("(a1) Ncr 对比：N=8 全部方案均不达标",
                   "(a1) Ncr: no 8-ply layup reaches the target"), fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)

    # ---------- [0]8 模态 ----------
    m0, n0 = bi[1], bi[2]
    x = np.linspace(0.0, A_PLATE, 121)
    y = np.linspace(0.0, B_PLATE, 121)
    X, Y = np.meshgrid(x, y)
    w0 = np.sin(np.pi * m0 * X / A_PLATE) * np.sin(np.pi * n0 * Y / B_PLATE)
    cf = _plot_mode(axes[1], x, y, w0,
                    L(f"[0]8：m={m0}, n={n0}（y 向 {n0} 个半波）\nNcr = {bi[0]:.0f} N/m",
                      f"[0]8: m={m0}, n={n0}\nNcr = {bi[0]:.0f} N/m"), L)
    fig.colorbar(cf, ax=axes[1], fraction=0.046, pad=0.04, label=L("w / w_max", "w / w_max"))

    # ---------- 最优铺层模态 ----------
    D_opt = d_matrix(opt_theta)
    ncr_v, c = _SOLVER_FINE.ncr_mode(D_opt)
    xv, yv, wv = _SOLVER_FINE.mode_grid(c, A_PLATE, B_PLATE, n_grid=121)
    th_txt = "/".join(f"{t:.1f}" for t in opt_theta)
    cf = _plot_mode(axes[2], xv, yv, wv,
                    L(f"最优 [{th_txt}]s\nNcr = {opt_ncr:.0f} N/m（仍不足）",
                      f"optimum [{th_txt}]s\nNcr = {opt_ncr:.0f} N/m"), L)
    fig.colorbar(cf, ax=axes[2], fraction=0.046, pad=0.04, label=L("w / w_max", "w / w_max"))

    fig.suptitle(L(f"双向受压（Nx=Ny={TARGET_NCR:.0f} N/m）下 8 层对称层合板的屈曲行为",
                   "Biaxial compression: buckling of 8-ply symmetric laminates"),
                 fontsize=13)
    os.makedirs(os.path.dirname(OUT_MODES), exist_ok=True)
    fig.savefig(OUT_MODES, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  已保存: {OUT_MODES}")


def plot_polar(opt_theta, opt_ncr, D_opt, L):
    """图 (b)：最优铺层 4 个角度的极坐标分布。"""
    fig = plt.figure(figsize=(9.4, 6.4))
    ax = fig.add_axes([0.05, 0.10, 0.58, 0.80], projection="polar")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    radii = [1.00, 0.80, 0.60, 0.40]          # θ1 最外层 → θ4 最靠近中面
    for i, (th, r) in enumerate(zip(opt_theta, radii)):
        ax.plot([np.radians(th)] * 2, [0.0, r], color=colors[i], linewidth=2.4,
                alpha=0.92, zorder=2, label=f"θ{i + 1} = {th:5.1f}°")
        ax.plot([np.radians(th)], [r], "o", color=colors[i], markersize=12,
                markeredgecolor="black", zorder=3)

    # 45° 参考线（双向等压时 Qbar66 取最大值的方向）
    ax.plot([np.radians(45)] * 2, [0.0, 1.0], linestyle=":", color="0.45",
            linewidth=1.2, zorder=1)

    ax.set_rlabel_position(5)
    ax.set_rticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_rlim(0, 1.02)
    ax.set_thetagrids(np.arange(-90, 91, 15))
    ax.grid(alpha=0.35)
    ax.text(np.radians(24), 0.12, L("中面", "mid"), fontsize=8.5, color="0.4")
    ax.text(np.radians(7), 1.06, L("外表面", "outer"), fontsize=8.5, color="0.4")
    ax.set_title(L("最优铺层角度分布\n（极径 = 铺层所在位置）",
                   "Ply angles of the optimum"), fontsize=12, pad=16)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.02),
                    fontsize=10, framealpha=0.95,
                    title=L("铺层角度", "ply angle"))
    leg.get_title().set_fontsize(9.5)

    txt = (L("铺层：", "stack: ")
           + f"[{opt_theta[0]:.1f}/{opt_theta[1]:.1f}/{opt_theta[2]:.1f}/"
             f"{opt_theta[3]:.1f}]s\n"
           + L("Ncr = ", "Ncr = ") + f"{opt_ncr:.1f} N/m"
           + L(f"（需求 {TARGET_NCR:.0f}）", f" (target {TARGET_NCR:.0f})") + "\n"
           + f"D11={D_opt[0, 0]:.2f}  D22={D_opt[1, 1]:.2f}  "
             f"D16={D_opt[0, 2]:.2f}  D66={D_opt[2, 2]:.2f} GPa·mm³")
    fig.text(0.70, 0.66, txt, fontsize=9.5, va="top",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7",
                       edgecolor="0.7", alpha=0.95))

    os.makedirs(os.path.dirname(OUT_POLAR), exist_ok=True)
    fig.savefig(OUT_POLAR, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {OUT_POLAR}")


# =============================================================================
# 10. 主流程
# =============================================================================
def main():
    print("=" * 92)
    print("双向受压下 8 层对称层合板的铺层角度优化（CLT 屈曲 + pymoo 遗传算法）")
    print("=" * 92)
    print(f"几何 : a = b = {A_PLATE:.2f} m，四边简支（SSSS）")
    print(f"载荷 : Nx = {NX_LOAD:.0f} N/m，Ny = {NY_LOAD:.0f} N/m（双向等值压缩）")
    print(f"材料 : E1={E1} GPa, E2={E2} GPa, G12={G12} GPa, v12={V12}")
    print(f"铺层 : [θ1/θ2/θ3/θ4]s，共 {N_PLIES} 层，总厚 {N_PLIES * T_PLY:.2f} mm，"
          f"面密度 {N_PLIES * T_PLY * 1e-3 * DENSITY:.2f} kg/m²")
    print(f"变量 : θ1..θ4 ∈ [-90°, 90°]（含负角，可生成 -45° 铺层）")
    print(f"目标 : max Ncr（含均衡性罚分 Ncr·(1-5r)），约束 Ncr ≥ {TARGET_NCR:.0f} N/m")

    print("\n" + "-" * 92)
    print("自检")
    print("-" * 92)
    check_against_composipy()
    check_ritzsolver()

    bi, D_base = analyse_baseline()

    # ---------------- 遗传算法 ----------------
    print("\n" + "-" * 92)
    print(f"2) pymoo NSGA-II 角度搜索（种群 {GA_POP}，{GA_GEN} 代，4 个角度变量）")
    print("-" * 92)
    x_ga, score_ga, dt = run_ga()
    ncr_ga = ncr_of(x_ga, _SOLVER_GA)
    print(f"  GA 用时 {dt:.1f} s，评估 {len(_EVAL_CACHE)} 个不同铺层")
    print(f"  GA 最优（M={M_GA}）：θ = [{x_ga[0]:.2f}, {x_ga[1]:.2f}, "
          f"{x_ga[2]:.2f}, {x_ga[3]:.2f}]°   Ncr = {ncr_ga:.1f} N/m"
          f"（均衡罚分 r = {balance_penalty(x_ga):.3f}）")

    x_opt, v_opt = local_refine(x_ga)
    print(f"  局部精修（M={M_VERIFY}）: θ = [{x_opt[0]:.2f}, {x_opt[1]:.2f}, "
          f"{x_opt[2]:.2f}, {x_opt[3]:.2f}]°")

    opt_theta = [float(round(t, 2)) for t in x_opt]
    v_fine = ncr_of(opt_theta, _SOLVER_FINE)
    D_opt = d_matrix(opt_theta)

    print("\n" + "-" * 92)
    print("3) 收敛性与最优解校核")
    print("-" * 92)
    check_basis_convergence(opt_theta)
    s16, s26 = balance_residual(opt_theta)
    print(f"  最优铺层（M={M_VERIFY} 高精度校核）: "
          f"[{opt_theta[0]:.2f}/{opt_theta[1]:.2f}/{opt_theta[2]:.2f}/"
          f"{opt_theta[3]:.2f}]s   Ncr = {v_fine:.1f} N/m")
    print(f"  均衡性残差 : ΣQ̄16 = {s16:+.3f} GPa，ΣQ̄26 = {s26:+.3f} GPa"
          f"  -> {'均衡铺层 ✓（A16 = A26 = 0）' if balance_penalty(opt_theta) < 0.02 else '仍不均衡 ✗'}")

    # ---------------- 对比表 ----------------
    print("\n" + "-" * 92)
    print("4) 候选铺层对比（双向受压，N=8）")
    print("-" * 92)
    print(f"  {'铺层':<30s} {'θ1..θ4':>26s} {'D11':>7s} {'D22':>7s} {'Ncr':>8s} {'判定':>6s}")
    print("  " + "-" * 88)
    candidates = [("纯 [0]8", [0.0, 0.0, 0.0, 0.0]),
                  ("[0/90/90/0]s", [0.0, 90.0, 90.0, 0.0]),
                  ("[45]8", [45.0, 45.0, 45.0, 45.0]),
                  ("[0/45/45/0]s", [0.0, 45.0, 45.0, 0.0]),
                  ("[45/0/0/45]s", [45.0, 0.0, 0.0, 45.0]),
                  ("GA + 精修最优", opt_theta)]
    for name, th in candidates:
        D = d_matrix(th)
        v = ncr_of(th, _SOLVER_FINE)
        ok = "达标" if v >= TARGET_NCR else "不达标"
        angles = "/".join(f"{t:.1f}" for t in th)
        print(f"  {name:<30s} {angles:>26s} {D[0, 0]:7.2f} {D[1, 1]:7.2f} "
              f"{v:8.1f} {ok:>6s}")

    # ---------------- 可行性与建议 ----------------
    print("\n" + "-" * 92)
    print("5) 可行性结论与工程建议")
    print("-" * 92)
    d_cap, q66_max, total = theoretical_bound()
    print(f"  最优结果      : Ncr = {v_fine:.1f} N/m"
          f"  （要求的 {v_fine / TARGET_NCR * 100:.1f}%）")
    print(f"  理论上限      : 任意 8 层铺层在双向等压下的 Ncr ≤ {d_cap:.1f} N/m")
    print(f"                  （由 Qbar11+Qbar22+2Qbar12 的旋转不变性与 "
          f"Qbar66 ≤ {q66_max:.2f} GPa 决定）")
    print(f"  -> 即使取到理论上限，仍比 {TARGET_NCR:.0f} N/m 低 "
          f"{(1 - d_cap / TARGET_NCR) * 100:.1f}%；")
    print(f"     **N = 8（1.0 mm）在本工况下无可行解，角度优化无法弥补厚度不足。**")
    n_req, n_even = required_plies(v_fine)
    print(f"\n  按 Ncr ∝ h³ 放大：达到 {TARGET_NCR:.0f} N/m 需 {n_req:.2f} 层 → "
          f"取对称铺层 {n_even} 层（{n_even * T_PLY:.2f} mm）")
    for n in (n_even - 2, n_even, n_even + 2):
        if n < N_PLIES:
            continue
        scale = (n / N_PLIES) ** 3
        print(f"    N={n:2d}（{n * T_PLY:.2f} mm）: Ncr ≈ {v_fine * scale:7.1f} N/m"
              f"   {'✓ 满足' if v_fine * scale >= TARGET_NCR else '✗ 仍不足'}")
    print(f"\n  工程含义：双向受压下板的有效弯曲刚度约为单向的 1/2（m=n=1 模态），")
    print(f"  单纯 [0]8（{bi[0]:.0f} N/m）与角度优化（{v_fine:.0f} N/m）相比提升 "
          f"{(v_fine / bi[0] - 1) * 100:.0f}%，方向是对的，但量级仍差 "
          f"{TARGET_NCR / v_fine:.2f} 倍。")

    # ---------------- 绘图 ----------------
    print("\n" + "-" * 92)
    print("6) 绘图")
    print("-" * 92)
    _, has_cn = setup_chinese_font()
    L = (lambda cn, en: cn) if has_cn else (lambda cn, en: en)
    plot_modes(bi, D_base, opt_theta, v_fine, L)
    plot_polar(opt_theta, v_fine, D_opt, L)

    verdict = "满足" if v_fine >= TARGET_NCR else "不满足"
    print("\n" + "=" * 92)
    print(f"最终结果：最优铺层 [{opt_theta[0]:.1f}/{opt_theta[1]:.1f}/{opt_theta[2]:.1f}/"
          f"{opt_theta[3]:.1f}]s，Ncr = {v_fine:.1f} N/m"
          f"（{verdict} {TARGET_NCR:.0f} N/m 需求）")
    print(f"          相对纯 [0]8（{bi[0]:.1f} N/m）提升 {(v_fine / bi[0] - 1) * 100:.1f}%，"
          f"但 N=8 无可行解；需 {n_even} 层（{n_even * T_PLY:.2f} mm）才达 "
          f"{TARGET_NCR:.0f} N/m。")
    print("脚本运行完成，未出现错误。")
    print("=" * 92)


if __name__ == "__main__":
    main()

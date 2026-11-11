"""级数解的收敛外推：量化 Ritz（M = 16）的截断误差。

为什么需要这一行：含 D16/D26 的铺层没有闭式解。SS1 边界的弯矩条件

    M_n = D11·w,xx + D12·w,yy + 2·D16·w,xy = 0

里的混合导数项使"沿一个方向取正弦"的做法不再成立——这也是 Navier 只在
D16 = D26 = 0（特殊正交异性）时适用的原因。因此"解析解"只能以级数形式存在。

本模块用同一求解器在 M = 16/24/32/40 上的值拟合

    ncr(M) = A + B·M^(-p)

取 A 作为收敛值。它不是独立方法（独立性由 ABAQUS 有限元提供），用途是把
"我用了 M = 16"这句话变成一条可核对的数字：A 与 M = 16 之差就是截断误差。
"""

from __future__ import annotations

import numpy as np

import angle_optimize as ao

MS_DEFAULT = (16, 24, 32, 40)


def _lam(D, nx: float, ny: float, m: int) -> float:
    """返回特征值 λ（Ncr = λ · max(Nx, Ny)）。"""
    return float(ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, m).ncr(D, nx=nx, ny=ny))


def converged_ncr(D, nx: float, ny: float, ms=MS_DEFAULT) -> dict:
    """D 为弯曲刚度矩阵（GPa·mm³）；返回外推的 Ncr 与各阶 M 的值。"""
    scale = max(float(nx), float(ny))
    pts = [(int(m), _lam(D, nx, ny, int(m)) * scale) for m in ms]

    M = np.array([p[0] for p in pts], float)
    Y = np.array([p[1] for p in pts], float)

    from scipy.optimize import least_squares

    def resid(par):
        A, B, e = par
        return A + B * M ** (-e) - Y

    guess = [Y[-1], Y[0] - Y[-1], 1.0]
    sol = least_squares(resid, guess, bounds=([0.0, -1e6, 0.1], [1e9, 1e6, 12.0]))
    A, B, e = (float(x) for x in sol.x)
    return {
        "ncr": A,
        "B": B,
        "points": pts,
        "p": e,
        "resid": float(np.max(np.abs(sol.fun))),
    }

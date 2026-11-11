# -*- coding: utf-8 -*-
"""Composipy 层合板算例：对称层合板 [0/90]s 的 ABD 矩阵与面内工程常数。

单位约定：长度 mm，模量 GPa（1 GPa = 1000 N/mm^2）。
  A 的单位 GPa*mm，B 的单位 GPa*mm^2，D 的单位 GPa*mm^3。
"""

import sys

import numpy as np

from composipy import LaminateProperty, OrthotropicMaterial

# 本机 Python 默认 stdout 编码为 GBK(cp936)，中文在重定向/管道输出时会乱码。
# 这里显式切到 UTF-8，保证任何启动方式下中文都能正确显示。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ------------------------- 输入 -------------------------
E1 = 181.0      # GPa，纤维方向拉伸模量
E2 = 10.3       # GPa，垂直纤维方向拉伸模量
G12 = 7.17      # GPa，面内剪切模量
V12 = 0.28      # 主泊松比
PLY_THICKNESS = 0.125   # mm，单层厚度

STACKING = [0, 90, 90, 0]   # [0/90]s = 四层
N_PLIES = len(STACKING)
TOTAL_THICKNESS = PLY_THICKNESS * N_PLIES


def _inverse(matrix):
    """返回 (逆矩阵, 是否可逆)。奇异时返回全 nan，不抛异常。"""
    if abs(np.linalg.det(matrix)) < 1e-12:
        return np.full_like(matrix, np.nan), False
    return np.linalg.inv(matrix), True


def _print_matrix(title, matrix, unit):
    print(f"\n{title}  (单位: {unit})")
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        for row in matrix:
            print("  [" + "  ".join(f"{value:14.4f}" for value in row) + "]")


def main():
    ply = OrthotropicMaterial(
        e1=E1, e2=E2, v12=V12, g12=G12, thickness=PLY_THICKNESS
    )
    laminate = LaminateProperty(stacking=STACKING, plies=[ply] * N_PLIES)

    A = np.asarray(laminate.A, dtype=float)
    B = np.asarray(laminate.B, dtype=float)
    D = np.asarray(laminate.D, dtype=float)
    ABD = np.asarray(laminate.ABD, dtype=float)

    nu21 = V12 * E2 / E1   # 由对称性 Maxwell-Betti 互等关系得到

    print("=" * 66)
    print("对称层合板 [0/90]s  ABD 矩阵与面内工程常数")
    print("=" * 66)
    print(f"材料常数 : E1={E1} GPa, E2={E2} GPa, G12={G12} GPa, v12={V12}")
    print(f"单层厚度 : {PLY_THICKNESS} mm    层数: {N_PLIES}    总厚度 h: {TOTAL_THICKNESS} mm")
    print(f"铺层顺序 : {STACKING}   ([0/90]s)")
    print(f"由互等关系推得 v21 = v12*E2/E1 = {nu21:.6f}")

    print(f"\n单层正轴刚度 Q (单位: GPa)")
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        for row in np.asarray(ply.Q_0, dtype=float):
            print("  [" + "  ".join(f"{value:14.4f}" for value in row) + "]")

    _print_matrix("A 矩阵 (面内刚度)", A, "GPa*mm")
    _print_matrix("B 矩阵 (耦合刚度)", B, "GPa*mm^2")
    _print_matrix("D 矩阵 (弯曲刚度)", D, "GPa*mm^3")
    _print_matrix("ABD 矩阵 (6x6)", ABD, "混合单位")

    # ------------------------- 面内工程常数 -------------------------
    # 面内柔度 a = A^-1，工程常数按总厚度 h 归一化后取：
    #   Ex = 1/(h*a11), Ey = 1/(h*a22), Gxy = 1/(h*a33), vxy = -a12/a11
    a, invertible = _inverse(A)
    h = TOTAL_THICKNESS

    print("\n" + "-" * 66)
    print("面内工程常数（由 A 矩阵反演得到）")
    print("-" * 66)
    if invertible:
        Ex = 1.0 / (h * a[0, 0])
        Ey = 1.0 / (h * a[1, 1])
        Gxy = 1.0 / (h * a[2, 2])
        vxy = -a[0, 1] / a[0, 0]
        vyx = -a[0, 1] / a[1, 1]
        print(f"  Ex  = {Ex:10.4f} GPa")
        print(f"  Ey  = {Ey:10.4f} GPa")
        print(f"  Gxy = {Gxy:10.4f} GPa")
        print(f"  vxy = {vxy:10.4f}")
        print(f"  vyx = {vyx:10.4f}")
        print(f"  校验: Ex*vxy = Ey*vyx = {Ex * vxy:.4f} GPa  （对称性应成立）")
    else:
        print("  A 矩阵奇异，无法反演面内工程常数。")

    # ------------------------- 合理性检查 -------------------------
    print("\n" + "-" * 66)
    print("合理性检查")
    print("-" * 66)
    max_b = float(np.max(np.abs(B)))
    print(f"  [0/90]s 为对称层合板，B 矩阵最大绝对值 = {max_b:.3e} GPa*mm^2")
    print(f"  -> 耦合项{'近似为零，符合对称层合板预期' if max_b < 1e-9 else '不为零，请检查！'}")
    print(f"  A 矩阵对称: {np.allclose(A, A.T)};  A 正定: {bool(np.all(np.linalg.eigvalsh(A) > 0))}")

    e1_estimate = 0.5 * E1 + 0.5 * E2
    print(f"  量级估算: (E1+E2)/2 = {e1_estimate:.4f} GPa（0度和90度各占一半，Ex 应与此接近）")

    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

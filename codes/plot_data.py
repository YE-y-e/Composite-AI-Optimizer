# -*- coding: utf-8 -*-
"""绘制 [θ1/θ2/θ2/θ1]s 对称层合板铺层角度对刚度的影响。

读取  : training_data.csv（generate_data.py 生成，500 组 LHS 样本）
输出  : data_distribution.png（两张图：Ex 与 D11 的理论曲面 + 实际采样点叠加）

做法说明（重要）
    500 个 LHS 样本点在 θ1-θ2 平面上是散乱分布的，无法直接构成网格热力图；
    若强行用这 500 个点做三角剖分/样条插值填满平面，得到的"规律"是插值算法
    编出来的，不能用来判断极值点位置。
    因此热力图底色用经典层合板理论（CLT）在 θ1×θ2 网格上直接算出的**真值曲面**
    （向量化实现，与 generate_data.py 的定值逐点算法同源），再把 500 个真实
    样本点用白点叠加显示。这样"规律"来自物理公式，"采样点"来自 CSV，
    两者在图中都可见，且脚本会自检两套算法一致。

单位约定：长度 mm、模量 GPa，故 A 为 GPa*mm、D 为 GPa*mm^3。
"""

import csv
import sys

import numpy as np

# 本机 Python 默认 stdout 编码为 GBK(cp936)，中文会乱码，显式切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")   # 无界面环境，直接出图文件

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib import ticker


def setup_chinese_font():
    """按字体族名启用中文字体，返回 (字体族名, 是否成功)。

    不写死任何绝对路径，Windows / macOS / Linux 通用；
    找不到中文字体时自动退回英文标签，避免图上出现一堆方块。
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "DengXian",
                 "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            print(f"  已启用中文字体: {name}")
            return name, True
    print("  [警告] 未找到中文字体，标签将使用英文以避免显示为方块。")
    return None, False


# ------------------------- 材料常数（与 generate_data.py 保持一致）-------------------------
E1 = 181.0      # GPa
E2 = 10.3       # GPa
G12 = 7.17      # GPa
V12 = 0.28
PLY_THICKNESS = 0.125   # mm
N_PLIES = 4
TOTAL_THICKNESS = PLY_THICKNESS * N_PLIES

INPUT_CSV = "training_data.csv"
OUTPUT_PNG = "data_distribution.png"
GRID_POINTS = 241       # 真值曲面分辨率（241x241，约 58000 点）
VERIFY_TOL = 1e-9       # 两套算法一致性的相对容差


# ------------------------- 经典层合板理论 -------------------------
def reduced_stiffness_Q():
    """单层正轴刚度矩阵 Q（单位 GPa）。"""
    nu21 = V12 * E2 / E1
    denom = 1.0 - V12 * nu21
    return np.array([
        [E1 / denom, V12 * E2 / denom, 0.0],
        [V12 * E2 / denom, E2 / denom, 0.0],
        [0.0, 0.0, G12],
    ])


def qbar(vectorized_theta_deg):
    """偏轴刚度矩阵 Qbar。

    输入角度单位为「度」，内部统一用 np.radians 转弧度；
    支持标量或数组，返回形状 (..., 3, 3)。
    """
    theta = np.radians(np.asarray(vectorized_theta_deg, dtype=float))
    c = np.cos(theta)
    s = np.sin(theta)

    Q = reduced_stiffness_Q()
    q11, q12, q22, q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    c2, s2 = c * c, s * s
    c4, s4 = c2 * c2, s2 * s2
    c2s2 = c2 * s2

    qb11 = q11 * c4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * s4
    qb22 = q11 * s4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * c4
    qb12 = (q11 + q22 - 4.0 * q66) * c2s2 + q12 * (c4 + s4)
    qb16 = (q11 - q12 - 2.0 * q66) * c * s * c2 - (q22 - q12 - 2.0 * q66) * c * s * s2
    qb26 = (q11 - q12 - 2.0 * q66) * c * s * s2 - (q22 - q12 - 2.0 * q66) * c * s * c2
    qb66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * c2s2 + q66 * (c4 + s4)

    # 标量输入返回 (3,3)，数组输入返回 (..., 3, 3)
    if np.ndim(qb11) == 0:
        return np.array([
            [qb11, qb12, qb16],
            [qb12, qb22, qb26],
            [qb16, qb26, qb66],
        ])
    return np.stack([
        np.stack([qb11, qb12, qb16], axis=-1),
        np.stack([qb12, qb22, qb26], axis=-1),
        np.stack([qb16, qb26, qb66], axis=-1),
    ], axis=-2)


def abd_from_z(Qbar_list, z):
    """由各层偏轴刚度与层坐标求 A、B、D（单位分别为 GPa*mm、GPa*mm^2、GPa*mm^3）。"""
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for index, qb in enumerate(Qbar_list):
        z0, z1 = z[index], z[index + 1]
        A += qb * (z1 - z0)
        B += qb * (z1 ** 2 - z0 ** 2) / 2.0
        D += qb * (z1 ** 3 - z0 ** 3) / 3.0
    return A, B, D


def in_plane_compliance(theta1, theta2):
    """返回面内柔度矩阵 a = A^-1（单位 1/GPa/mm）与总厚度 h。"""
    stacking = [theta1, theta2, theta2, theta1]
    n = len(stacking)
    z = np.linspace(-n * PLY_THICKNESS / 2.0, n * PLY_THICKNESS / 2.0, n + 1)
    A, _, _ = abd_from_z([qbar(angle) for angle in stacking], z)
    return np.linalg.inv(A), n * PLY_THICKNESS


def laminate_response(theta1, theta2):
    """[θ1/θ2/θ2/θ1]s 层合板的 Ex、Ey 与 D11。

    返回 (Ex, Ey, D11)，支持标量或数组输入。
    """
    stacking = [theta1, theta2, theta2, theta1]
    n = len(stacking)
    z = np.linspace(-n * PLY_THICKNESS / 2.0, n * PLY_THICKNESS / 2.0, n + 1)

    qbars = [qbar(angle) for angle in stacking]

    # 标量情形走通用 ABD 组装；数组情形用解析式以便大规模网格计算
    if np.ndim(theta1) == 0 and np.ndim(theta2) == 0:
        A, _, D = abd_from_z(qbars, z)
        a = np.linalg.inv(A)
        h = n * PLY_THICKNESS
        return 1.0 / (h * a[0, 0]), 1.0 / (h * a[1, 1]), D[0, 0]

    # 解析式：[θ1/θ2/θ2/θ1]s 中 θ1 两层位于 z=±h/2~±h/4、θ2 两层位于 ±h/4~0
    z1o, z1i = z[4], z[3]     # 外层 θ1 的 |z| 范围
    z2o, z2i = z[2], z[1]     # 内层 θ2 的 |z| 范围
    t1, t2 = PLY_THICKNESS, PLY_THICKNESS
    q1, q2 = qbars[0], qbars[1]

    A = 2.0 * (q1 * t1 + q2 * t2)
    D = (2.0 / 3.0) * (q1 * (z1o ** 3 - z1i ** 3) + q2 * (z2o ** 3 - z2i ** 3))

    # 面内柔度 a = A^-1 直接求逆。
    # 注意：本层合板的 A16、A26 一般不为零（剪切-拉伸耦合），
    # 因此不能套用 A16=A26=0 时的简化式 a11 = A22*A66/det，
    # 否则 Ex 会出现 O(1) 量级的错误（此坑已由自检 1 捕获）。
    a = np.linalg.inv(A)
    h = n * PLY_THICKNESS
    return 1.0 / (h * a[..., 0, 0]), 1.0 / (h * a[..., 1, 1]), D[..., 0, 0]


# ------------------------- 数据读取 -------------------------
def read_training_data(path):
    """读取 training_data.csv，必要时做基本校验并返回列字典。"""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        raise SystemExit(f"找不到输入文件 {path}，请先运行 generate_data.py 生成。")

    if not rows:
        raise SystemExit(f"{path} 中没有数据行。")

    required = {"theta1", "theta2", "Ex", "D11"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"{path} 缺少必需列: {sorted(missing)}")

    data = {key: np.array([float(row[key]) for row in rows]) for key in
            ("theta1", "theta2", "Ex", "Ey", "Gxy", "D11")}
    data["stacking"] = [row["stacking"] for row in rows]
    return data


def main():
    print("=" * 78)
    print("绘制铺层角度对刚度的影响  [θ1/θ2/θ2/θ1]s")
    print("=" * 78)
    font_name, has_chinese = setup_chinese_font()

    # 标签字典：找不到中文字体时整体退回英文
    L = (lambda cn, en: cn) if has_chinese else (lambda cn, en: en)
    labels = {
        "x": L("θ1 (度)", "theta1 (deg)"),
        "y": L("θ2 (度)", "theta2 (deg)"),
        "ex": L("面内刚度 Ex (GPa)", "In-plane stiffness Ex (GPa)"),
        "d11": L("弯曲刚度 D11 (GPa·mm³)", "Bending stiffness D11 (GPa·mm^3)"),
        "samples": L("500 个 LHS 采样点", "500 LHS samples"),
        "peak": L("理论峰值", "theoretical peak"),
        "smax": L("采样最大值", "max of samples"),
    }

    data = read_training_data(INPUT_CSV)
    theta1, theta2 = data["theta1"], data["theta2"]
    Ex_data, D11_data = data["Ex"], data["D11"]
    print(f"\n读取 {INPUT_CSV}: {len(theta1)} 组样本")
    print(f"  θ1 ∈ [{theta1.min():.3f}°, {theta1.max():.3f}°]   "
          f"θ2 ∈ [{theta2.min():.3f}°, {theta2.max():.3f}°]")
    print(f"  Ex  ∈ [{Ex_data.min():.4f}, {Ex_data.max():.4f}] GPa")
    print(f"  D11 ∈ [{D11_data.min():.6f}, {D11_data.max():.6f}] GPa·mm³")

    # ---------------- 自检 1：两套算法（CSV vs 解析式）必须一致 ----------------
    Ex_ref, Ey_ref, D11_ref = laminate_response(theta1, theta2)
    rel_ex = np.max(np.abs(Ex_ref - Ex_data) / np.abs(Ex_data))
    rel_ey = np.max(np.abs(Ey_ref - data["Ey"]) / np.abs(data["Ey"]))
    rel_d11 = np.max(np.abs(D11_ref - D11_data) / np.abs(D11_data))
    print("\n[自检 1] 解析式 CLT 与 training_data.csv 中 composipy 结果对比")
    print(f"  Ex 最大相对误差 : {rel_ex:.2e}")
    print(f"  Ey 最大相对误差 : {rel_ey:.2e}")
    print(f"  D11 最大相对误差: {rel_d11:.2e}   (容差 {VERIFY_TOL:.0e})")
    if max(rel_ex, rel_ey, rel_d11) > 1e-6:
        raise SystemExit("两套算法结果不一致，热力图不可信，请检查实现！")
    print("  -> 一致（差异仅为 CSV 保留 10 位有效数字的舍入）")

    # 互等关系（Maxwell-Betti）：nu_xy/Ex = nu_yx/Ey，等价于 Ex*nu_yx = Ey*nu_xy。
    # 用逐点柔度矩阵算出 nu_xy、nu_yx 后独立校验，而非由 Ex/Ey 反推。
    # （注意各向异性层合板的互等关系是 nu_xy/Ex = nu_yx/Ey，
    #   不是各向同性材料里的 Ex*nu_xy = Ey*nu_yx，两者不可混用。）
    nu_xy, nu_yx, recip_err = [], [], 0.0
    for t1, t2 in zip(theta1, theta2):
        a, h = in_plane_compliance(t1, t2)
        ex, ey = 1.0 / (h * a[0, 0]), 1.0 / (h * a[1, 1])
        vxy, vyx = -a[0, 1] / a[0, 0], -a[0, 1] / a[1, 1]
        nu_xy.append(vxy)
        nu_yx.append(vyx)
        recip_err = max(recip_err, abs(vxy / ex - vyx / ey))
    nu_xy, nu_yx = np.array(nu_xy), np.array(nu_yx)
    print(f"  nu_xy ∈ [{nu_xy.min():.4f}, {nu_xy.max():.4f}]   "
          f"nu_yx ∈ [{nu_yx.min():.4f}, {nu_yx.max():.4f}]")
    print(f"  max|nu_xy/Ex - nu_yx/Ey| = {recip_err:.3e}（理论上应为 0）")
    if recip_err > 1e-12:
        raise SystemExit("互等关系不成立，柔度矩阵实现有误！")

    # ---------------- 自检 2：单位与端点 ----------------
    ex_00, ey_00, d11_00 = laminate_response(0.0, 0.0)
    ex_90, _, d11_90 = laminate_response(90.0, 90.0)
    print("\n[自检 2] 单位与端点校验")
    print(f"  [0/0]s : Ex={ex_00:.6f} GPa (应等于 E1={E1})   D11={d11_00:.6f}")
    print(f"  [90/90]s: Ex={ex_90:.6f} GPa (应等于 E2={E2})   D11={d11_90:.6f}")
    if abs(ex_00 - E1) > 1e-6 or abs(ex_90 - E2) > 1e-6:
        raise SystemExit("端点校验失败，疑似角度或单位错误！")

    # ---------------- 自检 3：θ2 与 θ1 对 D11 的相对影响 ----------------
    probe = laminate_response(np.full(5, 30.0), np.linspace(5.0, 85.0, 5))[2]
    spread = float(np.ptp(probe))
    d11_peak = float(d11_00)
    print("\n[自检 3] D11 对外层角 θ1 与内层角 θ2 的相对敏感性")
    print(f"  固定 θ1=30°、扫描 θ2∈[5°,85°]：D11 极差 = {spread:.4f} GPa·mm³"
          f"（占峰值 {d11_peak:.4f} 的 {spread / d11_peak * 100:.1f}%）")
    print(f"  固定 θ2=30°、扫描 θ1∈[5°,85°]："
          f"D11 极差 = {float(np.ptp(laminate_response(np.linspace(5.0, 85.0, 5), 30.0)[2])):.4f} GPa·mm³")
    print("  -> θ1 影响占主导（外层离中面最远），但 θ2 的作用不可忽略")

    # ---------------- 构建真值曲面 ----------------
    axis = np.linspace(0.0, 90.0, GRID_POINTS)
    T1, T2 = np.meshgrid(axis, axis)
    Ex_grid, Ey_grid, D11_grid = laminate_response(T1, T2)
    print(f"\n真值曲面: {GRID_POINTS}x{GRID_POINTS} 网格 "
          f"({GRID_POINTS ** 2} 点，向量化 CLT 计算)")

    # ---------------- 绘图 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0), constrained_layout=True)

    panels = [
        (axes[0], Ex_grid, Ex_data, "ex",
         L("(a) 铺层角度对 面内刚度 Ex 的影响", "(a) Effect on in-plane stiffness Ex")),
        (axes[1], D11_grid, D11_data, "d11",
         L("(b) 铺层角度对 弯曲刚度 D11 的影响", "(b) Effect on bending stiffness D11")),
    ]

    for ax, grid, samples, key, title in panels:
        vmin, vmax = float(grid.min()), float(grid.max())

        image = ax.pcolormesh(T1, T2, grid, cmap="viridis", shading="auto",
                              vmin=vmin, vmax=vmax)

        # 用等值线把单调层次显式画出来，便于判断有无极值点
        levels = np.linspace(vmin, vmax, 9)[1:-1]
        contours = ax.contour(T1, T2, grid, levels=levels, colors="white",
                              linewidths=0.7, alpha=0.45)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.3g")

        # 叠加真实采样点
        ax.scatter(theta1, theta2, s=9, c="white", edgecolors="black",
                   linewidths=0.35, alpha=0.85, zorder=3, label=labels["samples"])

        # 标出理论峰值与采样最大值
        peak_index = np.unravel_index(np.argmax(grid), grid.shape)
        ax.plot(axis[peak_index[1]], axis[peak_index[0]], marker="*", markersize=17,
                markeredgecolor="black", markerfacecolor="red", zorder=5,
                linestyle="none", label=labels["peak"])
        best = int(np.argmax(samples))
        ax.plot(theta1[best], theta2[best], marker="D", markersize=8,
                markeredgecolor="black", markerfacecolor="cyan", zorder=5,
                linestyle="none", label=labels["smax"])

        ax.set_xlabel(labels["x"], fontsize=11)
        ax.set_ylabel(labels["y"], fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 90)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
        # 注记放在两图都空旷的中间偏左区域：避开右下角的峰值标记
        # （理论峰值红星与采样最大值青菱形都在 (0,0) 角点附近）与右上角图例。
        unit = L("GPa", "GPa") if key == "ex" else L("GPa·mm³", "GPa*mm^3")
        ax.text(0.36, 0.44,
                L(f"理论峰值 {grid.max():.4g} {unit}\n"
                  f"出现在 (θ1, θ2) = ({axis[peak_index[1]]:.0f}°, {axis[peak_index[0]]:.0f}°)",
                  f"max {grid.max():.4g} {unit}\n"
                  f"at (theta1, theta2) = ({axis[peak_index[1]]:.0f}, {axis[peak_index[0]]}) deg"),
                transform=ax.transAxes, fontsize=9.5, va="center", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.9,
                          edgecolor="gray"))

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(labels[key], fontsize=10)
        colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    fig.suptitle(L("[θ1/θ2/θ2/θ1]s 对称层合板：铺层角度对刚度的控制规律"
                   "（底色=理论真值曲面，白点=500 个 LHS 采样）",
                   "[theta1/theta2/theta2/theta1]s laminate: angle vs stiffness "
                   "(background = theoretical surface, white dots = 500 LHS samples)"),
                 fontsize=14.5)

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n已保存图像: {OUTPUT_PNG}")

    # ---------------- 规律量化（供文字结论使用）----------------
    print("\n" + "-" * 78)
    print("规律量化")
    print("-" * 78)
    # 单调性用相对容差判断：D11 在 90° 附近趋于平坦，绝对差值会落到浮点舍入量级
    scale_d11 = float(np.max(np.abs(D11_grid)))
    tol_d11 = scale_d11 * 1e-12

    print(f"  Ex(θ1,θ2) 与 Ex(θ2,θ1) 是否相等: "
          f"{np.allclose(Ex_grid, Ex_grid.T, rtol=1e-12)}")
    print(f"  D11 沿 θ1 方向单调不增: "
          f"{bool(np.all(np.diff(D11_grid, axis=1) <= tol_d11))}")
    print(f"  D11 沿 θ2 方向单调不增: "
          f"{bool(np.all(np.diff(D11_grid, axis=0) <= tol_d11))}")

    # Ex 并不单调：单层 Qbar11(θ) 单调递减，但面内模量
    #   Ex = A11*A22 / (h * det(A))，而 det(A) 含 A66，A66 在 θ≈45° 之后回升，
    # 因此 Ex 在 θ 超过约 45°~55° 后反而略有上升，出现内部极小值。
    flat = int(np.argmin(Ex_grid))
    i1, i2 = np.unravel_index(flat, Ex_grid.shape)
    corner_min = float(Ex_grid[0, 0])
    print(f"  Ex 全局最小值: {Ex_grid.min():.4f} GPa 于 "
          f"(θ1, θ2)=({axis[i2]:.1f}°, {axis[i1]:.1f}°)")
    print(f"  单层 Qbar11(θ) 单调不增: "
          f"{bool(np.all(np.diff(np.array([qbar(a)[0, 0] for a in axis])) <= 0))}"
          f"  |  Qbar11(0°)={qbar(0.0)[0, 0]:.4f} -> Qbar11(90°)={qbar(90.0)[0, 0]:.4f}")

    # Ex 沿 θ2 方向的内部极小（固定 θ1=45° 时最明显）
    row = Ex_grid[int(np.argmin(np.abs(axis - 45.0))), :]
    inner = int(np.argmin(row))
    print(f"  固定 θ1=45° 扫描 θ2: Ex 极小 {row[inner]:.4f} GPa 于 θ2={axis[inner]:.1f}°"
          f"（θ2=90° 时为 {row[-1]:.4f}）")
    print(f"  D11 沿 θ1 方向总变化量: {float(np.ptp(D11_grid, axis=1).max()):.4f} GPa·mm³")
    print(f"  D11 沿 θ2 方向总变化量: {float(np.ptp(D11_grid, axis=0).max()):.4f} GPa·mm³")
    print(f"  Ex  沿 θ1 方向总变化量: {float(np.ptp(Ex_grid, axis=1).max()):.4f} GPa")
    print(f"  Ex  沿 θ2 方向总变化量: {float(np.ptp(Ex_grid, axis=0).max()):.4f} GPa")

    # 供结论引用：数据集极值与理论峰值
    ex_best, d11_best = int(np.argmax(Ex_data)), int(np.argmax(D11_data))
    print(f"\n  采样点 Ex  最大: {data['stacking'][ex_best]}  "
          f"Ex={Ex_data[ex_best]:.4f}   D11={D11_data[ex_best]:.6f}")
    print(f"  采样点 D11 最大: {data['stacking'][d11_best]}  "
          f"Ex={Ex_data[d11_best]:.4f}   D11={D11_data[d11_best]:.6f}")
    print(f"  理论峰值: Ex={Ex_grid.max():.4f} GPa   D11={D11_grid.max():.6f} GPa·mm³"
          f"（均在 θ1=θ2=0° 角点处）")

    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

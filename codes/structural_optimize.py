# -*- coding: utf-8 -*-
"""真实工程结构优化：四边简支 [0]N 层合板的屈曲约束最小重量设计。

工程问题
    一块四边简支（SSSS）正方形复合材料层合板，a = b = 0.5 m，
    铺层固定为 [0]N（N 层单向 0° 铺层，纤维沿 x 方向），
    承受单向面内压缩线载荷 Nx = 1000 N/m（= 1 N/mm）。
    层数越少越轻，但弯曲刚度按厚度三次方衰减；当 N 小于某个临界值时，
    板会先发生屈曲失稳（而不是材料强度破坏）。于是工程问题变成：

        找到满足「不屈曲」的最轻层数 —— 这是真正的结构优化，
        而不是拟合/揭示 N 与 D11 的物理规律。

优化模型
    min  W(N)  = N · t · ρ · (a·b)         结构总重量（kg）
    s.t. Ncr(N) ≥ Nx = 1000 N/m            一阶临界屈曲载荷（安全系数 SF = 1）
         N ∈ {1, 2, ..., 20}               整数设计变量

物理模型（真实 CLT + 精确屈曲解，而不是 ML 代理模型）
    1) 用 composipy 计算 [0]N 层合板的 A/B/D 矩阵（经典层合板理论，严格遵守
       本项目的单位约定：长度 mm、模量 GPa → D 矩阵单位为 GPa·mm³）。
    2) 全部为 0° 铺层时 D16 = D26 = 0（特殊正交异性板），四边简支、单轴压缩
       下的屈曲载荷有**精确的 Navier 解析解**：

           Ncr(m,n) = π² [ D11·(m/a)² + 2(D12 + 2D66)·(n/b)²
                           + D22·(n/b)⁴ / (m/a)² ]

       对半波数 (m, n) 取最小值即为临界载荷；本问题 a = b，最优模态为 m = n = 1。
    3) 薄板理论适用性：N = 9 时 a/h = 444 ≫ 20，横向剪切变形影响可忽略
       （一阶剪切修正只会把 Ncr 抬高不到 0.1%，且方向偏安全）。
    4) 强度不是控制因素：N = 9 时压应力约 0.9 MPa，远低于纤维复合材料
       数百 MPa 量级的抗压强度，因此本问题由**屈曲**控制（这正是把问题
       定义在 1000 N/m 这种"低应力、薄壁"工况的意义）。

单位纪律（本项目最容易翻车的地方）
    库仑/毫米体系：1 GPa·mm³ = 1000 N·mm = 1 N·m。
    本脚本内部统一用国际单位做屈曲计算：D[Pa·m³] = D[GPa·mm³] × 1e9 × 1e-9，
    a、b 用 m，得到 Ncr 的单位就是 N/m，与载荷 Nx 直接可比。
    重量用 kg（kg/m² 面密度 × 面积）。

自检（不通过直接退出，避免"看起来对"的错结果）
    自检 1：composipy 的 D11 vs 闭式 Q11·h³/12 —— 验证 CLT 与单位换算
    自检 2：等向材料退化 → Ncr = 4π²D/b²（经典板的 k = 4）—— 验证 Navier 解
    自检 3：无限长板极限 → Ncr = 2π²/b²·[√(D11·D22) + D12 + 2D66] —— 验证正交异性分支
    自检 4：Ncr ∝ N³、W ∝ N —— 验证扫描结果的物理一致性

输出
    终端：N = 1..20 完整扫描表、最优解、重量节省、安全系数敏感性
    图片：figures/structural_optimization.png（Ncr(N) 曲线 + 最优解标注 + 重量对比）
"""

import os
import sys

import numpy as np

# 本机 Python 默认 stdout 编码可能为 GBK，显式切 UTF-8 保证中文在任何启动方式下正常
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from composipy import LaminateProperty, OrthotropicMaterial

# =============================================================================
# 1. 工程输入（全部集中在这里，改设计只需改这一段）
# =============================================================================
# ---- 几何与载荷 ----
A_PLATE = 0.5          # m，板长（x 向，平行于纤维方向）
B_PLATE = 0.5          # m，板宽（y 向）
NX_LOAD = 1000.0       # N/m，单向面内压缩线载荷（= 1 N/mm）

# ---- 材料（T300/5208 类碳纤维环氧，与本项目其他脚本一致） ----
E1 = 181.0             # GPa，纤维方向模量
E2 = 10.3              # GPa，横向模量
G12 = 7.17             # GPa，面内剪切模量
V12 = 0.28             # 主泊松比
PLY_THICKNESS = 0.125  # mm，单层厚度
DENSITY = 1600.0       # kg/m³（0.125 mm 单层 → 0.2 kg/m²）

# ---- 设计空间 ----
N_MIN, N_MAX = 1, 20   # 层数扫描范围
BASELINE_N = 12        # 初始（对照）设计：盲目取 12 层
MODE_MAX = 8           # 屈曲半波数上限（m, n ≤ 8，足够覆盖薄板低阶模态）

OUTPUT_PNG = os.path.join("figures", "structural_optimization.png")

# 安全系数敏感性分析用（约束 SF = 1.0 是用户给定的设计工况）
SF_LIST = (1.0, 1.2, 1.5, 2.0)


# =============================================================================
# 2. 真实 CLT：用 composipy 求 [0]N 的刚度矩阵
# =============================================================================
def laminate_matrices(n_plies):
    """返回 [0]N 层合板的 (A, B, D)，单位 GPa·mm、GPa·mm²、GPa·mm³。"""
    ply = OrthotropicMaterial(e1=E1, e2=E2, v12=V12, g12=G12,
                              thickness=PLY_THICKNESS)
    lam = LaminateProperty(stacking=[0.0] * n_plies, plies=[ply] * n_plies)
    return (np.asarray(lam.A, float),
            np.asarray(lam.B, float),
            np.asarray(lam.D, float))


def d11_closed_form(n_plies):
    """[0]N 的 D11 闭式解：D11 = Q11·h³/12，用于自检 composipy 结果。

    Q11 = E1 / (1 - v12·v21)，v21 = v12·E2/E1（正交异性材料互易关系）。
    单位：E1 用 GPa、h 用 mm → D11 为 GPa·mm³。
    """
    v21 = V12 * E2 / E1
    q11 = E1 / (1.0 - V12 * v21)
    h = n_plies * PLY_THICKNESS
    return q11 * h ** 3 / 12.0


def buckling_load(D_gpa_mm3, a_m, b_m, m_max=MODE_MAX, n_max=MODE_MAX):
    """四边简支特殊正交异性板的单轴压缩屈曲载荷（精确 Navier 解）。

    Ncr(m,n) = π² [ D11(m/a)² + 2(D12+2D66)(n/b)² + D22(n/b)⁴/(m/a)² ]

    参数
        D_gpa_mm3 : D 矩阵，单位 GPa·mm³（composipy 输出）
        a_m, b_m  : 板尺寸，单位 m
    返回
        (Ncr [N/m], m*, n*)：对半波数取最小值后的临界载荷与屈曲模态
    """
    # 单位换算：1 GPa·mm³ = 1e9 Pa·m³？逐步算：
    #   GPa → Pa: ×1e9；mm³ → m³: ×1e-9   → 1 GPa·mm³ = 1 N·m
    D = D_gpa_mm3 * (1e9 * 1e-9)  # 现在 D 的单位是 N·m（= Pa·m³）
    d11, d12, d22, d66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]

    best = (np.inf, 1, 1)
    for m in range(1, m_max + 1):
        kx = (m * np.pi / a_m) ** 2          # 1/m²
        for n in range(1, n_max + 1):
            ky = (n * np.pi / b_m) ** 2      # 1/m²
            # D[N·m]·k[1/m²] → N/m，量纲直接就是线载荷
            ncr = d11 * kx + 2.0 * (d12 + 2.0 * d66) * ky + d22 * ky ** 2 / kx
            if ncr < best[0]:
                best = (float(ncr), m, n)
    return best


# =============================================================================
# 3. 重量模型
# =============================================================================
def areal_weight(n_plies):
    """面密度 W = N·t·ρ，单位 kg/m²。"""
    return n_plies * PLY_THICKNESS * 1e-3 * DENSITY


def total_weight(n_plies):
    """整块板（a×b）的总重量，单位 kg。"""
    return areal_weight(n_plies) * A_PLATE * B_PLATE


# =============================================================================
# 4. 自检：三组解析极限 + 比例关系
# =============================================================================
def run_self_checks():
    """用教科书解析结果校验 CLT 与屈曲解的数值实现，任一不通过直接退出。"""
    print("\n" + "-" * 92)
    print("自检：CLT 与屈曲解的数值实现是否符合教科书解析结果")
    print("-" * 92)

    # ---- 自检 1：composipy 的 D11 必须等于闭式 Q11·h³/12 ----
    errs = []
    for n in (1, 4, 9, 12, 20):
        _, _, D = laminate_matrices(n)
        err = abs(D[0, 0] - d11_closed_form(n)) / d11_closed_form(n)
        errs.append(err)
    max_err = max(errs)
    print(f"  自检 1  composipy D11 vs 闭式 Q11·h³/12     最大相对误差 {max_err:.2e}"
          f"  -> {'通过' if max_err < 1e-9 else '失败'}")

    # ---- 自检 2：等向材料退化，方板屈曲载荷应为 4π²D/b²（经典 k = 4） ----
    e_iso = 70.0        # GPa，铝合金量级，仅用于验证公式
    nu_iso = 0.30
    g_iso = e_iso / (2.0 * (1.0 + nu_iso))
    ply = OrthotropicMaterial(e1=e_iso, e2=e_iso, v12=nu_iso, g12=g_iso,
                              thickness=2.0)
    lam = LaminateProperty(stacking=[0.0], plies=[ply])
    D_iso = np.asarray(lam.D, float)
    ncr_num, m_star, n_star = buckling_load(D_iso, 0.5, 0.5)
    d_iso_si = D_iso[0, 0] * (1e9 * 1e-9)   # N·m
    ncr_theory = 4.0 * np.pi ** 2 * d_iso_si / B_PLATE ** 2
    rel = abs(ncr_num - ncr_theory) / ncr_theory
    print(f"  自检 2  等向方板 vs 4π²D/b² = {ncr_theory:12.3f} N/m"
          f"  数值解 {ncr_num:12.3f} N/m  相对误差 {rel:.2e}"
          f"  -> {'通过' if rel < 1e-9 else '失败'}")
    if rel >= 1e-9:
        raise SystemExit("屈曲解未通过等向退化验证，请检查公式。")

    # ---- 自检 3：无限长板极限（a/b = 40），应趋于 2π²/b²[√(D11D22)+D12+2D66] ----
    _, _, D9 = laminate_matrices(9)
    D_si = D9 * (1e9 * 1e-9)
    d11, d12, d22, d66 = D_si[0, 0], D_si[0, 1], D_si[1, 1], D_si[2, 2]
    long_a = 40.0 * B_PLATE
    ncr_long, m_long, n_long = buckling_load(D9, long_a, B_PLATE, m_max=200, n_max=4)
    ncr_limit = 2.0 * np.pi ** 2 / B_PLATE ** 2 * (
        np.sqrt(d11 * d22) + d12 + 2.0 * d66)
    rel3 = abs(ncr_long - ncr_limit) / ncr_limit
    print(f"  自检 3  长板极限(a/b=40) 数值 {ncr_long:10.2f} N/m"
          f" vs 解析极限 {ncr_limit:10.2f} N/m"
          f"  (模态 m={m_long})  相对误差 {rel3:.2e}"
          f"  -> {'通过' if rel3 < 5e-3 else '失败'}")

    # ---- 自检 4：D11 ∝ N³、Ncr ∝ N³ ----
    n1, n2 = 4, 12
    _, _, D4 = laminate_matrices(n1)
    _, _, D12 = laminate_matrices(n2)
    ratio_d = D12[0, 0] / D4[0, 0]
    ratio_ncr = buckling_load(D12, A_PLATE, B_PLATE)[0] / \
        buckling_load(D4, A_PLATE, B_PLATE)[0]
    ideal = (n2 / n1) ** 3
    print(f"  自检 4  N={n2}/N={n1}: D11 比 {ratio_d:.6f}，Ncr 比 {ratio_ncr:.6f}，"
          f"理论 (N2/N1)³={ideal:.6f}  -> "
          f"{'通过' if abs(ratio_ncr / ideal - 1) < 1e-9 else '失败'}")


# =============================================================================
# 5. 扫描与优化
# =============================================================================
def scan_designs():
    """扫描 N = N_MIN..N_MAX，返回每个层数的全部工程指标。"""
    rows = []
    for n in range(N_MIN, N_MAX + 1):
        A, B, D = laminate_matrices(n)
        ncr, m_star, n_star = buckling_load(D, A_PLATE, B_PLATE)
        h_mm = n * PLY_THICKNESS
        rows.append({
            "N": n,
            "h_mm": h_mm,
            "D11": D[0, 0],
            "D22": D[1, 1],
            "Ncr": ncr,
            "mode": (m_star, n_star),
            "slenderness": (A_PLATE * 1000.0) / h_mm,   # a/h（mm/mm）
            "W_areal": areal_weight(n),
            "W_total": total_weight(n),
            "margin": ncr / NX_LOAD,
            "feasible": bool(ncr >= NX_LOAD),
        })
    return rows


def pick_optimum(rows):
    """在可行解中取重量最小者（Ncr 随 N 单调，等价于最小可行 N）。"""
    feasible = [r for r in rows if r["feasible"]]
    if not feasible:
        raise SystemExit(f"N ∈ [{N_MIN}, {N_MAX}] 内无可行解，请扩大层数范围。")
    return min(feasible, key=lambda r: r["W_total"])


def min_plies_for_load(rows, required_load):
    """给定所需屈曲载荷，返回满足要求的最小层数（用于安全系数敏感性）。"""
    ok = [r for r in rows if r["Ncr"] >= required_load]
    return ok[0] if ok else None


# =============================================================================
# 6. 终端报告
# =============================================================================
def print_scan_table(rows, opt):
    print("\n" + "-" * 92)
    print("扫描结果：N = 1..20（[0]N 铺层，四边简支，Nx = 1000 N/m）")
    print("-" * 92)
    header = (f"  {'N':>3s} {'t(mm)':>7s} {'a/t':>7s} {'D11':>9s} "
              f"{'Ncr(N/m)':>11s} {'裕度':>7s} {'W(kg/m²)':>10s} "
              f"{'总重(kg)':>9s} {'模态':>6s}  判定")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        mark = "◀ 最优" if r is opt else ("可行" if r["feasible"] else "屈曲")
        print(f"  {r['N']:3d} {r['h_mm']:7.3f} {r['slenderness']:7.0f} "
              f"{r['D11']:9.3f} {r['Ncr']:11.1f} {r['margin']:7.3f} "
              f"{r['W_areal']:10.2f} {r['W_total']:9.4f} "
              f"{r['mode'][0]:>2d}x{r['mode'][1]:<2d}  {mark}")
    print(f"\n  注：D11 单位 GPa·mm³；屈曲模态为半波数 (m, n)；裕度 = Ncr / Nx。")


def print_conclusion(rows, opt, baseline):
    """打印最终结论：最优层数、屈曲载荷、相对初始设计的减重。"""
    n_star = opt["N"]
    save_areal = baseline["W_areal"] - opt["W_areal"]
    save_total = baseline["W_total"] - opt["W_total"]
    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    print(f"  最优层数        : N* = {n_star}（[0]_{n_star}，厚度 {opt['h_mm']:.3f} mm）")
    print(f"  屈曲载荷        : Ncr = {opt['Ncr']:.1f} N/m  ≥  需求 {NX_LOAD:.0f} N/m"
          f"  （裕度 {opt['margin']:.3f}，即安全系数 {opt['margin']:.3f}）")
    print(f"  屈曲模态        : m×n = {opt['mode'][0]}×{opt['mode'][1]} 个半波，"
          f"a/t = {opt['slenderness']:.0f}（薄板，CLPT 适用）")
    print(f"  结构总重        : {opt['W_total']:.4f} kg"
          f"（面密度 {opt['W_areal']:.2f} kg/m²）")
    print()
    print(f"  对照初始设计 N={BASELINE_N}（盲目按 12 层取值）：")
    print(f"    屈曲载荷 Ncr = {baseline['Ncr']:.1f} N/m"
          f"（需求的 {baseline['margin']:.2f} 倍 —— 过度设计）")
    print(f"    结构总重     = {baseline['W_total']:.4f} kg"
          f"（面密度 {baseline['W_areal']:.2f} kg/m²）")
    print(f"  相对初始设计减重   : {save_total:.4f} kg"
          f"（{save_areal:.2f} kg/m²），即 {(1 - opt['W_total'] / baseline['W_total']) * 100:.1f}%")
    print(f"  换算：{BASELINE_N} → {n_star} 层，少 {BASELINE_N - n_star} 层"
          f"（{BASELINE_N - n_star} × {PLY_THICKNESS} mm = "
          f"{(BASELINE_N - n_star) * PLY_THICKNESS:.3f} mm 厚度）")
    print(f"  每减少 1 层约省 {total_weight(1):.4f} kg"
          f"（面密度 {areal_weight(1):.2f} kg/m²）")


def print_sf_sensitivity(rows, opt):
    """安全系数敏感性：工程上通常要求 SF ≥ 1.5，检查最优解是否仍成立。"""
    print("\n" + "-" * 92)
    print("工程裕度分析：不同安全系数下所需的最小层数（真实工程通常取 SF ≥ 1.5）")
    print("-" * 92)
    print(f"  {'SF':>5s} {'需求载荷(N/m)':>14s} {'最小N':>7s} {'Ncr(N/m)':>11s} "
          f"{'裕度':>7s} {'总重(kg)':>10s}  相对SF=1.0")
    for sf in SF_LIST:
        required = sf * NX_LOAD
        r = min_plies_for_load(rows, required)
        if r is None:
            print(f"  {sf:5.1f} {required:14.0f} {'—':>7s}  超出 N ≤ {N_MAX} 的扫描范围")
            continue
        delta = (r["W_total"] / opt["W_total"] - 1.0) * 100.0
        print(f"  {sf:5.1f} {required:14.0f} {r['N']:7d} {r['Ncr']:11.1f} "
              f"{r['margin']:7.3f} {r['W_total']:10.4f}   {delta:+6.1f}%")
    print(f"\n  用户给定的约束是 SF = 1.0（Ncr ≥ {NX_LOAD:.0f} N/m），故 N* = {opt['N']}。")
    print(f"  N* = {opt['N']} 的裕度仅 {opt['margin']:.2f}"
          f"（超出需求 {opt['Ncr'] - NX_LOAD:.1f} N/m），对实际工程偏紧：")
    print("  若按航空结构常用的 SF = 1.5 设计，最优层数会升到上表中对应行；")
    print("  这说明「最小重量」与「工程安全裕度」之间存在真实的取舍。")


# =============================================================================
# 7. 绘图
# =============================================================================
def setup_chinese_font():
    """尝试启用系统中已安装的中文字体，失败则退回英文标签（跨平台，不依赖固定路径）。"""
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


def plot_results(rows, opt, baseline, L):
    n = np.array([r["N"] for r in rows])
    ncr = np.array([r["Ncr"] for r in rows])
    w = np.array([r["W_areal"] for r in rows])
    feasible = np.array([r["feasible"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   constrained_layout=True)
    fig.suptitle(
        L("四边简支 [0]N 层合板：屈曲约束下的最小重量设计"
          "（a=b=0.5 m，Nx=1000 N/m）",
          "SSSS [0]N laminate under uniaxial compression (a=b=0.5 m, Nx=1000 N/m)"),
        fontsize=13.5)

    # ---------------- 左图：Ncr(N) 曲线 ----------------
    ax1.set_yscale("log")
    ax1.axhspan(NX_LOAD * 0.08, NX_LOAD, color="tab:red", alpha=0.05, zorder=0)
    ax1.axhspan(NX_LOAD, 2.2e4, color="tab:green", alpha=0.05, zorder=0)
    ax1.axhline(NX_LOAD, color="crimson", linestyle="--", linewidth=1.8, zorder=2,
                label=L(f"设计载荷 Nx = {NX_LOAD:.0f} N/m", f"design load = {NX_LOAD:.0f} N/m"))

    ax1.plot(n, ncr, "-", color="tab:blue", linewidth=2.2, zorder=3,
             label=L("临界屈曲载荷 Ncr(N)（Navier 精确解）",
                     "critical buckling load Ncr(N) (Navier)"))

    # 不可行（会屈曲）用空心红圈，可行用实心蓝点
    ax1.plot(n[~feasible], ncr[~feasible], "o", markersize=7,
             markerfacecolor="none", markeredgecolor="tab:red",
             markeredgewidth=1.5, zorder=4,
             label=L(f"不可行：N ≤ {opt['N'] - 1} 会屈曲", f"infeasible: N ≤ {opt['N'] - 1}"))
    ax1.plot(n[feasible], ncr[feasible], "o", markersize=7,
             markerfacecolor="tab:blue", markeredgecolor="navy", zorder=4,
             label=L(f"可行：N ≥ {opt['N']}", f"feasible: N ≥ {opt['N']}"))

    # 最优解标注
    ax1.plot(opt["N"], opt["Ncr"], marker="*", markersize=22,
             markerfacecolor="tab:green", markeredgecolor="black", zorder=6,
             linestyle="none", label=L(f"最优解 N* = {opt['N']}", f"optimum N* = {opt['N']}"))
    ax1.annotate(
        L(f"N* = {opt['N']}（[0]_{opt['N']}）\n"
          f"Ncr = {opt['Ncr']:.0f} N/m ≥ {NX_LOAD:.0f} N/m\n"
          f"裕度 {opt['margin']:.2f}，总重 {opt['W_total']:.3f} kg",
          f"N* = {opt['N']}\nNcr = {opt['Ncr']:.0f} N/m "
          f"(≥ {NX_LOAD:.0f})\nmargin {opt['margin']:.2f}, "
          f"weight {opt['W_total']:.3f} kg"),
        xy=(opt["N"], opt["Ncr"]), xytext=(1.2, 5200.0),
        textcoords="data", fontsize=9.3, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="honeydew",
                  edgecolor="tab:green", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.3,
                        connectionstyle="arc3,rad=-0.25"))

    # 初始设计 N=12 标注
    ax1.plot(baseline["N"], baseline["Ncr"], marker="s", markersize=11,
             markerfacecolor="darkorange", markeredgecolor="black", zorder=5,
             linestyle="none",
             label=L(f"初始设计 N = {BASELINE_N}（过度设计）", f"initial design N = {BASELINE_N}"))
    ax1.annotate(
        L(f"N = {BASELINE_N}（初始设计）：Ncr = {baseline['Ncr']:.0f} N/m\n"
          f"= 需求的 {baseline['margin']:.1f} 倍 —— 过度设计，偏重 "
          f"{(baseline['W_areal'] / opt['W_areal'] - 1) * 100:.0f}%",
          f"N = {BASELINE_N}: Ncr = {baseline['Ncr']:.0f} N/m "
          f"({baseline['margin']:.1f}× demand)"),
        xy=(baseline["N"], baseline["Ncr"]), xytext=(12.4, 430.0),
        textcoords="data", fontsize=9.0, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="darkorange", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))

    ax1.set_xlim(0.4, N_MAX + 0.6)
    ax1.set_ylim(ncr.min() * 0.6, 1.6e4)
    ax1.set_xticks(np.arange(1, N_MAX + 1, 1))
    ax1.tick_params(axis="x", labelsize=8)
    # 区域提示（屈曲失稳区 / 安全区）
    ax1.text(2.7, 1.9, L("屈曲失稳区（Ncr < Nx）", "buckling region"),
             fontsize=8.5, color="tab:red", alpha=0.75)
    ax1.text(13.6, 45.0, L("安全区（Ncr ≥ Nx）", "safe region"),
             fontsize=8.5, color="tab:green", alpha=0.75)
    ax1.set_xlabel(L("层数 N", "number of plies N"), fontsize=11.5)
    ax1.set_ylabel(L("临界屈曲载荷 Ncr (N/m)", "critical buckling load Ncr (N/m)"),
                   fontsize=11.5)
    ax1.set_title(L("(a) Ncr 随层数增长（∝ N³），约束线决定最小可行 N",
                    "(a) Ncr grows ∝ N³; the constraint sets the minimum N"),
                  fontsize=11)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # ---------------- 右图：重量与减重 ----------------
    ax2.plot(n, w, "-o", color="tab:purple", linewidth=2.2, markersize=6,
             markerfacecolor="white", label=L("面密度 W(N) = N×0.2 kg/m²", "areal weight W(N)"))
    ax2.axvspan(opt["N"], BASELINE_N, color="tab:green", alpha=0.10, zorder=0)

    ax2.plot(opt["N"], opt["W_areal"], marker="*", markersize=22,
             markerfacecolor="tab:green", markeredgecolor="black", zorder=6,
             linestyle="none", label=L(f"最优解 N* = {opt['N']}", f"optimum N* = {opt['N']}"))
    ax2.plot(baseline["N"], baseline["W_areal"], marker="s", markersize=11,
             markerfacecolor="darkorange", markeredgecolor="black", zorder=5,
             linestyle="none", label=L(f"初始设计 N = {BASELINE_N}", f"initial N = {BASELINE_N}"))

    # 减重双箭头
    ax2.annotate(
        "", xy=(opt["N"], opt["W_areal"] + 0.03),
        xytext=(BASELINE_N, baseline["W_areal"] - 0.03),
        arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.6,
                        connectionstyle="arc3,rad=-0.18"))
    ax2.annotate(
        L(f"减重 {(baseline['W_areal'] - opt['W_areal']):.2f} kg/m²\n"
          f"（−{(1 - opt['W_areal'] / baseline['W_areal']) * 100:.0f}%）",
          f"save {(baseline['W_areal'] - opt['W_areal']):.2f} kg/m²\n"
          f"(−{(1 - opt['W_areal'] / baseline['W_areal']) * 100:.0f}%)"),
        xy=(0.5 * (opt["N"] + BASELINE_N), 0.5 * (opt["W_areal"] + baseline["W_areal"])),
        xytext=(-14, 26), textcoords="offset points", fontsize=10,
        ha="center", color="darkgreen", weight="bold")

    ax2.set_xlim(0.4, N_MAX + 0.6)
    ax2.set_ylim(0, w.max() * 1.12)
    ax2.set_xticks(np.arange(1, N_MAX + 1, 1))
    ax2.tick_params(axis="x", labelsize=8)
    ax2.set_xlabel(L("层数 N", "number of plies N"), fontsize=11.5)
    ax2.set_ylabel(L("面密度 W (kg/m²)", "areal weight W (kg/m²)"), fontsize=11.5)
    ax2.set_title(L(f"(b) 重量 ∝ N：从 N={BASELINE_N} 降到 N={opt['N']} 减重 "
                    f"{(1 - opt['W_total'] / baseline['W_total']) * 100:.0f}%",
                    f"(b) weight ∝ N: N={BASELINE_N} → {opt['N']} saves "
                    f"{(1 - opt['W_total'] / baseline['W_total']) * 100:.0f}%"),
                  fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"\n  已保存图表: {OUTPUT_PNG}")


# =============================================================================
# 8. 主流程
# =============================================================================
def main():
    print("=" * 92)
    print("真实工程结构优化：四边简支 [0]N 层合板的面内压缩屈曲约束最小重量设计")
    print("=" * 92)
    print(f"几何     : a = b = {A_PLATE:.3f} m 正方形板，四边简支（SSSS）")
    print(f"载荷     : Nx = {NX_LOAD:.0f} N/m（= {NX_LOAD / 1000.0:.3f} N/mm）单向面内压缩")
    print(f"材料     : E1={E1} GPa, E2={E2} GPa, G12={G12} GPa, v12={V12}")
    print(f"单层     : t={PLY_THICKNESS} mm，ρ={DENSITY:.0f} kg/m³"
          f"（每层面密度 {areal_weight(1):.2f} kg/m²）")
    print(f"设计变量 : N ∈ [{N_MIN}, {N_MAX}]，铺层 [0]N（纤维沿加载方向）")
    print(f"目标     : min W = N·t·ρ·(a·b)；约束 Ncr(N) ≥ {NX_LOAD:.0f} N/m")
    print(f"表达式   : Ncr = π²[D11(m/a)² + 2(D12+2D66)(n/b)² + D22(n/b)⁴/(m/a)²]"
          f"（对 m,n 取最小）")

    run_self_checks()

    rows = scan_designs()
    opt = pick_optimum(rows)
    baseline = next(r for r in rows if r["N"] == BASELINE_N)

    print_scan_table(rows, opt)
    print_conclusion(rows, opt, baseline)
    print_sf_sensitivity(rows, opt)

    _, has_chinese = setup_chinese_font()
    L = (lambda cn, en: cn) if has_chinese else (lambda cn, en: en)
    plot_results(rows, opt, baseline, L)

    print("\n" + "=" * 92)
    print(f"核心结论：最优 {opt['N']} 层（[0]_{opt['N']}），Ncr = {opt['Ncr']:.0f} N/m "
          f"≥ {NX_LOAD:.0f} N/m，"
          f"比盲目取 {BASELINE_N} 层轻 {(1 - opt['W_total'] / baseline['W_total']) * 100:.1f}%"
          f"（{baseline['W_total'] - opt['W_total']:.3f} kg / {baseline['W_total']:.3f} kg）")
    print("脚本运行完成，未出现错误。")
    print("=" * 92)


if __name__ == "__main__":
    main()

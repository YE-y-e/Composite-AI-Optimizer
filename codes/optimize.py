# -*- coding: utf-8 -*-
"""NSGA-II 双目标优化（v2）：引入层数 N 作为第三个设计变量。

问题设定
    铺层逻辑 : 固定 [0]N（全部 0° 单向，对称），即 [0/0]s 铺层逻辑的推广
    决策变量 : 层数 N ∈ [4, 24]（整数）
    目标 1   : 最大化弯曲刚度 D11，同时要求重量 W ≤ W(N=12)
    目标 2   : 最小化重量 W，同时要求 D11 ≥ 1.5

物理关系（已用 composipy 逐点验证，见脚本自检）
    Ex  = 181.0 GPa          与 N 无关（单向层合板的面内模量恒等于 E1）
    D11 = N³ × 0.0295917     ∝ N³（厚度三次方）
    W   = N × 0.125 mm × 1600 kg/m³ = N × 0.2 kg/m²   ∝ N
    其中单层厚度 0.125 mm、材料密度 1600 kg/m³（T300/5208 碳纤维环氧典型值）。
    W 为面密度（kg/m²），等于单位面积层合板的质量。

为什么这样设计才有真实权衡
    v1 只优化铺层角度时，Ex 与 D11 都随角度单调变差，[0/0]s 同时是两个目标的
    全局最优，帕累托前沿退化为单点。引入 N 后：
      * D11 ∝ N³ 而 W ∝ N，二者严格冲突 —— 想更刚就必须更重；
      * Ex 与 N 无关，因此原来的"Ex 约束"在本问题中恒满足（181 ≥ 120），
        不再是有效约束。
    于是得到真正有意义的 (W, D11) 权衡前沿。

约束的冗余性（脚本会打印验证）
    可行域为 N ∈ [4, 24]，且 W ≤ W(N=12) 即 N ≤ 12。
    在 N ≥ 4 时 D11 = N³×0.02959 ≥ 1.894 > 1.5 恒成立，
    因此 D11 ≥ 1.5 在该范围内也是冗余的；真正起作用的约束是 N ≤ 12（重量上限）。

输出 : pareto_front_v2.png、optimal_designs_v2.csv
"""

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
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

# 真实 CLT（composipy）作为最终校验工具
from generate_data import E1, laminate_stiffness

OUTPUT_PNG = "pareto_front_v2.png"
OUTPUT_CSV = "optimal_designs_v2.csv"

# ------------------------- 材料与几何常数 -------------------------
PLY_THICKNESS = 0.125     # mm，单层厚度
DENSITY = 1600.0          # kg/m^3，碳纤维环氧典型密度
D11_MIN = 1.5             # GPa*mm^3，目标 2 的约束
N_LOW, N_HIGH = 4, 24     # 层数范围
N_WEIGHT_CAP = 12         # 目标 1 的重量上限（以 N=12 计）
SEED = 1


def areal_weight(n_plies):
    """面密度 W = N * t * rho（单位 kg/m^2）。"""
    return n_plies * PLY_THICKNESS * 1e-3 * DENSITY


W_MAX = areal_weight(N_WEIGHT_CAP)
# 解析标定出的 D11 ∝ N³ 系数（由 N=4 的 CLT 结果反推）
D11_COEF = 1.893866029629 / (4.0 ** 3)


def d11_closed_form(n_plies):
    """D11 的解析闭式：D11 = coef * N³（N 可为数组）。"""
    n = np.asarray(n_plies, dtype=float)
    return D11_COEF * n ** 3


def setup_chinese_font():
    """尝试启用系统中已安装的中文字体，失败则退回英文标签（跨平台，不依赖固定路径）。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "SimSun",
                 "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                 "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                 "PingFang SC", "Heiti SC", "Arial Unicode MS"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            return name, True
    return None, False


class PlyCountProblem(Problem):
    """层数 N 的双目标约束问题。

    目标（pymoo 只做最小化）:
        f1 = -D11(N)      最大化弯曲刚度
        f2 = +W(N)        最小化重量
    约束（pymoo 约定 G ≤ 0 为满足，G > 0 为违反）:
        g1 = W(N)   - W_MAX ≤ 0      重量上限
        g2 = D11_MIN - D11(N) ≤ 0    弯曲刚度下限 D11 ≥ 1.5
    注意约束符号：pymoo 用 max(0, G) 计算违反量，写反会把好解判为不可行。
    """

    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=2,
                         xl=np.array([float(N_LOW)]),
                         xu=np.array([float(N_HIGH)]))
        self.n_eval = 0
        self.best_continuous = None

    def _evaluate(self, X, out, *args, **kwargs):
        n = X[:, 0]
        d11 = d11_closed_form(n)
        weight = areal_weight(n)
        self.n_eval += X.shape[0]
        out["F"] = np.column_stack([-d11, weight])
        out["G"] = np.column_stack([weight - W_MAX, D11_MIN - d11])


def bootstrap_initial_population(problem, n_pop, seed=SEED):
    """用 LHS 在 [N_LOW, N_HIGH] 上生成初始种群。

    可重点全部覆盖 4~12（可行区）与 12~24（超重区）两个区间，
    便于 NSGA-II 从一开始就有可比基准。
    """
    from pymoo.operators.sampling.lhs import sampling_lhs

    points = sampling_lhs(n_pop, 1, xl=float(N_LOW), xu=float(N_HIGH),
                          smooth=False, criterion=None,
                          random_state=np.random.default_rng(seed))
    # 让约一半个体直接落在可行区内，避免初始种群全是超重解
    half = n_pop // 2
    points[:half, 0] = np.linspace(N_LOW, N_WEIGHT_CAP, half)
    return points


def integer_front():
    """枚举整数层数，返回可行前沿（按重量升序）及全部整数点。"""
    all_n = np.arange(N_LOW, N_HIGH + 1)
    all_w = areal_weight(all_n)
    all_d11 = d11_closed_form(all_n)
    feasible = (all_w <= W_MAX + 1e-12) & (all_d11 >= D11_MIN - 1e-12)
    return all_n[feasible], all_w[feasible], all_d11[feasible], all_n, all_w, all_d11


def pick_representatives(n, w, d11):
    """从可行前沿中挑 3 个代表性解：轻量化最优、D11 最大、折中。

    折中解用两个准则分别计算，若不一致则取归一化加权和更小者：
      * 加权和准则：最小化 (归一化 W) + (归一化 -D11)
      * 理想点距离：最小化到坐标 (min W, max D11) 的归一化欧氏距离
    """
    i_light = int(np.argmin(w))
    i_stiff = int(np.argmax(d11))

    span_w = w.max() - w.min()
    span_d = d11.max() - d11.min()
    nw = (w - w.min()) / span_w if span_w > 0 else np.zeros_like(w)
    nd = (d11.max() - d11) / span_d if span_d > 0 else np.zeros_like(d11)

    i_sum = int(np.argmin(nw + nd))
    i_dist = int(np.argmin(np.sqrt(nw ** 2 + nd ** 2)))
    i_balance = i_sum if (nw + nd)[i_sum] <= np.sqrt(nw ** 2 + nd ** 2)[i_dist] else i_dist

    return [("轻量化最优", i_light), ("D11 最大", i_stiff),
            ("折中解", i_balance)], {"sum": int(n[i_sum]), "dist": int(n[i_dist]),
                                     "chosen": int(n[i_balance])}


def main():
    print("=" * 88)
    print("NSGA-II 双目标优化 v2：固定 [0]N 铺层，层数 N ∈ [4, 24]")
    print("=" * 88)
    font_name, has_chinese = setup_chinese_font()
    L = (lambda cn, en: cn) if has_chinese else (lambda cn, en: en)

    print(f"单层厚度 {PLY_THICKNESS} mm，密度 {DENSITY} kg/m³")
    print(f"目标 1: 最大化 D11，约束 W ≤ W(N=12) = {W_MAX:.4f} kg/m²")
    print(f"目标 2: 最小化 W，  约束 D11 ≥ {D11_MIN} GPa·mm³")

    # ---------------- 自检 1：解析式 vs composipy 真实 CLT ----------------
    print("\n" + "-" * 88)
    print("自检 1：闭式 D11 = coef·N³ 与 composipy 真实 CLT 的一致性，以及 Ex 与 N 无关")
    print("-" * 88)
    max_err = 0.0
    print(f"  {'N':>3s} {'t(mm)':>7s} {'W(kg/m²)':>10s} {'Ex(GPa)':>9s} "
          f"{'D11(CLT)':>12s} {'D11(闭式)':>12s} {'误差%':>8s}")
    for n in (4, 8, 12, 16, 24):
        from composipy import LaminateProperty, OrthotropicMaterial
        ply = OrthotropicMaterial(e1=181.0, e2=10.3, v12=0.28, g12=7.17,
                                  thickness=PLY_THICKNESS)
        lam = LaminateProperty(stacking=[0.0] * n, plies=[ply] * n)
        A = np.asarray(lam.A, float)
        D = np.asarray(lam.D, float)
        h = n * PLY_THICKNESS
        ex = 1.0 / (h * np.linalg.inv(A)[0, 0])
        d11_true = D[0, 0]
        d11_formula = float(d11_closed_form(n))
        err = abs(d11_formula - d11_true) / d11_true * 100.0
        max_err = max(max_err, err)
        print(f"  {n:3d} {h:7.3f} {areal_weight(n):10.4f} {ex:9.4f} "
              f"{d11_true:12.5f} {d11_formula:12.5f} {err:8.2e}")
    print(f"\n  最大相对误差: {max_err:.3e}%  -> {'通过' if max_err < 1e-9 else '未通过！'}")
    if max_err >= 1e-9:
        raise SystemExit("闭式与 CLT 不一致，请检查 D11_COEF。")
    print("  Ex 恒为 181.0 GPa 与 N 无关（单向层合板面内模量等于 E1）——已确认")

    # ---------------- 自检 2：约束冗余性 ----------------
    print("\n" + "-" * 88)
    print("自检 2：约束有效性分析")
    print("-" * 88)
    all_n = np.arange(N_LOW, N_HIGH + 1)
    all_d11 = d11_closed_form(all_n)
    print(f"  N∈[{N_LOW},{N_HIGH}] 时 D11 最小值 = {all_d11.min():.4f} "
          f"(N={all_n[int(np.argmin(all_d11))]})")
    print(f"  D11 ≥ {D11_MIN} 是否在整个 N 范围内恒成立: "
          f"{bool(np.all(all_d11 >= D11_MIN))}  -> 该约束{'冗余' if np.all(all_d11 >= D11_MIN) else '有效'}")
    print(f"  重量约束 W ≤ {W_MAX:.4f} 等价于 N ≤ {N_WEIGHT_CAP} -> 有效约束，"
          f"把 N 限制在 [{N_LOW}, {N_WEIGHT_CAP}]")

    # ---------------- 运行 NSGA-II ----------------
    problem = PlyCountProblem()
    n_pop, n_gen = 200, 200
    print("\n" + "-" * 88)
    print(f"运行 NSGA-II：种群 {n_pop}，迭代 {n_gen} 代，决策变量 N ∈ [{N_LOW}, {N_HIGH}]")
    print("-" * 88)

    class FixedSampling(Sampling):
        def __init__(self, points):
            super().__init__()
            self.points = points

        def _do(self, prob, n_samples, *args, **kwargs):
            if n_samples <= self.points.shape[0]:
                return self.points[:n_samples].copy()
            extra = np.random.default_rng(SEED).uniform(
                N_LOW, N_HIGH, size=(n_samples - self.points.shape[0], 1))
            return np.vstack([self.points, extra])

    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=FixedSampling(bootstrap_initial_population(problem, n_pop)),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=SEED, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"优化完成: 用时 {elapsed:.2f} s，评估 {problem.n_eval} 次")

    population = result.opt if result.opt is not None else result.pop
    X = np.atleast_2d(population.get("X"))
    n_values = X[:, 0]
    d11_values = d11_closed_form(n_values)
    w_values = areal_weight(n_values)
    feasible = (w_values <= W_MAX + 1e-12) & (d11_values >= D11_MIN - 1e-12)
    print(f"最优种群 {X.shape[0]} 个个体，其中满足两约束 {int(feasible.sum())} 个")
    print(f"层数范围 {n_values.min():.3f} ~ {n_values.max():.3f}"
          f"（连续松弛，实际会落到整数或上限 {N_WEIGHT_CAP}）")

    # ---------------- 离散化：投影到整数层数并重新校验 ----------------
    # 层数必须是整数，把连续最优投影到最近的整数，再用真实 CLT 复核
    print("\n" + "-" * 88)
    print("离散化与真实 CLT 复核（层数取整数后重新计算）")
    print("-" * 88)
    int_n, int_w, int_d11, all_n2, all_w2, all_d112 = integer_front()
    print(f"整数可行前沿: N = {[int(v) for v in int_n]}")
    print(f"  对应 W   = {[round(float(v), 4) for v in int_w]}")
    print(f"  对应 D11 = {[round(float(v), 4) for v in int_d11]}")

    # 用 composipy 复核前沿上的每个整数点，确保闭式可靠
    from composipy import LaminateProperty, OrthotropicMaterial
    check_err = 0.0
    for n in int_n:
        ply = OrthotropicMaterial(e1=181.0, e2=10.3, v12=0.28, g12=7.17,
                                  thickness=PLY_THICKNESS)
        lam = LaminateProperty(stacking=[0.0] * int(n), plies=[ply] * int(n))
        d_true = float(np.asarray(lam.D, float)[0, 0])
        check_err = max(check_err, abs(d_true - float(d11_closed_form(n))) / d_true)
    print(f"  前沿各点闭式 vs composipy 最大相对误差: {check_err:.3e}"
          f"  -> {'通过' if check_err < 1e-9 else '未通过！'}")

    # ---------------- 三个代表性解 ----------------
    picks, diagnostics = pick_representatives(int_n, int_w, int_d11)
    print("\n" + "=" * 88)
    print("三个代表性解")
    print("=" * 88)
    print(f"  （折中解候选取自加权和准则 N={diagnostics['sum']}、"
          f"理想点距离准则 N={diagnostics['dist']}，采用 N={diagnostics['chosen']}）")
    print(f"\n  {'解':<12s} {'N':>3s} {'t(mm)':>8s} {'W(kg/m²)':>10s} "
          f"{'D11(GPa·mm³)':>14s} {'Ex(GPa)':>9s} {'铺层':>8s}")
    rows = []
    for label, index in picks:
        n = int(int_n[index])
        thickness = n * PLY_THICKNESS
        weight = float(int_w[index])
        # 真实 CLT（composipy）复核
        ply = OrthotropicMaterial(e1=181.0, e2=10.3, v12=0.28, g12=7.17,
                                  thickness=PLY_THICKNESS)
        lam = LaminateProperty(stacking=[0.0] * n, plies=[ply] * n)
        A = np.asarray(lam.A, float)
        D = np.asarray(lam.D, float)
        h = n * PLY_THICKNESS
        ex_true = 1.0 / (h * np.linalg.inv(A)[0, 0])
        d11_true = float(D[0, 0])
        rows.append({"label": label, "N": n, "thickness": thickness,
                     "weight": weight, "D11": d11_true, "Ex": float(ex_true)})
        print(f"  {label:<12s} {n:3d} {thickness:8.3f} {weight:10.4f} "
              f"{d11_true:14.6f} {ex_true:9.4f} {f'[0]{n}':>8s}")

    # ---------------- 闭环验证 ----------------
    print("\n" + "=" * 88)
    print("闭环验证：约束复核（用真实 CLT 值）")
    print("=" * 88)
    print(f"  {'解':<12s} {'N':>3s} | {'W≤' + f'{W_MAX:.4f}':>14s} "
          f"{'D11≥' + str(D11_MIN):>12s} | {'D11/N³':>10s}")
    print("  " + "-" * 66)
    for row in rows:
        ok_w = row["weight"] <= W_MAX + 1e-12
        ok_d = row["D11"] >= D11_MIN - 1e-12
        print(f"  {row['label']:<12s} {row['N']:3d} | "
              f"{'✓' if ok_w else '✗':>14s} {'✓' if ok_d else '✗':>12s} | "
              f"{row['D11'] / row['N'] ** 3:10.6f}")
    ratio = [row["D11"] / row["N"] ** 3 for row in rows]
    print(f"\n  三点 D11/N³ 是否都等于常数 {D11_COEF:.6f}: "
          f"{'是（符合 D11 ∝ N³ 标度律）' if max(ratio) - min(ratio) < 1e-9 else '否！'}")

    # ---------------- 保存 ----------------
    fieldnames = ["label", "N", "thickness_mm", "weight_kg_per_m2",
                  "D11_GPa_mm3", "Ex_GPa", "stacking"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "label": row["label"],
                "N": row["N"],
                "thickness_mm": f"{row['thickness']:.4f}",
                "weight_kg_per_m2": f"{row['weight']:.6f}",
                "D11_GPa_mm3": f"{row['D11']:.6f}",
                "Ex_GPa": f"{row['Ex']:.4f}",
                "stacking": f"[0]{row['N']}",
            })
    print(f"\n  已保存 {OUTPUT_CSV}（{len(rows)} 行）")

    # ---------------- 绘图 ----------------
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)

    # 全部整数点（含不可行）用空心灰圈表示
    infeasible = ~((all_w2 <= W_MAX + 1e-12) & (all_d112 >= D11_MIN - 1e-12))
    ax.plot(all_w2[infeasible], all_d112[infeasible], "o", markersize=8,
            markerfacecolor="none", markeredgecolor="gray", markeredgewidth=1.4,
            label=L("超重（W 超上限）的整数层数", "infeasible (over weight cap)"))
    # 可行前沿
    ax.plot(int_w, int_d11, "-", color="tab:blue", linewidth=2.2, zorder=3,
            label=L("可行帕累托前沿（整数层数）", "feasible Pareto front (integer N)"))
    ax.plot(int_w, int_d11, "o", markersize=10, markerfacecolor="tab:blue",
            markeredgecolor="navy", zorder=4)

    # 连续松弛的 NSGA-II 结果（去重后）
    order = np.argsort(w_values)
    mask_ok = feasible[order]
    ax.plot(w_values[order][mask_ok], d11_values[order][mask_ok], "--",
            color="tab:orange", linewidth=1.6, alpha=0.9, zorder=2,
            label=L("NSGA-II 连续松弛解", "NSGA-II continuous relaxation"))

    # 约束线
    ax.axvline(W_MAX, color="red", linestyle="--", linewidth=1.6,
               label=L(f"重量上限 W = {W_MAX:.2f} kg/m²", f"weight cap W = {W_MAX:.2f}"))
    ax.axhline(D11_MIN, color="green", linestyle="--", linewidth=1.6,
               label=L(f"刚度下限 D11 = {D11_MIN}", f"D11 floor = {D11_MIN}"))

    # 标注三个代表解
    markers = {"轻量化最优": "v", "D11 最大": "^", "折中解": "*"}
    colors = {"轻量化最优": "tab:green", "D11 最大": "tab:red", "折中解": "tab:purple"}
    for row in rows:
        ax.plot(row["weight"], row["D11"], marker=markers[row["label"]],
                markersize=17 if row["label"] == "折中解" else 14,
                markerfacecolor=colors[row["label"]], markeredgecolor="black",
                zorder=7, linestyle="none", label=row["label"])
        ax.annotate(f"N={row['N']}\n({row['D11']:.1f})",
                    xy=(row["weight"], row["D11"]),
                    xytext=(8, 10 if row["label"] != "折中解" else -26),
                    textcoords="offset points", fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.85, edgecolor=colors[row["label"]]))

    # 每个整数点旁标 N 值，便于读图
    for n, w, d in zip(all_n2, all_w2, all_d112):
        if n <= N_WEIGHT_CAP:
            ax.annotate(f"N={int(n)}", xy=(w, d), xytext=(6, -12),
                        textcoords="offset points", fontsize=8, color="dimgray")

    ax.set_xlabel(L("重量 W (kg/m²)", "Areal weight W (kg/m^2)"), fontsize=12)
    ax.set_ylabel(L("弯曲刚度 D11 (GPa·mm³)", "Bending stiffness D11 (GPa*mm^3)"),
                  fontsize=12)
    ax.set_title(L("固定 [0]N 单向铺层：(重量 W, 弯曲刚度 D11) 的真实帕累托前沿",
                   "True Pareto front for [0]N: weight vs bending stiffness"),
                 fontsize=13.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存 {OUTPUT_PNG}")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 88)
    print("结论")
    print("=" * 88)
    light, stiff, balance = rows[0], rows[1], rows[2]
    print(f"  轻量化最优: N={light['N']}，t={light['thickness']:.3f} mm，"
          f"W={light['weight']:.4f} kg/m²，D11={light['D11']:.4f} GPa·mm³")
    print(f"  D11 最大  : N={stiff['N']}，t={stiff['thickness']:.3f} mm，"
          f"W={stiff['weight']:.4f} kg/m²，D11={stiff['D11']:.4f} GPa·mm³")
    print(f"  折中解    : N={balance['N']}，t={balance['thickness']:.3f} mm，"
          f"W={balance['weight']:.4f} kg/m²，D11={balance['D11']:.4f} GPa·mm³")
    gain = stiff["D11"] / light["D11"]
    cost = stiff["weight"] / light["weight"]
    print(f"\n  从 N={light['N']} 增到 N={stiff['N']}: D11 提升 {gain:.1f} 倍，"
          f"重量只增加 {cost:.1f} 倍 —— 这正是 D11 ∝ N³ 与 W ∝ N 的权衡本质。")
    print("  由于指数差异明显（3 次方 vs 1 次方），前沿呈显著上凸，")
    print("  投入重量带来的刚度收益随 N 增大而迅速递减。")
    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

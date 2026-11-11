# -*- coding: utf-8 -*-
"""拉丁超立方采样(LHS)生成 4 层对称铺层 [θ1/θ2/θ2/θ1] 训练数据集。

设计变量 : θ1, θ2 ∈ [0°, 90°]
输出     : training_data.csv（面内刚度 Ex、弯曲刚度 D11 等）
方法     : pymoo 的 sampling_lhs 在角度区间内对 θ1、θ2 分别做一维拉丁超立方
           抽样，保证每个变量在每个维度上分层覆盖。

单位约定（全局统一，防错的核心）：
    长度 mm、模量 GPa（1 GPa = 1000 N/mm^2）
    -> A 矩阵 GPa*mm，B 矩阵 GPa*mm^2，D 矩阵 GPa*mm^3
    角度一律以「度」为单位传入 composipy，仅在需要手算时用 np.radians 转换。
"""

import csv
import sys
import time

import numpy as np
from composipy import LaminateProperty, OrthotropicMaterial
from pymoo.operators.sampling.lhs import sampling_lhs

# 本机 Python 默认 stdout 编码为 GBK(cp936)，中文在管道输出时会乱码，
# 显式切到 UTF-8 保证任何启动方式下中文都正常。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ------------------------- 材料常数 -------------------------
E1 = 181.0      # GPa
E2 = 10.3       # GPa
G12 = 7.17      # GPa
V12 = 0.28
PLY_THICKNESS = 0.125   # mm

# ------------------------- 采样设置 -------------------------
N_SAMPLES = 500
THETA_MIN, THETA_MAX = 0.0, 90.0
SEED = 42
OUTPUT_CSV = "training_data.csv"

# ------------------------- 防错阈值 -------------------------
# 采样点的最小允许间距（度）：LHS 分层后理论最小间距约 90/500 = 0.18°
MIN_ANGLE_GAP = 1e-3
# 对称层合板 B 矩阵应为数值零，超过该值视为异常
B_ZERO_TOL = 1e-9
# 材料常数本身构成的物理边界：任何铺层的 Ex 必须落在 [E2, E1] 内
EX_LOWER, EX_UPPER = E2, E1
# D11 的正下界：对 [θ1/θ2]s，两侧各一层偏离中面 h/2，理论最小 D11 > 0
D11_MIN = 0.0


def stack_angles(theta1, theta2):
    """[θ1/θ2/θ2/θ1] 的展开角度序列。"""
    return [theta1, theta2, theta2, theta1]


def stack_label(theta1, theta2, ndigits=2):
    """铺层顺序的可读标签，如 [30.5/61.2]s。"""
    return "[{0:.{2}f}/{1:.{2}f}]s".format(theta1, theta2, ndigits)


def sample_angles(n_samples, theta_min, theta_max, seed):
    """LHS 采样两个设计变量并映射到角度区间。

    用 pymoo 的 sampling_lhs 在 [theta_min, theta_max] 内分别对 θ1、θ2 各做一次
    一维拉丁超立方抽样（每次调用各占一个新的随机流），再按样本序号配对：
      * 每个变量的 n 个取值恰好落在 n 个互不重叠的等宽分层区间内（LHS 分层性质）；
      * 由于 θ1 的取值两两互异，配对后得到的铺层必然两两不同。

    说明：对 [θ1/θ2]s 而言，交换 θ1 与 θ2、或同时取反（θ → -θ），
    得到的层合板力学性能完全相同；因此直接对二维空间采样容易抽到等效重复铺层，
    分开抽样可以避免这一点。
    """
    random_state = np.random.default_rng(seed)
    theta1 = sampling_lhs(n_samples, 1, xl=theta_min, xu=theta_max,
                          random_state=random_state).ravel()
    theta2 = sampling_lhs(n_samples, 1, xl=theta_min, xu=theta_max,
                          random_state=random_state).ravel()
    return theta1, theta2


def check_angle(theta, name):
    """防错：角度必须是有限实数且落在 [THETA_MIN, THETA_MAX] 内（单位为度）。

    composipy 内部按「度」解释传入角度（源码里做 np.radians），
    这里挡住单位误用（如误传弧度值）和 NaN/inf 流入计算。
    """
    value = float(theta)
    if not np.isfinite(value):
        raise ValueError(f"{name} 不是有限数值: {theta!r}")
    if not (THETA_MIN - 1e-9 <= value <= THETA_MAX + 1e-9):
        raise ValueError(
            f"{name}={value} 超出设计区间 [{THETA_MIN}, {THETA_MAX}]，"
            f"请检查角度单位是否为「度」"
        )
    return value


def laminate_stiffness(theta1, theta2):
    """计算 [θ1/θ2]s 层合板的面内刚度 Ex、Ey、Gxy 与弯曲刚度 D11。

    Ex、Ey、Gxy 由面内柔度 a = A^-1 按总厚度归一化得到：
        Ex = 1/(h*a11),  Ey = 1/(h*a22),  Gxy = 1/(h*a33)
    """
    theta1 = check_angle(theta1, "theta1")
    theta2 = check_angle(theta2, "theta2")

    ply = OrthotropicMaterial(
        e1=E1, e2=E2, v12=V12, g12=G12, thickness=PLY_THICKNESS
    )
    laminate = LaminateProperty(
        stacking=stack_angles(theta1, theta2), plies=[ply] * 4
    )

    A = np.asarray(laminate.A, dtype=float)
    B = np.asarray(laminate.B, dtype=float)
    D = np.asarray(laminate.D, dtype=float)

    # 防错：矩阵必须有限、对称、正定，否则 Ex 等归一化结果无物理意义
    for label, matrix in (("A", A), ("B", B), ("D", D)):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"[{theta1}/{theta2}]s 的 {label} 矩阵含非有限值（NaN/inf）"
            )
    if not np.allclose(A, A.T, rtol=1e-9, atol=1e-9):
        raise ValueError(f"[{theta1}/{theta2}]s 的 A 矩阵不对称")
    if not np.all(np.linalg.eigvalsh(A) > 0):
        raise ValueError(f"[{theta1}/{theta2}]s 的 A 矩阵非正定，无法求面内柔度")

    a = np.linalg.inv(A)
    h = 4.0 * PLY_THICKNESS

    # 防错：归一化后的工程常数必须落在材料常数构成的物理边界内
    Ex = 1.0 / (h * a[0, 0])
    Ey = 1.0 / (h * a[1, 1])
    if not (EX_LOWER - 1e-6 <= Ex <= EX_UPPER + 1e-6):
        raise ValueError(
            f"[{theta1}/{theta2}]s 的 Ex={Ex:.6f} GPa 超出物理边界 "
            f"[{EX_LOWER}, {EX_UPPER}] GPa，疑似单位不一致"
        )
    if not (EX_LOWER - 1e-6 <= Ey <= EX_UPPER + 1e-6):
        raise ValueError(
            f"[{theta1}/{theta2}]s 的 Ey={Ey:.6f} GPa 超出物理边界 "
            f"[{EX_LOWER}, {EX_UPPER}] GPa，疑似单位不一致"
        )
    D11 = D[0, 0]
    if D11 <= D11_MIN:
        raise ValueError(f"[{theta1}/{theta2}]s 的 D11={D11!r} 非正")

    return {
        "theta1": theta1,
        "theta2": theta2,
        "Ex": Ex,
        "Ey": Ey,
        "Gxy": 1.0 / (h * a[2, 2]),
        "A11": A[0, 0],
        "A22": A[1, 1],
        "A66": A[2, 2],
        "D11": D11,
        "B_max_abs": float(np.max(np.abs(B))),
    }


def main():
    print("=" * 78)
    print("LHS 采样生成 4 层对称铺层 [θ1/θ2/θ2/θ1] 数据集")
    print("=" * 78)
    print(f"材料常数 : E1={E1} GPa, E2={E2} GPa, G12={G12} GPa, v12={V12}")
    print(f"单层厚度 : {PLY_THICKNESS} mm    总厚度 h: {4 * PLY_THICKNESS} mm")
    print("采样设置 : pymoo LHS, {0} 组, θ ∈ [{1}°, {2}°], seed={3}".format(
        N_SAMPLES, THETA_MIN, THETA_MAX, SEED))

    theta1_values, theta2_values = sample_angles(
        N_SAMPLES, THETA_MIN, THETA_MAX, SEED
    )

    print("\n" + "-" * 78)
    print("采样得到的铺层角度（前 5 组预览）")
    print("-" * 78)
    for index, (theta1, theta2) in enumerate(
            list(zip(theta1_values, theta2_values))[:5], start=1):
        label = stack_label(theta1, theta2)
        print(f"  样本 {index:2d}:  θ1={theta1:7.3f}°  θ2={theta2:7.3f}°   {label}")
    if N_SAMPLES > 5:
        print(f"  ... 其余 {N_SAMPLES - 5} 组见 CSV")

    # ------------------------- 逐个计算刚度 -------------------------
    print("\n" + "-" * 78)
    print("正在用 composipy 计算各铺层的面内刚度 Ex 与弯曲刚度 D11")
    print("-" * 78)
    start_time = time.perf_counter()
    rows = []
    progress_step = max(1, N_SAMPLES // 10)   # 每约 10% 报告一次进度
    for index, (theta1, theta2) in enumerate(zip(theta1_values, theta2_values), start=1):
        result = laminate_stiffness(theta1, theta2)
        result["sample_id"] = index
        result["stacking"] = stack_label(theta1, theta2)
        rows.append(result)
        if index % progress_step == 0 or index == N_SAMPLES:
            print(f"  进度 {index:4d}/{N_SAMPLES}"
                  f"  最近一组 {result['stacking']:>18s}"
                  f"  Ex={result['Ex']:8.4f} GPa"
                  f"  D11={result['D11']:9.6f} GPa*mm^3")
    elapsed = time.perf_counter() - start_time
    print(f"\n  计算完成，共 {N_SAMPLES} 组，用时 {elapsed:.3f} s "
          f"（平均 {elapsed / N_SAMPLES * 1000:.2f} ms/组）")

    # ------------------------- 保存 CSV -------------------------
    fieldnames = [
        "sample_id", "theta1", "theta2", "stacking",
        "Ex", "Ey", "Gxy", "A11", "A22", "A66", "D11", "B_max_abs",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{row[key]:.10g}" if isinstance(row[key], float) else row[key])
                for key in fieldnames
            })
    print(f"\n  已保存 {len(rows)} 行数据到 {OUTPUT_CSV}（UTF-8 with BOM，Excel 可直接打开）")

    # ------------------------- 汇总与检查 -------------------------
    Ex = np.array([row["Ex"] for row in rows])
    D11 = np.array([row["D11"] for row in rows])

    print("\n" + "-" * 78)
    print("汇总统计")
    print("-" * 78)
    print(f"  Ex  : min={Ex.min():8.4f}  max={Ex.max():8.4f}  "
          f"mean={Ex.mean():8.4f}  std={Ex.std():.4f}  GPa")
    print(f"  D11 : min={D11.min():8.6f}  max={D11.max():8.6f}  "
          f"mean={D11.mean():8.6f}  std={D11.std():.6f}  GPa*mm^3")

    print("\n" + "-" * 78)
    print("数据质量检查")
    print("-" * 78)

    # 1) 铺层唯一性：[θ1/θ2]s 与 [θ2/θ1]s 机械等效，同时取反（θ → -θ）也等效
    canonical = {frozenset({abs(round(row["theta1"], 6)), abs(round(row["theta2"], 6))})
                 for row in rows}
    print(f"  机械等效铺层去重后数量 : {len(canonical)} / {len(rows)}"
          f"  -> {'无重复铺层' if len(canonical) == len(rows) else '存在重复铺层，请检查！'}")

    # 2) 角度落在设计区间内
    all_angles = np.concatenate([np.array([r['theta1'] for r in rows]),
                                 np.array([r['theta2'] for r in rows])])
    in_range = bool(np.all(all_angles >= THETA_MIN) and np.all(all_angles <= THETA_MAX))
    print(f"  角度是否全部落在 [{THETA_MIN}, {THETA_MAX}]° : {in_range}"
          f"  (实际范围 {all_angles.min():.3f}° ~ {all_angles.max():.3f}°)")

    # 3) LHS 分层性质：排序后第 k 个值必须落在第 k 个等宽区间内
    #    直接对 digitize 分箱在区间边界上会受浮点误差影响，故改为区间包含判断。
    for name, values in (("θ1", theta1_values), ("θ2", theta2_values)):
        ordered = np.sort(values)
        step = (THETA_MAX - THETA_MIN) / N_SAMPLES
        lower = THETA_MIN + step * np.arange(N_SAMPLES)
        upper = lower + step
        tol = step * 1e-9
        ok = bool(np.all(ordered >= lower - tol) and np.all(ordered <= upper + tol))
        distinct = len(np.unique(np.round(ordered, 10)))
        print(f"  {name} 分层性质（每层恰好一点）: {ok}"
              f"  |  互异取值 {distinct} / {N_SAMPLES}"
              f"  -> {'成立' if ok and distinct == N_SAMPLES else '被破坏，请检查！'}")

    # 4) 对称层合板不应有拉伸-弯曲耦合
    max_b = max(row["B_max_abs"] for row in rows)
    print(f"  对称铺层的 B 矩阵最大绝对值 : {max_b:.3e} GPa*mm^2"
          f"  -> {'符合预期（数值零）' if max_b < B_ZERO_TOL else '不为零，请检查！'}")

    # 5) 数据完整性与物理合理性
    finite = all(np.isfinite(row[key]) for row in rows
                 for key in ("Ex", "Ey", "Gxy", "D11"))
    print(f"  所有数值均为有限值 : {finite}")
    print(f"  Ex 是否全部落在 [E2, E1] = [{EX_LOWER}, {EX_UPPER}] GPa 内 : "
          f"{bool(np.all(Ex >= EX_LOWER - 1e-6) and np.all(Ex <= EX_UPPER + 1e-6))}")
    print(f"  D11 是否全部为正 : {bool(np.all(D11 > D11_MIN))}")

    # 6) Ex 与 Ey 的对称性：交换 θ1/θ2 即交换 x/y 轴，故满足 Ex(θ1,θ2)=Ey(θ2,θ1)
    swap_ok = True
    for row in rows:
        mirrored = next((r for r in rows
                         if abs(r["theta1"] - row["theta2"]) < 1e-12
                         and abs(r["theta2"] - row["theta1"]) < 1e-12), None)
        if mirrored is not None and abs(mirrored["Ey"] - row["Ex"]) > 1e-6:
            swap_ok = False
            break
    print(f"  交换 θ1/θ2 后 Ex 与 Ey 一致（若存在配对样本）: {swap_ok}")

    print("\n" + "=" * 78)
    print(f"前 5 行数据预览（完整 {len(rows)} 行见 {OUTPUT_CSV}）")
    print("=" * 78)
    preview_columns = [
        ("sample_id", "序号", 7),
        ("theta1", "theta1(°)", 11),
        ("theta2", "theta2(°)", 11),
        ("stacking", "铺层", 18),
        ("Ex", "Ex(GPa)", 11),
        ("Ey", "Ey(GPa)", 11),
        ("Gxy", "Gxy(GPa)", 11),
        ("A11", "A11", 11),
        ("D11", "D11(GPa*mm^3)", 15),
    ]
    header = "  ".join(f"{title:>{width}s}" for _, title, width in preview_columns)
    print(header)
    print("-" * len(header))
    for row in rows[:5]:
        cells = []
        for key, _, width in preview_columns:
            value = row[key]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(f"{text:>{width}s}")
        print("  ".join(cells))

    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""在原有 500 组 LHS 样本基础上做边界加密，修复代理模型在边界严重低估的问题。

背景
    上一轮 train_model.py 的诊断显示：随机森林结构上无法外推，
    在真值峰值处 [0/0]s 把 Ex 低估了 28.8%（预测 128.9 GPa，真值 181.0 GPa）。
    根因是 LHS 在 0°/90° 附近的样本太稀疏，叶节点均值把陡峭的峰值"削平"了。

本脚本做三件事
    1. 额外生成 100 组边界加密样本：θ1、θ2 分别在 0°、5°、10°、85°、90°
       的邻域内随机取值（共 5x5=25 种锚点组合）；
    2. 用 composipy 精确算出这 100 组（以及 2 个解析极值点）的 Ex 与 D11；
    3. 并入原有 500 组，并强制加入解析极值点 [0/0]s（峰值）与 [90/90]s（谷值），
       保存为 augmented_training_data.csv。

输出列比原来多一个 source 列：
    lhs      = 原有 500 组拉丁超立方样本
    boundary = 100 组边界加密样本
    extreme  = 2 组解析极值点（角点）

单位约定：长度 mm、模量 GPa，故 A 为 GPa*mm、B 为 GPa*mm^2、D 为 GPa*mm^3。
"""

import csv
import os
import sys
import time

import numpy as np

# 本机 Python 默认 stdout 编码为 GBK(cp936)，中文会乱码，显式切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 复用已通过验证的实现，避免重复写出公式（两模块均无导入副作用）
from generate_data import (E1, E2, PLY_THICKNESS, laminate_stiffness, stack_label)
from plot_data import laminate_response

# ------------------------- 配置 -------------------------
INPUT_CSV = "training_data.csv"
OUTPUT_CSV = "augmented_training_data.csv"

N_BOUNDARY = 100                        # 边界加密样本数
ANCHORS = [0.0, 5.0, 10.0, 85.0, 90.0]  # 边界锚点（度）
WINDOW = 2.5                            # 邻域半宽（度）-> 0°周边为 [0, 2.5]，其余 ±2.5
SEED = 2024

# 强制加入的解析极值点。
# 代码里给的是**端点角**，目标值由 composipy/CLT 精确算出（不使用四舍五入值），
# 理由：手抄的舍入值（如 1.8939）会与其余行的 12 位精度不一致，
#       且会掩盖真实数值。下方会校验算得的 D11 四舍五入到 4 位小数确实是 1.8939。
EXTREME_ANGLES = [(0.0, 0.0), (90.0, 90.0)]
EXPECTED_PEAK = {"Ex": E1, "D11": 1.8939}   # 用于校验的解析/四舍五入参考值

FIELDNAMES = [
    "sample_id", "source", "theta1", "theta2", "stacking",
    "Ex", "Ey", "Gxy", "A11", "A22", "A66", "D11", "B_max_abs",
]
SIGNIFICANT_DIGITS = 12   # 比原 CSV 的 10 位更精确，避免极值点被舍入污染


def normalize_angle(value):
    """把略微越界的角度吸附到 0 或 90，其余原样返回。

    用容差吸附是必要的：随机取值偶尔会落到 1e-17 这样的越界量级，
    若不吸附就会写出 -0.0 / 90.00000000000001 这类脏数据。
    """
    if abs(value) < 1e-8:
        return 0.0
    if abs(value - 90.0) < 1e-8:
        return 90.0
    return float(value)


def key_of(theta1, theta2):
    """角度组合的去重键。"""
    return (round(normalize_angle(theta1), 9), round(normalize_angle(theta2), 9))


def sample_boundary_angles(n_samples, anchors, window, seed):
    """在锚点邻域内随机取角度，返回 (theta1, theta2) 两个数组。

    每个样本独立地：先随机选一个锚点，再在该锚点的 [a-w, a+w] 邻域内均匀取值，
    最后吸附到 [0, 90]。0° 与 90° 的邻域被边界截断，因此这两个锚点附近的取值
    会有一部分正好落在 0° 或 90° 上 —— 这正是我们最需要加密的位置。
    """
    rng = np.random.default_rng(seed)
    anchor_values = np.asarray(anchors, dtype=float)

    def draw():
        chosen = rng.choice(anchor_values, size=n_samples)
        offsets = rng.uniform(-window, window, size=n_samples)
        return np.array([normalize_angle(value)
                         for value in np.clip(chosen + offsets, 0.0, 90.0)])

    return draw(), draw()


def compute_rows(pairs, source, start_id):
    """用 composipy 计算每个铺层对的刚度，同时用独立解析式交叉校验。

    返回记录列表。任一点两套算法不一致即抛错（防止错误数据流入数据集）。
    """
    rows = []
    for offset, (theta1, theta2) in enumerate(pairs):
        try:
            result = laminate_stiffness(theta1, theta2)
        except ValueError as error:
            raise SystemExit(f"计算 {theta1}/{theta2} 失败: {error}")

        # 独立校验：plot_data 的解析式 CLT 实现
        ex_ref, ey_ref, d11_ref = laminate_response(theta1, theta2)
        checks = {"Ex": (result["Ex"], ex_ref), "Ey": (result["Ey"], ey_ref),
                  "D11": (result["D11"], d11_ref)}
        for key, (computed, reference) in checks.items():
            if abs(computed - reference) > 1e-9 * max(1.0, abs(reference)):
                raise SystemExit(
                    f"[{theta1:.6f}/{theta2:.6f}]s 的 {key} 两套算法不一致: "
                    f"{computed!r} vs {reference!r}")

        row = {"sample_id": start_id + offset, "source": source,
               "theta1": float(theta1), "theta2": float(theta2),
               "stacking": stack_label(theta1, theta2)}
        for key in ("Ex", "Ey", "Gxy", "A11", "A22", "A66", "D11", "B_max_abs"):
            row[key] = float(result[key])
        rows.append(row)
    return rows


def read_original(path):
    """读取原有 CSV 并保留其数值。"""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        raise SystemExit(f"找不到 {path}，请先运行 generate_data.py。")


def format_value(value):
    """按有效数字格式化；文本列（如 source、stacking）原样返回。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{SIGNIFICANT_DIGITS}g}"


def main():
    print("=" * 80)
    print("边界加密：扩充训练数据集")
    print("=" * 80)

    # ---------------- 1) 读取原有数据 ----------------
    original = read_original(INPUT_CSV)
    print(f"原有数据 {INPUT_CSV}: {len(original)} 组")
    original_keys = {key_of(float(row["theta1"]), float(row["theta2"]))
                     for row in original}

    # ---------------- 2) 解析极值点（先固定，避免被随机采样"碰巧"覆盖）----------------
    print("\n" + "-" * 80)
    print("2) 固定解析极值点")
    print("-" * 80)
    extreme_rows = compute_rows(
        EXTREME_ANGLES, "extreme", len(original) + N_BOUNDARY + 1)
    for row in extreme_rows:
        print(f"  [{row['theta1']:.0f}/{row['theta2']:.0f}]s  "
              f"Ex={row['Ex']:.4f} GPa  D11={row['D11']:.4f} GPa·mm³")

    # 校验：算得的极值与理论/用户给定值一致
    peak = next(row for row in extreme_rows if row["theta1"] == 0.0)
    print(f"\n  校验 [0/0]s: Ex={peak['Ex']:.6f}（应等于 E1={E1}）"
          f" -> {'一致' if abs(peak['Ex'] - E1) < 1e-9 else '不一致！'}")
    print(f"              D11={peak['D11']:.6f}（四舍五入 4 位 = "
          f"{round(peak['D11'], 4)}，用户给定 1.8939）"
          f" -> {'一致' if round(peak['D11'], 4) == EXPECTED_PEAK['D11'] else '不一致！'}")
    if abs(peak["Ex"] - E1) > 1e-9 or round(peak["D11"], 4) != EXPECTED_PEAK["D11"]:
        raise SystemExit("解析极值点校验失败！")
    trough = next(row for row in extreme_rows if row["theta1"] == 90.0)
    print(f"  校验 [90/90]s: Ex={trough['Ex']:.6f}（应等于 E2={E2}）"
          f" -> {'一致' if abs(trough['Ex'] - E2) < 1e-9 else '不一致！'}")
    if abs(trough["Ex"] - E2) > 1e-9:
        raise SystemExit("解析极值点校验失败！")

    # 极值点的角度组合先行占用，确保后面的随机边界样本不与它们重复
    reserved_keys = {key_of(row["theta1"], row["theta2"]) for row in extreme_rows}

    # ---------------- 3) 边界加密采样 ----------------
    print("\n" + "-" * 80)
    print(f"3) 生成 {N_BOUNDARY} 组边界加密样本")
    print("-" * 80)
    print(f"  锚点: {ANCHORS}   邻域半宽: ±{WINDOW}°   随机种子: {SEED}")
    print(f"  0° 邻域被边界截断为 [0, {WINDOW}]，90° 邻域为 [{90 - WINDOW}, 90]")
    theta1_b, theta2_b = sample_boundary_angles(N_BOUNDARY, ANCHORS, WINDOW, SEED)

    # 统计落入各锚点邻域的样本数，确认加密分布均匀
    for name, values in (("θ1", theta1_b), ("θ2", theta2_b)):
        counts = []
        for anchor in ANCHORS:
            low, high = anchor - WINDOW, anchor + WINDOW
            counts.append(int(np.sum((values >= low - 1e-9) & (values <= high + 1e-9))))
        print(f"  {name} 落入各锚点邻域的计数 {dict(zip(ANCHORS, counts))}")

    # 去重：取值被吸附到边界时会产生完全相同的角度组合，
    # 也可能与原有 500 组或已固定的极值点重合
    blocked = original_keys | reserved_keys
    drawn_keys, unique_pairs = set(), []
    for theta1, theta2 in zip(theta1_b, theta2_b):
        key = key_of(theta1, theta2)
        if key in drawn_keys or key in blocked:
            continue
        drawn_keys.add(key)
        unique_pairs.append((theta1, theta2))
    print(f"  原始采样 {len(theta1_b)} 组 -> 去重后 {len(unique_pairs)} 组"
          f"（丢弃 {len(theta1_b) - len(unique_pairs)} 组与已有/彼此重复的角度组合）")

    # 补足到 N_BOUNDARY 组：从后续批次继续取，同样跳过重复
    needed = N_BOUNDARY - len(unique_pairs)
    if needed > 0:
        extra1, extra2 = sample_boundary_angles(needed + 50, ANCHORS, WINDOW, SEED + 1)
        for theta1, theta2 in zip(extra1, extra2):
            key = key_of(theta1, theta2)
            if key in drawn_keys or key in blocked:
                continue
            drawn_keys.add(key)
            unique_pairs.append((theta1, theta2))
            if len(unique_pairs) == N_BOUNDARY:
                break
    if len(unique_pairs) != N_BOUNDARY:
        raise SystemExit(f"仅生成 {len(unique_pairs)} 组唯一边界样本，未达到 {N_BOUNDARY} 组。")
    print(f"  最终边界样本: {len(unique_pairs)} 组，与原有数据及极值点均无重复")

    start = time.perf_counter()
    boundary_rows = compute_rows(unique_pairs, "boundary", len(original) + 1)
    print(f"  已用 composipy 算出 {len(boundary_rows)} 组真实 Ex、D11"
          f"（并逐点通过解析式交叉校验），用时 {time.perf_counter() - start:.3f} s")

    # ---------------- 4) 合并并写出 ----------------
    print("\n" + "-" * 80)
    print("4) 合并新老数据并保存")
    print("-" * 80)

    merged = []
    # 原有数据：统一 source 标记，逐行核对与重算结果一致
    mismatches = 0
    for row in original:
        theta1, theta2 = float(row["theta1"]), float(row["theta2"])
        recomputed = laminate_stiffness(theta1, theta2)
        for key in ("Ex", "Ey", "Gxy", "A11", "A22", "A66", "D11"):
            if abs(recomputed[key] - float(row[key])) > 1e-6 * max(1.0, abs(float(row[key]))):
                mismatches += 1
                break
        merged.append({
            "source": "lhs",
            "theta1": theta1,
            "theta2": theta2,
            "stacking": row["stacking"],
            **{key: float(row[key]) for key in
               ("Ex", "Ey", "Gxy", "A11", "A22", "A66", "D11", "B_max_abs")},
        })
    print(f"  原有 {len(original)} 组已重新核对（与重算结果不符的行数: {mismatches}）")

    # 边界样本与极值点并入：按角度组合去重（保留先加入者）
    seen_keys = set(original_keys)
    kept_boundary = []
    for row in boundary_rows:
        key = key_of(row["theta1"], row["theta2"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept_boundary.append(row)
    merged.extend(kept_boundary)

    kept_extreme = []
    for row in extreme_rows:
        key = key_of(row["theta1"], row["theta2"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kept_extreme.append(row)
    merged.extend(kept_extreme)
    print(f"  并入边界样本 {len(kept_boundary)} 组、解析极值点 {len(kept_extreme)} 组")

    # 统一重编 sample_id：原有 -> 边界 -> 极值
    for index, row in enumerate(merged, start=1):
        row["sample_id"] = index

    # 重复检查：角度完全相同的行
    keys = [key_of(row["theta1"], row["theta2"]) for row in merged]
    duplicates = len(keys) - len(set(keys))
    print(f"  合并后共 {len(merged)} 组；角度完全相同的重复行: {duplicates}"
          f" -> {'无重复' if duplicates == 0 else '存在重复，请检查！'}")
    if duplicates:
        raise SystemExit("合并后仍有重复角度组合，数据不干净！")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in merged:
            writer.writerow({key: format_value(row[key]) for key in FIELDNAMES})
    size_kb = os.path.getsize(OUTPUT_CSV) / 1024.0
    print(f"  已保存 {OUTPUT_CSV}（{len(merged)} 行，{size_kb:.1f} KB，"
          f"{SIGNIFICANT_DIGITS} 位有效数字）")

    # ---------------- 4) 规模与极值分布 ----------------
    print("\n" + "=" * 80)
    print("新数据集规模与极值分布")
    print("=" * 80)
    sources = {}
    for row in merged:
        sources[row["source"]] = sources.get(row["source"], 0) + 1
    print(f"  总样本数: {len(merged)}")
    for name, count in sources.items():
        print(f"    {name:<9s} {count:4d} 组")

    Ex = np.array([row["Ex"] for row in merged])
    D11 = np.array([row["D11"] for row in merged])
    Ex_old = np.array([float(row["Ex"]) for row in original])
    D11_old = np.array([float(row["D11"]) for row in original])

    print(f"\n  {'目标':<5s} {'':>10s} {'min':>12s} {'max':>12s} {'mean':>10s} {'std':>10s}")
    for name, values in (("Ex", Ex), ("D11", D11)):
        print(f"  {name:<5s} 扩充后  {values.min():12.4f} {values.max():12.4f} "
              f"{values.mean():10.4f} {values.std():10.4f}")
    for name, new, old in (("Ex", Ex, Ex_old), ("D11", D11, D11_old)):
        print(f"  {name:<5s} 原数据  {old.min():12.4f} {old.max():12.4f} "
              f"{old.mean():10.4f} {old.std():10.4f}")

    # 峰值/谷值落在哪一组
    for name, values in (("Ex", Ex), ("D11", D11)):
        high = int(np.argmax(values))
        low = int(np.argmin(values))
        print(f"\n  {name} 峰值: {merged[high]['stacking']}  {values[high]:.6f}"
              f"  (来源 {merged[high]['source']})")
        print(f"  {name} 谷值: {merged[low]['stacking']}  {values[low]:.6f}"
              f"  (来源 {merged[low]['source']})")

    # 边界加密效果：峰值附近 (θ1,θ2 都 <=12°) 的样本密度
    near = (merged and [row for row in merged
                        if row["theta1"] <= 12.0 and row["theta2"] <= 12.0])
    near_old = [row for row in original
                if float(row["theta1"]) <= 12.0 and float(row["theta2"]) <= 12.0]
    print(f"\n  [0°,12°]x[0°,12°] 峰值区样本数: 原 {len(near_old)} 组 -> "
          f"扩充后 {len(near)} 组")
    print(f"  Ex 最大值提升: {Ex_old.max():.4f} -> {Ex.max():.4f} GPa "
          f"(+{Ex.max() - Ex_old.max():.4f})")
    print(f"  D11 最大值提升: {D11_old.max():.6f} -> {D11.max():.6f} GPa·mm³ "
          f"(+{D11.max() - D11_old.max():.6f})")

    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

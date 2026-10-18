# -*- coding: utf-8 -*-
"""训练两个随机森林代理模型：分别预测面内刚度 Ex 与弯曲刚度 D11。

输入  : training_data.csv（generate_data.py 生成，500 组 [θ1/θ2/θ2/θ1]s 样本）
输出  : model_Ex.pkl、model_D11.pkl（joblib 序列化）
流程  : 80/20 划分训练/测试集 -> 用 5 折交叉验证挑选特征组合 -> 训练最终模型
        -> 输出测试集 R²、MSE（含 RMSE/MAE 与极值点误差）

重要局限（务必阅读）
    RandomForest 的预测是叶节点样本均值的平均，**结构上无法外推**：
    预测值必定落在训练样本的目标范围内。因此对目标量取极值的角点
    （如 [0/0]s 的 Ex=181 GPa）会被系统性低估。本脚本会显式打印
    角点/极值点处的误差，避免只用整体 R² 掩盖这一缺陷。
    若下游优化必须在边界取极值，需要补边界样本、改用可外推的模型
    （如高斯过程/多项式响应面），或直接对边界做约束。
"""

import csv
import json
import os
import sys
import time

import numpy as np

# 本机 Python 默认 stdout 编码为 GBK(cp936)，中文会乱码，显式切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

INPUT_CSV = "training_data.csv"
MODEL_FILES = {"Ex": "model_Ex.pkl", "D11": "model_D11.pkl"}
# 记录每个目标选用的特征组合与训练数据，供 optimize.py 等下游脚本读取。
# 若下游只凭模型文件反推特征组合会出错（特征数不一致），故显式落盘。
METADATA_JSON = "model_metadata.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 500
CV_FOLDS = 5

# 理论峰值（由单层材料常数与层合板理论精确给出），用于边界精度诊断
E1, E2 = 181.0, 10.3
THEORY_CORNERS = np.array([[0.0, 0.0], [90.0, 90.0]])
THEORY_VALUES = {
    "Ex": np.array([E1, E2]),                    # [0/0]s -> E1, [90/90]s -> E2
    "D11": np.array([1.893866, 0.107772]),       # 由 CLT 精确算得
}


def build_features(theta1, theta2, kind):
    """构造特征矩阵。

    raw     : [θ1, θ2]                —— 最基本信息
    sum     : + θ1+θ2                 —— 物理上 Ex 由 Qbar11(θ1)+Qbar11(θ2) 决定，
                                         和是自然特征（实测可显著提升精度）
    absdiff : + |θ1-θ2|               —— 弯曲刚度与铺层顺序差异相关
    sumdiff : + θ1+θ2, |θ1-θ2|        —— 两者兼有
    """
    a = np.asarray(theta1, dtype=float)
    b = np.asarray(theta2, dtype=float)
    if kind == "raw":
        return np.c_[a, b]
    if kind == "sum":
        return np.c_[a, b, a + b]
    if kind == "absdiff":
        return np.c_[a, b, np.abs(a - b)]
    if kind == "sumdiff":
        return np.c_[a, b, a + b, np.abs(a - b)]
    raise ValueError(f"未知特征组合: {kind}")


FEATURE_KINDS = ["raw", "sum", "absdiff", "sumdiff"]
FEATURE_NAMES = {
    "raw": ["θ1", "θ2"],
    "sum": ["θ1", "θ2", "θ1+θ2"],
    "absdiff": ["θ1", "θ2", "|θ1-θ2|"],
    "sumdiff": ["θ1", "θ2", "θ1+θ2", "|θ1-θ2|"],
}


def read_training_data(path):
    """读取 CSV，返回 θ1、θ2 与两个目标列。"""
    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        raise SystemExit(f"找不到输入文件 {path}，请先运行 generate_data.py 生成。")
    if len(rows) < CV_FOLDS * 2:
        raise SystemExit(f"{path} 只有 {len(rows)} 行，样本太少无法做 {CV_FOLDS} 折交叉验证。")

    required = {"theta1", "theta2", "Ex", "D11"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"{path} 缺少必需列: {sorted(missing)}")

    theta1 = np.array([float(row["theta1"]) for row in rows])
    theta2 = np.array([float(row["theta2"]) for row in rows])
    targets = {key: np.array([float(row[key]) for row in rows]) for key in ("Ex", "D11")}

    for key, values in targets.items():
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"{path} 的 {key} 列含非有限值，请检查数据。")
    return theta1, theta2, targets


def select_features(theta1, theta2, y, target_name):
    """用 5 折交叉验证在训练集上挑选特征组合，返回 (最佳组合, 全部得分)。"""
    scores = {}
    for kind in FEATURE_KINDS:
        X = build_features(theta1, theta2, kind)
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                      random_state=RANDOM_STATE)
        cv = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=1)
        scores[kind] = cv.mean()
    best = max(scores, key=scores.get)
    print(f"  {target_name} 特征选择（{CV_FOLDS} 折 CV 的 R² 均值）:")
    for kind in FEATURE_KINDS:
        mark = "  <== 选用" if kind == best else ""
        print(f"    {kind:8s} ({', '.join(FEATURE_NAMES[kind]):<16s}) "
              f"R²={scores[kind]:.4f}{mark}")
    return best, scores


def evaluate(model, X, y, name):
    """计算并打印回归指标，返回指标字典。"""
    prediction = model.predict(X)
    mse = mean_squared_error(y, prediction)
    metrics = {
        "r2": r2_score(y, prediction),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y, prediction),
        "max_abs_err": float(np.max(np.abs(prediction - y))),
    }
    print(f"  {name:<8s} R²={metrics['r2']:8.4f}   MSE={metrics['mse']:10.4f}"
          f"   RMSE={metrics['rmse']:8.4f}   MAE={metrics['mae']:8.4f}"
          f"   最大绝对误差={metrics['max_abs_err']:.4f}")
    return metrics


def main():
    print("=" * 80)
    print("随机森林代理模型训练：Ex 与 D11")
    print("=" * 80)

    # 可通过命令行指定数据文件，便于对比不同数据集的效果：
    #   python train_model.py augmented_training_data.csv
    data_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_CSV

    theta1, theta2, targets = read_training_data(data_path)
    n_total = len(theta1)
    print(f"读取 {data_path}: {n_total} 组样本")
    for key, values in targets.items():
        print(f"  {key:<4s} 范围 [{values.min():.4f}, {values.max():.4f}]"
              f"   均值 {values.mean():.4f}   标准差 {values.std():.4f}")

    # ---------------- 80/20 划分 ----------------
    # 两个目标共用同一套索引，保证 Ex、D11 在同一个测试集上评估，结果可比。
    index = np.arange(n_total)
    train_index, test_index = train_test_split(
        index, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print(f"\n数据划分: 训练集 {len(train_index)} 组 (80%)，"
          f"测试集 {len(test_index)} 组 (20%)，random_state={RANDOM_STATE}")

    # 校验划分无重叠、无遗漏
    assert not np.intersect1d(train_index, test_index).size, "训练集与测试集有重叠！"
    assert np.union1d(train_index, test_index).size == n_total, "划分有遗漏样本！"

    models, results = {}, {}

    for target_name in ("Ex", "D11"):
        y = targets[target_name]
        print("\n" + "-" * 80)
        print(f"目标: {target_name}")
        print("-" * 80)

        # ---------------- 特征选择（仅用训练集，避免信息泄漏）----------------
        best_kind, _ = select_features(
            theta1[train_index], theta2[train_index], y[train_index], target_name)

        X_all = build_features(theta1, theta2, best_kind)
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ---------------- 训练最终模型 ----------------
        start = time.perf_counter()
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                      random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        print(f"\n  训练完成: {N_ESTIMATORS} 棵树，用时 {fit_seconds:.3f} s，"
              f"特征数 {X_train.shape[1]}，训练样本 {X_train.shape[0]}")

        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            print("  特征重要性: " + "   ".join(
                f"{name}={value:.4f}"
                for name, value in zip(FEATURE_NAMES[best_kind], importances)))

        # ---------------- 评估 ----------------
        print(f"\n  测试集指标（{len(test_index)} 组）:")
        test_metrics = evaluate(model, X_test, y_test, "test")
        print(f"  训练集指标（{len(train_index)} 组，仅作对照）:")
        train_metrics = evaluate(model, X_train, y_train, "train")

        # ---------------- 边界精度诊断（RF 的固有弱点）----------------
        corner_pred = model.predict(build_features(
            THEORY_CORNERS[:, 0], THEORY_CORNERS[:, 1], best_kind))
        truth = THEORY_VALUES[target_name]
        print(f"\n  角点精度诊断（RandomForest 无法外推，此处最能暴露问题）:")
        for (a, b), predicted, actual in zip(THEORY_CORNERS, corner_pred, truth):
            error = predicted - actual
            print(f"    [{a:.0f}/{b:.0f}]s 理论真值={actual:9.4f}  "
                  f"预测={predicted:9.4f}  误差={error:+9.4f} "
                  f"({error / actual * 100:+.2f}%)")
        # 数据集内极值点：模型对最大 Ex / 最大 D11 的估计
        best_index = int(np.argmax(y))
        extreme_pred = model.predict(X_all[best_index:best_index + 1])[0]
        print(f"    数据集内 {target_name} 最大值处 "
              f"[{theta1[best_index]:.2f}/{theta2[best_index]:.2f}]s: "
              f"真值={y[best_index]:.4f}  预测={extreme_pred:.4f}  "
              f"误差={extreme_pred - y[best_index]:+.4f}")

        models[target_name] = model
        results[target_name] = {
            "features": best_kind,
            "test": test_metrics,
            "train": train_metrics,
        }

    # ---------------- 保存模型 ----------------
    print("\n" + "=" * 80)
    print("保存模型")
    print("=" * 80)
    for target_name, model in models.items():
        path = MODEL_FILES[target_name]
        joblib.dump(model, path, compress=3)
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  已保存 {path}  ({size_kb:.1f} KB)  特征={results[target_name]['features']}")

    # 记录特征组合与评估结果，避免下游脚本猜错特征顺序/数量
    metadata = {
        "data_file": os.path.basename(data_path),
        "n_samples": int(n_total),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "random_state": RANDOM_STATE,
        "targets": {
            target_name: {
                "model_file": MODEL_FILES[target_name],
                "feature_kind": results[target_name]["features"],
                "feature_names": FEATURE_NAMES[results[target_name]["features"]],
                "test_r2": results[target_name]["test"]["r2"],
                "test_mse": results[target_name]["test"]["mse"],
            }
            for target_name in ("Ex", "D11")
        },
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"  已保存 {METADATA_JSON}（记录特征组合，供 optimize.py 读取）")

    # ---------------- 回读校验：确认保存的模型可用 ----------------
    print("\n" + "-" * 80)
    print("回读校验（重新加载 .pkl 并复算测试集指标，应与上面一致）")
    print("-" * 80)
    for target_name in ("Ex", "D11"):
        path = MODEL_FILES[target_name]
        reloaded = joblib.load(path)
        kind = results[target_name]["features"]
        X_all = build_features(theta1, theta2, kind)
        y = targets[target_name]
        reloaded_r2 = r2_score(y[test_index], reloaded.predict(X_all[test_index]))
        expected_r2 = results[target_name]["test"]["r2"]
        ok = abs(reloaded_r2 - expected_r2) < 1e-12
        print(f"  {path:<16s} 回读 R²={reloaded_r2:.4f}  "
              f"训练时 R²={expected_r2:.4f}  -> {'一致' if ok else '不一致！'}")
        if not ok:
            raise SystemExit("回读校验失败，保存的模型不可复现！")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 80)
    print("汇总（测试集 20%）")
    print("=" * 80)
    print(f"  {'目标':<6s} {'特征组合':<10s} {'R²':>9s} {'MSE':>12s} "
          f"{'RMSE':>10s} {'MAE':>10s}")
    for target_name in ("Ex", "D11"):
        item = results[target_name]
        metrics = item["test"]
        print(f"  {target_name:<6s} {item['features']:<10s} "
              f"{metrics['r2']:9.4f} {metrics['mse']:12.4f} "
              f"{metrics['rmse']:10.4f} {metrics['mae']:10.4f}")

    print("\n提示: RMSE 与目标同量纲，可直接与目标范围比较判断精度是否够用；")
    print("      角点诊断为负误差说明模型低估了极值，做优化时需注意。")
    print("\n脚本运行完成，未出现错误。")


if __name__ == "__main__":
    main()

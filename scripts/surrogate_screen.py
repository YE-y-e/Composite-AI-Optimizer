# -*- coding: utf-8 -*-
"""临界载荷代理模型：用它做筛选器，量化"命中率"与"加速比"。

问题：Ritz 单次约 10 ms，扫 5000 个候选要 50 s——这在交互式设计里太慢。
做法：先用 Ritz 标注 2000 个候选训练随机森林，再用它给 5000 个候选打分，
只把前 K 名送去 Ritz 复核；与"全部 5000 个都用 Ritz 算"的真值比较。

输出：模型精度（R²/MAE）、top-K 命中率、质量损失、加速比。
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad          # noqa: E402
import angle_optimize as ao       # noqa: E402

SEED = 42
N_TRAIN = 2000
N_TEST = 500
N_SCREEN = 5000
TOPK = (1, 5, 10, 50, 200)
N_PLIES = 10
NX = NY = 500.0

SOLVER = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)


def ncr_of(theta4):
    stack = ad.symmetric_stack(list(theta4), N_PLIES)
    D = ad.d_of_angles(stack)
    return float(SOLVER.ncr(D, NX, NY)) * max(NX, NY)


def sample(n, rng):
    return rng.uniform(-90.0, 90.0, size=(n, 4))


def features(X):
    t1, t2, t3, t4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.column_stack([X, np.abs(X), t1 + t3, t1 - t3, t2 + t4, t2 - t4])


def main():
    rng = np.random.default_rng(SEED)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    # 1) 标注训练/测试集
    Xtr = sample(N_TRAIN, rng)
    ytr = np.array([ncr_of(t) for t in Xtr])
    Xte = sample(N_TEST, rng)
    yte = np.array([ncr_of(t) for t in Xte])

    t0 = time.time()
    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(features(Xtr), ytr)
    train_s = time.time() - t0

    pred_te = model.predict(features(Xte))
    r2 = float(r2_score(yte, pred_te))
    mae = float(mean_absolute_error(yte, pred_te))
    rel_mae = mae / float(yte.mean()) * 100.0

    # 2) 筛选实验：5000 个候选，全算真值 vs 只算 top-K
    Xs = sample(N_SCREEN, rng)
    t0 = time.time()
    y_true = np.array([ncr_of(t) for t in Xs])
    t_true = time.time() - t0

    t0 = time.time()
    y_hat = model.predict(features(Xs))
    t_pred = time.time() - t0

    order = np.argsort(y_hat)[::-1]
    best_true = float(y_true.max())
    from scipy.stats import spearmanr

    rho = float(spearmanr(y_hat, y_true).statistic)
    hits = {}
    for k in TOPK:
        idx = order[:k]
        t0 = time.time()
        verified = np.array([ncr_of(Xs[i]) for i in idx])
        t_k = time.time() - t0
        best_k = float(verified.max())
        # top-K 与"真实 top-K"的命中率：代理挑出的集合里有多少个真在前 K
        true_top = set(np.argsort(y_true)[::-1][:k].tolist())
        overlap = len(true_top & set(idx.tolist())) / k * 100.0
        hits[str(k)] = {
            "best_verified": best_k,
            "loss_pct": (best_true - best_k) / best_true * 100.0,
            "contains_global_opt": bool(best_k >= best_true - 1e-9),
            "rank_of_opt_in_screen": int(np.where(order == int(np.argmax(y_true)))[0][0]) + 1,
            "overlap_with_true_topk_pct": overlap,
            "time_s": t_k + t_pred,
            # 不记录 speedup_vs_full：单次求解耗时会随机器负载成倍波动，
            # 用"当次全扫墙钟"除以"基准单次耗时"得到的倍数没有可比性。
            # 效率证据统一用"复核次数"。见 bench_solver.py 的基准值。
        }

    np.savez_compressed(ROOT / "data" / "surrogate_screen_data.npz",
                        y_true=y_true, y_hat=y_hat, y_test=yte, pred_test=pred_te)

    out = {
        "seed": SEED,
        "n_plies": N_PLIES, "nx": NX, "ny": NY,
        "n_train": N_TRAIN, "n_test": N_TEST, "n_screen": N_SCREEN,
        "model": {"n_estimators": 300, "r2_test": r2, "mae": mae,
                  "rel_mae_pct": rel_mae, "train_s": train_s,
                  "spearman_rho": rho},
        "full_scan": {"n": N_SCREEN, "best_true": best_true,
                      "time_s": t_true,
                      "mean_solve_ms": t_true / N_SCREEN * 1000.0},
        "screening": hits,
        "screen": {"n": N_SCREEN, "time_s": t_pred,
                   "time_per_candidate_us": t_pred / N_SCREEN * 1e6},
    }
    path = ROOT / "data" / "surrogate_screen.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    import gzip
    import pickle
    # 300 棵树的模型约 52 MB，gzip 后约 14 MB；报告里用的是这份压缩件
    with gzip.open(ROOT / "data" / "model_ncr.pkl.gz", "wb", compresslevel=6) as fh:
        pickle.dump(model, fh, protocol=4)

    print(f"模型：R² = {r2:.4f}，MAE = {mae:.1f} N/m（{rel_mae:.2f}%），训练 {train_s:.1f} s")
    print(f"全扫 {N_SCREEN} 个：最优 {best_true:,.1f} N/m，用时 {t_true:.1f} s"
          f"（{t_true / N_SCREEN * 1000:.1f} ms/次）")
    print(f"代理打分 {N_SCREEN} 个：{t_pred * 1000:.0f} ms（{t_pred / N_SCREEN * 1e6:.0f} µs/次）")
    for k in TOPK:
        h = hits[str(k)]
        print(f"  top-{k:<3} 复核最优 {h['best_verified']:,.1f}"
              f"（损失 {h['loss_pct']:+.2f}%，最优排名 {h['rank_of_opt_in_screen']}，"
              f"复核 {k} 次 vs 全扫 {N_SCREEN} 次）")
    print("saved", path)


if __name__ == "__main__":
    main()

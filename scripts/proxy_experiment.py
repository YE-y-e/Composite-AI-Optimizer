# -*- coding: utf-8 -*-
"""代理筛选实验入口（对应完善计划 C1–C4）。

一条命令跑完一次"公平对照"实验，回答三个问题：
  1. 用随机森林排前 K 名再复核，能拿到多好的可行解？（质量损失）
  2. 同样的真实求解次数，随机挑选能拿到多好？（学习是否优于随机）
  3. 这种收益在不同随机种子下稳定吗？（稳定性）

设计原则（照计划执行）：
  * 两条路径分离：本脚本只做离线实验，不改工程主流程，不被前端调用；
  * 同域同精度：训练集、测试集、候选池使用同一个角度域与同一个求解器阶次，
    并按"实际铺层"去重，避免等价角度跨集合泄漏；
  * 统一评价：所有候选（含最终方案）都走 Auto_Design.evaluate_candidate，
    返回 λ、FI、制造规则与 feasible，预测值不决定最终是否通过；
  * 记录可追溯：配置、划分、seed、代码指纹、分项耗时、失败原因全部落 JSON。

用法：
  python scripts/proxy_experiment.py --n-plies 10 --nx 500 --ny 500 --sf 1.5 \
      --domain grid --seeds 0,1,2 --budget 5,10,20,52 --out data/proxy_grid.json
  python scripts/proxy_experiment.py --domain continuous --n-pool 3000 \
      --n-train 400 --n-test 200 --seeds 0,1,2,3,4 --budget 5,10,20,50,100,300 \
      --out data/proxy_continuous.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

import Auto_Design as ad          # noqa: E402
import angle_optimize as ao       # noqa: E402


# ---------------------------------------------------------------- 基础设施

def code_fingerprint() -> dict:
    """代码指纹：把参与计算的模块哈希写进结果，便于追溯"哪一版跑出来的"。"""
    files = ["codes/Auto_Design.py", "codes/angle_optimize.py",
             "scripts/proxy_experiment.py"]
    out = {}
    for rel in files:
        p = ROOT / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def make_solver(m: int):
    return ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, m)


def fold_angle(a: float) -> float:
    """把铺层角折到 [-90, 90)：θ 与 θ+180° 是同一层，必须视为同一设计（审查 A1）。"""
    return float(((float(a) + 90.0) % 180.0) - 90.0)


def layup_key(theta4, n_plies) -> tuple:
    """规范化铺层签名：先折角，再按实际铺层去重。"""
    stack = ad.symmetric_stack([fold_angle(t) for t in theta4], n_plies)
    return tuple(round(float(a), 6) for a in stack)


def evaluate(theta4, cfg, solver) -> dict:
    """统一评价接口：复用 Auto_Design 的完整校核（λ + FI + 规则 + feasible）。"""
    return ad.evaluate_candidate(theta4, cfg["n_plies"], cfg["nx"], cfg["ny"],
                                 cfg["sf"], solver=solver)


# ---------------------------------------------------------------- 候选生成

def gen_grid(cfg) -> list[list[float]]:
    """可制造离散域：4 个设计角 × 网格的全组合，按实际铺层去重。

    网格默认取工程用的 5 档（`ad.ANGLE_GRID`）；传 `--angles` 可换更细的网格，
    用于"候选池大到无法全扫"的对照（计划 E6 的"有收益"一侧）。
    """
    grid = sorted({fold_angle(a) for a in (cfg.get("angles") or list(ad.ANGLE_GRID))})
    seen, out = set(), []
    for combo in itertools.product(grid, repeat=4):
        cand = [fold_angle(a) for a in combo]
        rules = ad.check_rules(cand, cfg["n_plies"])
        if cfg["enforce_balance"] and not (rules["symmetric"] and rules["balanced"]):
            continue
        key = layup_key(cand, cfg["n_plies"])
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def gen_continuous(cfg, rng) -> list[list[float]]:
    """连续域：拉丁超立方采样 + 均衡规则过滤 + 按铺层去重。"""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=4, seed=int(rng.integers(1 << 30)))
    raw = sampler.random(cfg["n_raw"])
    cand = raw * 180.0 - 90.0
    seen, out = set(), []
    for theta in cand:
        theta = [float(a) for a in theta]
        # 与工程路径一致的制造规则：对称（构造保证）+ 面内均衡
        rules = ad.check_rules(theta, cfg["n_plies"])
        if cfg["enforce_balance"] and not rules["balanced"]:
            continue
        key = layup_key(theta, cfg["n_plies"])
        if key in seen:
            continue
        seen.add(key)
        out.append(theta)
        if len(out) >= cfg["n_pool"]:
            break
    return out


def split_disjoint(cands, n_train, n_test, n_pool, rng):
    """把候选切成互不重叠的三份：训练 / 测试 / 候选池（按铺层去重后再切）。

    n_train / n_test 传 0 表示按比例自动分配（离散域只有几百个候选时必须这样，
    否则固定 400/200 会把候选池挤到只剩个位数）。
    """
    idx = rng.permutation(len(cands))
    n_all = len(idx)
    if n_train <= 0:
        n_train = max(20, int(round(n_all * 0.30)))
    if n_test <= 0:
        n_test = max(10, int(round(n_all * 0.15)))
    n_train = min(n_train, max(1, n_all - 2))
    n_test = min(n_test, max(1, n_all - n_train - 1))
    train = [cands[i] for i in idx[:n_train]]
    test = [cands[i] for i in idx[n_train:n_train + n_test]]
    rest = [cands[i] for i in idx[n_train + n_test:]]
    if n_pool and len(rest) > n_pool:
        rest = rest[:n_pool]
    return train, test, rest


# ---------------------------------------------------------------- 特征与模型

def features(X: np.ndarray) -> np.ndarray:
    """特征：4 个设计角 + |角| + 两组和差（与旧版一致，便于对照）。"""
    t1, t2, t3, t4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.column_stack([X, np.abs(X), t1 + t3, t1 - t3, t2 + t4, t2 - t4])


def train_model(X, y, seed):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(features(X), y)
    return model, time.perf_counter() - t0


# ---------------------------------------------------------------- 单次实验

def run_seed(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed)
    solver = make_solver(cfg["m"])

    t0 = time.perf_counter()
    pool_all = gen_grid(cfg) if cfg["domain"] == "grid" else gen_continuous(cfg, rng)
    t_gen = time.perf_counter() - t0

    n_total = len(pool_all)
    train_c, test_c, pool_c = split_disjoint(
        pool_all, cfg["n_train"], cfg["n_test"], cfg["n_pool"], rng)

    # C2：候选不足要在训练前退出。训练集 <2 无法拟合、测试集 <1 无法评估精度，
    # 都不能硬跑（否则会崩在 features/train_model 里），也不能当成功样本统计。
    if len(train_c) < 2 or len(test_c) < 1:
        return {"seed": int(seed), "status": "insufficient_candidates",
                "counts": {"pool_generated": n_total, "train": len(train_c),
                           "test": len(test_c), "screen_pool": len(pool_c)},
                "model": {"r2_test": float("nan"), "mae_test": float("nan"),
                          "spearman_rho": float("nan"), "pickled_bytes": 0},
                "timing_s": {"generate": t_gen, "label_train": 0.0,
                             "fit": 0.0, "eval_test": 0.0},
                "by_budget": {}}

    # 标注训练集（成本：真实求解次数 = len(train_c)）
    t0 = time.perf_counter()
    tr_eval = [evaluate(t, cfg, solver) for t in train_c]
    t_label = time.perf_counter() - t0

    X_tr = np.array(train_c, float)
    y_tr = np.array([e["lam"] for e in tr_eval], float)

    # 训练集全不可行也要如实记录，不能悄悄跳过
    model, t_train = train_model(X_tr, y_tr, seed)
    model_bytes = len(__import__("pickle").dumps(model))

    # 测试集（模型精度，仅作辅助指标）
    t0 = time.perf_counter()
    te_eval = [evaluate(t, cfg, solver) for t in test_c]
    t_test_eval = time.perf_counter() - t0
    y_te = np.array([e["lam"] for e in te_eval], float)
    if len(test_c) >= 3:
        pred_te = model.predict(features(np.array(test_c, float)))
        from sklearn.metrics import mean_absolute_error, r2_score
        from scipy.stats import spearmanr

        r2 = float(r2_score(y_te, pred_te))
        mae = float(mean_absolute_error(y_te, pred_te))
        rho = float(spearmanr(pred_te, y_te).statistic)
    else:
        r2 = mae = rho = float("nan")

    if not pool_c:
        # 候选池为空（例如均衡过滤后没有幸存者）必须如实记录，不能当成功
        return {"seed": int(seed), "status": "empty_pool",
                "counts": {"pool_generated": n_total, "train": len(train_c),
                           "test": len(test_c), "screen_pool": 0},
                "model": {"r2_test": r2, "mae_test": mae,
                          "spearman_rho": rho, "pickled_bytes": 0},
                "timing_s": {"generate": t_gen, "label_train": t_label,
                             "fit": t_train, "eval_test": t_test_eval},
                "by_budget": {}}

    # 候选池真值（参照，仅评测用；不参与训练与选样）
    t0 = time.perf_counter()
    pool_eval = [evaluate(t, cfg, solver) for t in pool_c]
    t_pool = time.perf_counter() - t0
    pool_X = np.array(pool_c, float)
    pool_lam = np.array([e["lam"] for e in pool_eval], float)
    pool_feas = np.array([e["feasible"] for e in pool_eval], bool)

    # 参照最优必须是「可行」候选的最优：不可行方案的 λ 再大也不能当基准，
    # 否则会造出“代理永远追不上”的假差距。池内无可行解时给 NaN 并计数。
    n_feas_all = int(pool_feas.sum())
    if n_feas_all:
        best_all = float(pool_lam[pool_feas].max())
    else:
        best_all = float("nan")

    # 代理打分（批量预测，不逐条调模型）
    t0 = time.perf_counter()
    pred_pool = model.predict(features(pool_X))
    t_pred = time.perf_counter() - t0

    order_rf = np.argsort(pred_pool)[::-1]

    results = {}
    n_repeat = cfg["n_repeat"]
    for k in cfg["budget"]:
        k_eff = min(int(k), len(pool_c))
        if k_eff <= 0:
            continue

        # --- 代理辅助筛选：按预测排序取前 k，再真实复核
        idx_rf = order_rf[:k_eff]
        lam_rf = pool_lam[idx_rf]
        feas_rf = pool_feas[idx_rf]
        best_rf = float(lam_rf[feas_rf].max()) if feas_rf.any() else float("nan")

        # --- 随机挑选：同样的真实复核次数，重复 n_repeat 次
        # 保留每一次的原始值（未找到可行解记 0），这样才能算"含失败的期望质量"，
        # 而不是只在成功的那些次上取中位数——后者会把失败藏起来。
        rand_vals = []
        for _ in range(n_repeat):
            idx_r = rng.choice(len(pool_c), size=k_eff, replace=False)
            lam_r = pool_lam[idx_r]
            feas_r = pool_feas[idx_r]
            rand_vals.append(float(lam_r[feas_r].max()) if feas_r.any() else 0.0)
        rand_arr = np.array(rand_vals, float)
        rand_fail = int((rand_arr <= 0).sum())
        rand_ok = rand_arr[rand_arr > 0]
        best_rand = float(np.median(rand_ok)) if len(rand_ok) else float("nan")

        def loss(v):
            if np.isnan(v) or np.isnan(best_all):
                return None
            return float((best_all - v) / best_all * 100.0)

        results[str(k_eff)] = {
            "rf": {
                "best_verified_lam": None if np.isnan(best_rf) else best_rf,
                "loss_pct": loss(best_rf),
                "found_feasible": bool(feas_rf.any()),
                "real_evaluations": int(k_eff),
            },
            "random": {
                "best_verified_lam_median": None if np.isnan(best_rand) else best_rand,
                "loss_pct": loss(best_rand),
                "fail_runs": rand_fail,
                "n_repeat": n_repeat,
                "expected_lam": float(rand_arr.mean()),
                "real_evaluations": int(k_eff),
            },
            "reference_full_sweep": {
                "best_lam": None if np.isnan(best_all) else best_all,
                "n_feasible": n_feas_all,
                "real_evaluations": int(len(pool_c)),
            },
        }

    return {
        "seed": int(seed),
        "counts": {"pool_generated": n_total, "train": len(train_c),
                   "test": len(test_c), "screen_pool": len(pool_c)},
        "model": {"r2_test": r2, "mae_test": mae, "spearman_rho": rho,
                  "pickled_bytes": int(model_bytes)},
        "timing_s": {"generate": t_gen, "label_train": t_label,
                     "fit": t_train, "eval_test": t_test_eval,
                     "full_sweep": t_pool, "predict": t_pred},
        "by_budget": results,
    }


# ---------------------------------------------------------------- 入口

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="代理筛选实验（同域同精度、可复现）")
    p.add_argument("--n-plies", type=int, default=10)
    p.add_argument("--nx", type=float, default=500.0)
    p.add_argument("--ny", type=float, default=500.0)
    p.add_argument("--sf", type=float, default=1.5)
    p.add_argument("--m", type=int, default=16, help="Rayleigh-Ritz 级数阶次")
    p.add_argument("--domain", choices=["grid", "continuous"], default="grid")
    p.add_argument("--n-raw", type=int, default=4000, help="连续域原始采样数")
    p.add_argument("--n-pool", type=int, default=3000, help="候选池上限")
    p.add_argument("--n-train", type=int, default=0,
                   help="训练集大小；0 = 按候选池比例自动分配")
    p.add_argument("--n-test", type=int, default=0,
                   help="测试集大小；0 = 按候选池比例自动分配")
    p.add_argument("--budget", default="5,10,20,50,100",
                   help="真实复核次数，逗号分隔")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-repeat", type=int, default=200,
                   help="随机基线在每个预算下的重复次数")
    p.add_argument("--no-balance", action="store_true",
                   help="连续域不施加均衡过滤（对照用）")
    p.add_argument("--angles", default="",
                   help="离散域的角度网格，逗号分隔；默认用工程 5 档（-90,-45,0,45,90）")
    p.add_argument("--out", default="data/proxy_experiment.json")
    return p.parse_args(argv)


def summarize(runs: list) -> dict:
    """跨 seed 汇总：三种统计量分开报，两侧口径一致（审查 A3）。

    - hit_rate        找到可行解的比例
    - mean_score      未命中记 0 后的均值（含失败；是"分数"，不是物理 λ）
    - median_when_hit 只在命中样本上取中位数（质量，不含失败）
    """
    summary = {}
    keys = sorted({k for r in runs for k in r["by_budget"]}, key=int)
    for key in keys:
        rf_all, rf_hit_vals, rf_hit = [], [], 0
        rd_seed_mean, rd_hit_vals, rd_succ, rd_total = [], [], 0, 0
        refs = []
        for r in runs:
            b = r["by_budget"].get(key)
            if not b:
                continue
            rf = b["rf"]
            if rf["found_feasible"]:
                rf_hit += 1
                rf_hit_vals.append(rf["best_verified_lam"])
                rf_all.append(rf["best_verified_lam"])
            else:
                rf_all.append(0.0)
            rd = b["random"]
            rd_succ += rd["n_repeat"] - rd["fail_runs"]
            rd_total += rd["n_repeat"]
            rd_seed_mean.append(rd["expected_lam"])
            if rd.get("best_verified_lam_median") is not None:
                rd_hit_vals.append(rd["best_verified_lam_median"])
            ref = b["reference_full_sweep"]["best_lam"]
            if ref is not None:
                refs.append(float(ref))
        summary[key] = {
            "n_seeds": len(runs),
            "rf_hit_seeds": rf_hit,
            "rf_hit_rate": rf_hit / len(runs),
            "rf_mean_score": float(np.mean(rf_all)) if rf_all else None,
            "rf_median_when_hit": float(np.median(rf_hit_vals)) if rf_hit_vals else None,
            "random_hit_rate": (rd_succ / rd_total if rd_total else None),
            "random_mean_score": float(np.mean(rd_seed_mean)) if rd_seed_mean else None,
            "random_median_when_hit": float(np.median(rd_hit_vals)) if rd_hit_vals else None,
            "reference_full_sweep_best": float(np.mean(refs)) if refs else None,
        }
    return summary


def main(argv=None):
    args = parse_args(argv)
    cfg = {
        "n_plies": args.n_plies, "nx": args.nx, "ny": args.ny, "sf": args.sf,
        "m": args.m, "domain": args.domain,
        "n_raw": args.n_raw, "n_pool": args.n_pool,
        "n_train": args.n_train, "n_test": args.n_test,
        "budget": [int(x) for x in args.budget.split(",") if x.strip()],
        "seeds": [int(x) for x in args.seeds.split(",") if x.strip()],
        "n_repeat": args.n_repeat,
        "angles": [float(x) for x in args.angles.split(",") if x.strip()],
        "enforce_balance": not args.no_balance,
        "min_margin": ad.MIN_MARGIN,
    }

    t_start = time.perf_counter()
    runs = [run_seed(s, cfg) for s in cfg["seeds"]]
    wall = time.perf_counter() - t_start

    summary = summarize(runs)

    out = {
        "config": cfg,
        "code": code_fingerprint(),
        "env": {"numpy": np.__version__, "wall_s": wall},
        "runs": runs,
        "summary": summary,
    }

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 控制台：一张能直接贴进报告的对照表
    print(f"[{cfg['domain']}] N={cfg['n_plies']}  Nx={cfg['nx']:.0f} Ny={cfg['ny']:.0f} "
          f"SF={cfg['sf']}  M={cfg['m']}  seeds={cfg['seeds']}")
    for r in runs:
        c = r["counts"]
        print(f"  seed {r['seed']}: pool={c['pool_generated']} train={c['train']} "
              f"test={c['test']} screen={c['screen_pool']}  "
              f"R²={r['model']['r2_test']:.3f} ρ={r['model']['spearman_rho']:.3f}")
    print(f"  {'K':>5} {'RF命中':>7} {'RF均分':>9} {'随机命中':>8} "
          f"{'随机均分':>9} {'池内最优':>9}")
    for key, s in summary.items():
        rd_hit = s["random_hit_rate"]
        rf_m, rd_m, ref = s["rf_mean_score"], s["random_mean_score"], s["reference_full_sweep_best"]
        print(f"  {key:>5} {s['rf_hit_rate']*100:>7.0f}% "
              f"{rf_m:>9.3f} "
              f"{(rd_hit*100 if rd_hit is not None else float('nan')):>7.1f}% "
              f"{rd_m:>9.3f} "
              f"{(ref if ref is not None else float('nan')):>9.3f}")
    print(f"  用时 {wall:.1f} s，写出 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

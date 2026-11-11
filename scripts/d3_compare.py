"""D3：同预算对照（random / grid / ga）+ 离散完全枚举 + 预算—质量曲线（v2）。

关键口径
  · 可行 = λ ≥ 1.2·SF 且逐层 FI < 1 且对称 + 均衡（与主流程一致）
  · 随机基线与 GA 共享同一评价预算口径：一次候选评价 = 一次 Ncr 求解
  · GA 曲线用 pymoo Callback 逐代记录最优，单次运行 30 代
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize

ROOT = Path(r"F:\AI+力学")
sys.path.insert(0, str(ROOT / "Final_Submission" / "codes"))
import Auto_Design as ad  # noqa: E402

NX, NY, SF = 500.0, 500.0, 1.5
TARGET = max(NX, NY) * SF
N_MAIN = 10
SEEDS = list(range(10))
GEN = 30


def lam_of(theta4, n):
    return float(ad.ncr_equiv(theta4, n, NX, NY, ad._SOLVER_CHK)[1])


def rules_ok(theta4, n):
    r = ad.check_rules(theta4, n)
    return r["symmetric"] and r["balanced"]


def full_enum(n):
    best_t, best_lam, n_eval, n_bal = None, -np.inf, 0, 0
    for a in ad.ANGLE_GRID:
        for b in ad.ANGLE_GRID:
            for c in ad.ANGLE_GRID:
                for d in ad.ANGLE_GRID:
                    t = [a, b, c, d]
                    n_eval += 1
                    if not rules_ok(t, n):
                        continue
                    n_bal += 1
                    lam = lam_of(t, n)
                    if lam > best_lam:
                        best_t, best_lam = t, lam
    return {"n_plies": n, "best_theta4": best_t,
            "best_ncr": round(best_lam * max(NX, NY), 1),
            "best_lam": round(best_lam, 4),
            "best_layup": ad.stacking_text(best_t, n),
            "evaluations": n_eval, "balanced_ok": n_bal,
            "feasible": bool(best_lam >= ad.MIN_MARGIN * SF)}


print("[1/4] 离散完全枚举 ...")
enum8, enum10 = full_enum(8), full_enum(N_MAIN)
print("   N=8 :", enum8["best_ncr"], enum8["best_layup"], "feasible", enum8["feasible"])
print("   N=10:", enum10["best_ncr"], enum10["best_layup"], "feasible", enum10["feasible"])

print("[2/4] 随机搜索：严格可行基线（10 种子 × 2400 次抽样）...")
rng_curves, rng_feasible_rate = [], []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    best, curve, used, n_ok = -np.inf, [], 0, 0
    for budget in (25, 50, 100, 200, 400, 800, 1200, 1600, 2400):
        while used < budget:
            t = rng.uniform(-90.0, 90.0, 4)
            used += 1
            if not rules_ok(t, N_MAIN):
                continue
            n_ok += 1
            best = max(best, lam_of(t, N_MAIN))
        curve.append(round(float(best) * max(NX, NY), 1) if np.isfinite(best) else 0.0)
    rng_curves.append(curve)
    rng_feasible_rate.append(n_ok / used)
print("   均衡可行的抽样占比:", f"{np.mean(rng_feasible_rate) * 100:.1f}%")


class Rec(Callback):
    def __init__(self):
        super().__init__()
        self.best = []

    def notify(self, algorithm):
        f = np.atleast_1d(algorithm.pop.get("F")).ravel()
        self.best.append(float(np.max(-f)))


print("[3/4] GA：逐代最优（10 种子 × 30 代）...")
ga_curves = []
for seed in SEEDS:
    problem = ad.LayupProblem(N_MAIN, NX, NY, TARGET)
    algo = NSGA2(pop_size=ad.GA_POP,
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20, prob=1.0 / 4),
                 eliminate_duplicates=True)
    rec = Rec()
    pymoo_minimize(problem, algo, ("n_gen", GEN), seed=seed, verbose=False,
                   callback=rec)
    ga_curves.append(rec.best)
budgets = [ad.GA_POP * (i + 1) for i in range(GEN)]
ga_mean = [round(float(v), 1) for v in np.mean(ga_curves, axis=0)]
ga_std = [round(float(v), 1) for v in np.std(ga_curves, axis=0)]
print("   GA 末代:", ga_mean[-1], "±", ga_std[-1])

print("[4/4] 单次运行对照（含可行性判定）...")
ad.quick_ga(True)
one_shot = {}
for method in ("random", "grid", "ga"):
    vals, ok, evals = [], [], []
    for seed in SEEDS:
        ad.reset_eval_count()
        cand = ad.optimize_at_n(N_MAIN, NX, NY, SF, seed=seed, method=method)
        vals.append(round(cand["ncr"], 1))
        ok.append(bool(cand["feasible"]))
        evals.append(ad.eval_count())
    one_shot[method] = {"ncr_mean": round(float(np.mean(vals)), 1),
                        "ncr_all": vals,
                        "feasible_rate": round(float(np.mean(ok)), 2),
                        "evals_mean": int(round(float(np.mean(evals))))}
    print(f"   {method:6s} ncr={one_shot[method]['ncr_mean']:8.1f} "
          f"可行率={one_shot[method]['feasible_rate']:.0%} "
          f"评价={one_shot[method]['evals_mean']}")

out = {"case": {"nx": NX, "ny": NY, "sf": SF, "target": TARGET, "n_plies": N_MAIN},
       "enumeration": {"n8": enum8, "n10": enum10},
       "grid_budgets": [ad.GA_POP * (i + 1) for i in range(GEN)],
       "ga_mean": ga_mean, "ga_std": ga_std,
       "random_budgets": [25, 50, 100, 200, 400, 800, 1200, 1600, 2400],
       "random_mean": [round(float(v), 1) for v in np.mean(rng_curves, axis=0)],
       "random_std": [round(float(v), 1) for v in np.std(rng_curves, axis=0)],
       "random_balanced_rate": round(float(np.mean(rng_feasible_rate)), 4),
       "one_shot": one_shot}
p = ROOT / "Final_Submission" / "data" / "d3_compare.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写", p)

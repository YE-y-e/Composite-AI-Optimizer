"""从 proxy_experiment.py 的 runs 重算汇总（口径修正：先报命中率，再报"命中时的质量损失"）。

之所以单独一个脚本：第一次跑出来的 summary 只在"成功的 seed"上取中位数，
把 RF 在 2/5 个 seed 上没找到可行解这件事藏掉了——这正是计划里明令禁止的报法。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def summarize(runs):
    keys = sorted({k for r in runs for k in r["by_budget"]}, key=int)
    out = {}
    for key in keys:
        rf_loss, rf_ok = [], 0
        rd_loss, rd_succ, rd_total = [], 0, 0
        for r in runs:
            b = r["by_budget"].get(key)
            if not b:
                continue
            if b["rf"]["found_feasible"]:
                rf_ok += 1
                if b["rf"]["loss_pct"] is not None:
                    rf_loss.append(b["rf"]["loss_pct"])
            rd = b["random"]
            n_rep = rd["n_repeat"]
            succ = n_rep - rd["fail_runs"]
            rd_succ += succ
            rd_total += n_rep
            if rd["loss_pct"] is not None:
                rd_loss.append(rd["loss_pct"])
        out[key] = {
            "n_seeds": len(runs),
            "rf_hit_seeds": rf_ok,
            "rf_hit_rate": rf_ok / len(runs),
            "rf_loss_pct_median_when_hit": (float(np.median(rf_loss)) if rf_loss else None),
            "random_hit_rate": (rd_succ / rd_total if rd_total else None),
            "random_loss_pct_median_when_hit": (float(np.median(rd_loss)) if rd_loss else None),
        }
    return out


def main(paths):
    for p in paths:
        path = ROOT / p
        d = json.loads(path.read_text(encoding="utf-8"))
        d["summary"] = summarize(d["runs"])
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        cfg = d["config"]
        print(f"== {p}  [{cfg['domain']}] N={cfg['n_plies']} M={cfg['m']} "
              f"seeds={len(d['runs'])}")
        print(f"   {'K':>5} {'RF命中率':>9} {'RF损失(命中时)':>15} "
              f"{'随机命中率':>10} {'随机损失(命中时)':>16}")
        for k, s in d["summary"].items():
            rf = s["rf_loss_pct_median_when_hit"]
            rd = s["random_loss_pct_median_when_hit"]
            print(f"   {k:>5} {s['rf_hit_rate']*100:>8.0f}% "
                  f"{('—' if rf is None else f'{rf:+.2f}%'):>15} "
                  f"{s['random_hit_rate']*100:>9.1f}% "
                  f"{('—' if rd is None else f'{rd:+.2f}%'):>16}")


if __name__ == "__main__":
    main(sys.argv[1:])

# -*- coding: utf-8 -*-
"""补一个稳定的单次求解耗时基准（中位数），避免用一次全扫的墙钟时间代表速度。"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
import Auto_Design as ad          # noqa: E402
import angle_optimize as ao       # noqa: E402

N = 200
rng = np.random.default_rng(7)
X = rng.uniform(-90, 90, size=(N, 4))
solver = ao.BucklingSolver(ao.A_PLATE, ao.B_PLATE, 16)
times = []
for t in X:
    D = ad.d_of_angles(ad.symmetric_stack(list(t), 10))
    t0 = time.perf_counter()
    solver.ncr(D, 500.0, 500.0)
    times.append((time.perf_counter() - t0) * 1000.0)

p = ROOT / "data" / "surrogate_screen.json"
d = json.loads(p.read_text(encoding="utf-8"))
d["timing_benchmark_ms"] = {
    "n": N,
    "median": float(np.median(times)),
    "p90": float(np.percentile(times, 90)),
    "mean": float(np.mean(times)),
}
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("median ms", d["timing_benchmark_ms"])

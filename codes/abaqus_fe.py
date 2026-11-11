"""用模板 .inp 生成并求解一个铺层算例（不依赖 CAE API）。

模板：`abaqus/Job_1_Drone.inp`（50×50 S4R、四边简支、单位面载荷 1 N/mm）。
本模块只做三件事：替换铺层行、按载荷比缩放面载荷、跑 `abaqus job=... interactive` 并读回特征值。

单位与换算：模板里 `*Dsload Surf-1, EDNOR, 1.` 表示 1 N/mm = 1000 N/m；
一阶特征值 λ 即「在当前载荷比下，把载荷放大到 λ 倍时失稳」，
故 Ncr（按 max(Nx, Ny) 计）= λ × 1000 N/m。
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "abaqus" / "Job_1_Drone.inp"
EDNOR_N_PER_MM = 1000.0  # 模板面载荷幅值 1 N/mm 对应的 N/m


def expand_layup(theta4, n_plies):
    """4 个设计角循环填充半叠层再镜像（与 Auto_Design.symmetric_stack 一致）。"""
    if n_plies % 2:
        raise ValueError("层数必须为偶数，收到 N=%d" % n_plies)
    half = [float(theta4[i % 4]) for i in range(n_plies // 2)]
    return half + half[::-1]


def _ply_lines(angles, t_ply_mm, material="T300/5208"):
    out = []
    for i, ang in enumerate(angles, 1):
        out.append("%.6g, 3, %s, %.1f, Ply-%d" % (t_ply_mm, material, ang, i))
    return out


def build_inp(theta4, n_plies, nx, ny, t_ply_mm=0.125, template=TEMPLATE):
    """返回替换好铺层与面载荷的 .inp 文本。"""
    text = Path(template).read_text(encoding="utf-8", errors="replace")
    angles = expand_layup(theta4, n_plies)

    # 1) 定位 *Shell Section, composite 之后连续的单层行
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("*shell section") and "composite" in ln.lower():
            start = i + 1
            break
    if start is None:
        raise RuntimeError("模板里找不到 composite 壳截面")

    end = start
    while end < len(lines) and re.match(r"^\s*[-+0-9.]+\s*,", lines[end]):
        end += 1
    old = lines[start:end]
    if not old:
        raise RuntimeError("composite 截面下没有铺层行")

    lines[start:end] = _ply_lines(angles, t_ply_mm)
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    # 2) 按载荷比缩放 Surf-2（y 向）的面载荷，保留 1 作为 max 方向
    ratio = float(ny) / float(nx) if float(nx) > float(ny) else 1.0
    if float(nx) < float(ny):
        # x 向较小：把 Surf-1 缩到 nx/ny，Surf-2 保持 1
        ratio = float(nx) / float(ny)
        text = _set_dsload(text, "Surf-1", ratio)
    else:
        text = _set_dsload(text, "Surf-2", ratio)
    return text


def _set_dsload(text, surface, magnitude):
    """把 `*Dsload` 之后该面的 EDNOR 幅值改成 magnitude。"""
    pat = re.compile(r"(%s\s*,\s*EDNOR\s*,\s*)([-+0-9.eEdD]+)" % re.escape(surface))
    new, n = pat.subn(lambda m: m.group(1) + ("%.6g" % magnitude), text)
    if n == 0:
        raise RuntimeError("模板里找不到 %s 的面载荷" % surface)
    return new


def run_inp(cmd, inp_text, timeout=1800, keep_dir=False):
    """在临时目录写 .inp 并交互式求解，返回 (特征值, 工作目录)。"""
    workdir = Path(tempfile.mkdtemp(prefix="abq_"))
    job = "verify"
    (workdir / (job + ".inp")).write_text(inp_text, encoding="utf-8")
    proc = subprocess.run([cmd, "job=%s" % job, "interactive"],
                          cwd=str(workdir), capture_output=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    eigen = parse_dat(workdir / (job + ".dat"))
    if eigen is None:
        hint = "" if keep_dir else "（临时目录：%s）" % workdir
        raise RuntimeError("未从 .dat 读到特征值（退出码 %s）%s\n---- 输出尾部 ----\n%s"
                           % (proc.returncode, hint,
                              out.decode("utf-8", "replace")[-1500:]))
    if not keep_dir:
        shutil.rmtree(workdir, ignore_errors=True)
    return eigen, workdir


def parse_dat(path):
    """从 .dat 的「EIGENVALUE OUTPUT」表读第一阶特征值。

    表格形如::

         MODE NO      EIGENVALUE

               1       1.1986
               2       2.5009
    """
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    head = text.find("E I G E N V A L U E")
    block = text[head:] if head >= 0 else text
    for line in block.splitlines():
        m = re.match(r"^\s+1\s+([-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?)\s*$", line)
        if m:
            try:
                val = float(m.group(1).replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if val > 0:
                return val
    return None


# 在 Abaqus 自带 Python 里读 .odb，把一阶模态的节点坐标与 U3 导出成 JSON
_MODE_SCRIPT = r'''
import sys, json
from odbAccess import openOdb

odb = openOdb(sys.argv[-2], readOnly=True)
step = odb.steps[list(odb.steps.keys())[-1]]
# 屈曲步里每个特征模态各占一帧（frameValue 即特征值），取最小的正特征值那一帧
frames = [f for f in step.frames if f.frameValue > 0]
frame = min(frames, key=lambda f: f.frameValue) if frames else step.frames[-1]
U = frame.fieldOutputs["U"]
inst = odb.rootAssembly.instances[list(odb.rootAssembly.instances.keys())[0]]
coords = {n.label: n.coordinates for n in inst.nodes}
rec = []
for v in U.values:
    c = coords.get(v.nodeLabel)
    if c is None:
        continue
    rec.append([float(c[0]), float(c[1]), float(v.data[2])])
with open(sys.argv[-1], "w") as fh:
    json.dump(rec, fh)
odb.close()
'''


def extract_mode(cmd, odb_path, a_mm, b_mm, max_mode=5, timeout=600):
    """从 .odb 读一阶屈曲模态，投影到 sin(mπx/a)·sin(nπy/b) 判断 (m, n)。

    有限元只给特征值与位移场，没有整数模态号；这里用基函数投影取相关度最大的
    (m, n) 作为"最接近的半波数"，并同时返回该相关度供判断拟合是否干净。
    """
    odb_path = Path(odb_path)
    if not odb_path.exists():
        return None
    script = odb_path.parent / "_dump_u3.py"
    script.write_text(_MODE_SCRIPT, encoding="utf-8")
    out = odb_path.parent / "_u3.json"
    proc = subprocess.run([cmd, "python", str(script), str(odb_path), str(out)],
                          cwd=str(odb_path.parent), capture_output=True,
                          timeout=timeout)
    if not out.exists():
        return None
    try:
        rec = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not rec:
        return None

    import numpy as np

    arr = np.array(rec, float)
    x = (arr[:, 0] - arr[:, 0].min()) / max(a_mm, 1e-9) * np.pi
    y = (arr[:, 1] - arr[:, 1].min()) / max(b_mm, 1e-9) * np.pi
    w = arr[:, 2]
    nw = float(np.linalg.norm(w))
    if nw <= 0:
        return None
    best = None
    for mi in range(1, max_mode + 1):
        for ni in range(1, max_mode + 1):
            phi = np.sin(mi * x) * np.sin(ni * y)
            nphi = float(np.linalg.norm(phi))
            if nphi <= 0:
                continue
            corr = abs(float(np.dot(w, phi)) / (nw * nphi))
            if best is None or corr > best["corr"]:
                best = {"m": mi, "n": ni, "corr": corr}
    return best

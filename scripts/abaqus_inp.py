"""兼容层：实现在 codes/abaqus_fe.py（网页按钮与命令行共用同一套逻辑）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))

from abaqus_fe import (  # noqa: E402,F401
    EDNOR_N_PER_MM,
    TEMPLATE,
    build_inp,
    expand_layup,
    parse_dat,
    run_inp,
)

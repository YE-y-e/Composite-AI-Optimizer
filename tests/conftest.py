"""两个测试模块共用的 AppTest 夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "codes" / "app.py"
TIMEOUT = 300


@pytest.fixture(autouse=True)
def demo_mode(monkeypatch):
    """所有用例都在演示模式下跑，不触发 GA。"""
    monkeypatch.setenv("APP_MODE", "demo")


@pytest.fixture
def at():
    test = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    test.run()
    assert not test.exception, [e.value for e in test.exception]
    return test

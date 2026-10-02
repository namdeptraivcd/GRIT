"""Run the original standalone math/data checks as part of the normal test suite."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", sorted(Path(__file__).parent.glob("check_*.py")), ids=lambda p: p.stem)
def test_math_check(script):
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

"""Test fixtures built from the fully fictional data in ``hand_injury_ai.synthetic``.

No real patient data is used anywhere in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hand_injury_ai.config import Paths
from hand_injury_ai.synthetic import write_synthetic_inputs


@pytest.fixture
def synthetic_paths(tmp_path: Path) -> Paths:
    workbook, dictionary = write_synthetic_inputs(tmp_path / "data")
    return Paths(
        workbook=workbook,
        data_dictionary=dictionary,
        output_dir=tmp_path / "outputs",
        reports_dir=tmp_path / "reports",
    )

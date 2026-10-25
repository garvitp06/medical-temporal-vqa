import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def small_patient(tmp_path_factory):
    """One synthetic abnormal patient (two exams) at reduced resolution."""
    from src.data.synthetic import generate_cohort
    out = tmp_path_factory.mktemp("synth")
    man = generate_cohort(out, n_patients=1, frac_abnormal=1.0, seed=3, shape_xy=128)
    return out, man

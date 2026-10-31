"""Shared paths for the sepsis discharge-disposition study.

Derived data and results live inside this project. The credentialed source
databases remain outside it and are located through ``SEPSIS_SOURCE_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_ROOT = Path(
    os.environ.get("SEPSIS_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)

DATA = ROOT / "data" / "processed"
DATA_MIMIC4 = DATA / "mimic4"
DATA_EICU = DATA / "eicu"

RESULTS = ROOT / "results"
RESULTS_MIMIC4 = RESULTS / "mimic4"
RESULTS_EICU = RESULTS / "eicu"
RESULTS_COMPARISON = RESULTS / "comparison"

MIMIC4_SOURCE = SOURCE_ROOT / "mimic-iv-3.1"
EICU_SOURCE = SOURCE_ROOT / "eicu"

for _path in (
    DATA_MIMIC4,
    DATA_EICU,
    RESULTS_MIMIC4,
    RESULTS_EICU,
    RESULTS_COMPARISON,
):
    _path.mkdir(parents=True, exist_ok=True)

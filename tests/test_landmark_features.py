from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("module_name", "id_column", "last_column"),
    [
        ("sepsis.build_landmarks_mimic4", "stay_id", "hr_w1_last"),
        ("sepsis.build_landmarks_eicu", "patientunitstayid", "hr_w1_last"),
        ("sepsis.build_mimic3_sensitivity", "icustay_id", "hr_w1_last"),
    ],
)
def test_last_feature_uses_chronological_order(
    tmp_path, monkeypatch, module_name, id_column, last_column
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "OUT", tmp_path)

    rows = []
    for hh, hr in ((20, 80.0), (22, 100.0), (23, 110.0), (4, 90.0)):
        rows.append(
            {
                id_column: 7,
                "hh": hh,
                "hr": hr,
                "rr": 18.0,
                "spo2": 95.0,
                "sbp": 100.0,
                "dbp": 60.0,
                "mbp": 73.0,
                "tempf": 98.6,
                "gcs": 15.0,
            }
        )
    hourly = pd.DataFrame(rows)

    features = module.build_features(hourly)

    assert features.loc[0, last_column] == 110.0
    assert np.isfinite(features.loc[0, "hr_w1_mean"])

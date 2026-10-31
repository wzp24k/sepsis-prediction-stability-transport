# -*- coding: utf-8 -*-
"""Rebuild eICU complete-window design and sequence views from aligned sources."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("SEPSIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "data" / "processed" / "eicu"


def main() -> None:
    design_all = pd.read_parquet(OUT / "design_all.parquet")
    complete = design_all.loc[
        design_all["y"].notna() & (design_all["los_h"] >= 72)
    ].copy()
    complete.to_parquet(OUT / "design_v3.parquet", index=False)

    source = np.load(OUT / "seq_all72.npz", allow_pickle=True)
    arrays = {key: source[key] for key in source.files}
    source_ids = arrays["patientunitstayid"].astype(np.int64)
    lookup = {int(identifier): index for index, identifier in enumerate(source_ids)}
    target_ids = complete["patientunitstayid"].astype(np.int64).to_numpy()
    if len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("eICU complete design contains duplicate stays")
    index = np.asarray([lookup[int(identifier)] for identifier in target_ids], dtype=int)
    selected = {key: value[index] for key, value in arrays.items()}
    if not np.array_equal(
        selected["patientunitstayid"].astype(np.int64), target_ids
    ):
        raise ValueError("eICU complete design and sequence rows are misaligned")
    expected_y3 = complete["y3"].to_numpy(dtype=np.int64)
    if not np.array_equal(selected["y3"].astype(np.int64), expected_y3):
        raise ValueError("eICU complete design and sequence outcomes disagree")
    np.savez_compressed(OUT / "seq72.npz", **selected)
    print(
        {
            "design_v3": complete.shape,
            "seq72": selected["X"].shape,
            "class_counts": {
                str(key): int(value)
                for key, value in complete["y3"].value_counts().sort_index().items()
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

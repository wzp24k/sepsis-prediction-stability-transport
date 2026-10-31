| Database | Model / validation | Changed predictions, n | 24->72 h class change, % | Non-death to death conversion, % | Home to facility, % | Ordinal-distance closer / farther / tie among changed predictions, % | Higher eventual-class probability, % | Mean change in eventual-class probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIMIC-IV | XGBoost, shared-fold | 1,591 | 23.32 | 10.22 | 2.46 | 56.88 / 41.04 / 2.07 | 57.57 | +0.05 |
| MIMIC-IV | Ordinal DG-GRU, no temporal | 818 | 11.99 | 4.34 | 0.62 | 60.39 / 39.36 / 0.24 | 60.51 | +0.03 |
| MIMIC-IV | Ordinal DG-GRU, temporal | 786 | 11.52 | 4.40 | 0.45 | 60.56 / 39.31 / 0.13 | 60.56 | +0.03 |
| eICU-CRD | XGBoost, shared-fold | 1,845 | 20.02 | 6.71 | 4.72 | 56.10 / 41.63 / 2.28 | 57.40 | +0.05 |
| eICU-CRD | Ordinal DG-GRU, no temporal | 1,538 | 16.69 | 2.69 | 2.63 | 58.58 / 39.53 / 1.89 | 59.62 | +0.03 |
| eICU-CRD | Ordinal DG-GRU, temporal | 1,468 | 15.93 | 2.55 | 2.49 | 58.92 / 39.51 / 1.57 | 59.81 | +0.03 |
| MIMIC-III | XGBoost, same-institution cross-era | 876 | 24.08 | 9.92 | 3.02 | 55.02 / 42.58 / 2.40 | 55.82 | +0.05 |

| Database | Analysis | Class change, % | Non-death to death conversion, % / pp | home to facility, % / pp | Closer / farther / equal among all stays, %; temporal category change, pp | Flip-level mean true-class probability change | Flip-level true-class probability improved, % / pp | Death subgroup change, pp (new conversion / death not predicted) |
|---|---|---|---|---|---|---|---|---|
| MIMIC-IV | XGBoost, shared-fold | 23.32 | 10.22 | 2.46 | 13.26 / 9.57 / 77.17 | +0.05 | 57.57 |  |
| MIMIC-IV | Ordinal DG-GRU, no temporal penalty | 11.99 | 4.34 | 0.62 | 7.24 / 4.72 / 88.04 | +0.03 | 60.51 |  |
| MIMIC-IV | Ordinal DG-GRU, temporal penalty | 11.52 | 4.40 | 0.45 | 6.98 / 4.53 / 88.49 | +0.03 | 60.56 |  |
| MIMIC-IV | Temporal minus no temporal (pp) | -0.47 | +0.06 | -0.16 | -0.10 / 0.00 / -0.16 / -0.21 (closer-higher / farther-higher / closer-lower / farther-lower) | -0.01 | +0.05 | -0.14 / +0.05 |
| eICU-CRD | XGBoost, shared-fold | 20.02 | 6.71 | 4.72 | 11.23 / 8.33 / 80.43 | +0.05 | 57.40 |  |
| eICU-CRD | Ordinal DG-GRU, no temporal penalty | 16.69 | 2.69 | 2.63 | 9.78 / 6.60 / 83.62 | +0.03 | 59.62 |  |
| eICU-CRD | Ordinal DG-GRU, temporal penalty | 15.93 | 2.55 | 2.49 | 9.39 / 6.29 / 84.32 | +0.03 | 59.81 |  |
| eICU-CRD | Temporal minus no temporal (pp) | -0.76 | -0.14 | -0.14 | -0.15 / -0.13 / -0.24 / -0.24 (closer-higher / farther-higher / closer-lower / farther-lower) | -0.24 | +0.19 | -0.52 / +0.58 |

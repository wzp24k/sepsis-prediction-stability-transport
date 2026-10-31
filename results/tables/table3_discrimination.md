| Database | Model / validation design | Macro-AUC at 24 h | Macro-AUC at 48 h | Macro-AUC at 72 h |
| --- | --- | --- | --- | --- |
| MIMIC-IV | XGBoost, patient-grouped | 0.71 | 0.73 | 0.75 |
| eICU-CRD | XGBoost, patient-grouped | 0.67 | 0.69 | 0.71 |
| eICU-CRD | XGBoost, hospital-held-out | 0.66 | 0.68 | 0.70 |
| MIMIC-III | XGBoost, same-institution cross-era | 0.70 | 0.72 | 0.74 |
| MIMIC-IV | DG-GRU ordinal + temporal | 0.67 | 0.69 | 0.70 |
| eICU-CRD | DG-GRU ordinal + temporal | 0.64 | 0.66 | 0.68 |

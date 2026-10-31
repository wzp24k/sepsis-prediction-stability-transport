| Database | Model | 72 h macro-AUC | Accuracy | Log loss | Brier score | Top-label ECE |
|---|---|---|---|---|---|---|
| MIMIC-IV | DG-GRU ordinal, no temporal penalty | 0.70 | 0.56 | 0.91 | 0.55 | 0.02 |
| MIMIC-IV | DG-GRU ordinal + temporal penalty | 0.70 | 0.56 | 0.91 | 0.55 | 0.02 |
| MIMIC-IV | DG-GRU nominal, no temporal penalty | 0.67 | 0.54 | 0.94 | 0.57 | 0.01 |
| MIMIC-IV | DG-GRU nominal + temporal penalty | 0.67 | 0.55 | 0.94 | 0.57 | 0.01 |
| eICU-CRD | DG-GRU ordinal, no temporal penalty | 0.68 | 0.54 | 0.95 | 0.57 | 0.02 |
| eICU-CRD | DG-GRU ordinal + temporal penalty | 0.68 | 0.54 | 0.95 | 0.57 | 0.02 |
| eICU-CRD | DG-GRU nominal, no temporal penalty | 0.65 | 0.52 | 0.97 | 0.59 | 0.0064 |
| eICU-CRD | DG-GRU nominal + temporal penalty | 0.65 | 0.52 | 0.97 | 0.59 | 0.0066 |

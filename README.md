# Predicted-Class Stability and Transport of Discharge-Disposition Models Among ICU Stays With Sepsis at 24, 48, and 72 Hours

This public repository accompanies a three-database study of predicted
discharge-destination stability among ICU stays with sepsis. It contains
analysis code, the fixed study configuration, aggregate result tables, figures,
and verification records. It does not contain source database records,
patient-level data, or manuscript files.

## Study scope

- Databases: MIMIC-IV 3.1, eICU-CRD 2.0, and MIMIC-III 1.4.
- Landmarks: ICU hours 24, 48, and 72.
- Outcomes: home discharge, facility discharge, and in-hospital death.
- Primary estimand: the proportion of model-predicted classes that change
  through 72 h.
- Supporting analyses: fixed-weight transport, local recalibration, and a
  retrospective enhanced-monitoring trigger simulation.

The cohort is conditional on observation through 72 h. The three landmarks
estimate instability through 72 h. They do not estimate an exact time to
stability. All reported transitions are changes in predicted classes, not
observed patient transitions.

## Evidence boundaries

- A lower DG-GRU class-flip rate does not establish better clinical
  reliability. Discrimination and probability movement are reported alongside
  flip frequency.
- Temporal regularization has no clinical validation.
- The home-facility-death ordinal order is an analytic assumption.
- MIMIC-III is a same-institution cross-era replication. Its source period
  overlaps MIMIC-IV.
- eICU-CRD hospital-held-out analysis is internal-external validation.
- Fixed-weight transport is a retrospective experiment, not prospective
  deployment.
- The reassessment simulation does not establish benefits to workflow,
  resources, or patient outcomes.

## Repository contents

- `src/sepsis/`: cohort construction, modelling, analysis, assembly, and
  validation code.
- `configs/study.json`: fixed study configuration.
- `tests/`: focused tests for landmark features and causal DG-GRU structure.
- `results/tables/`: main and supplementary aggregate tables.
- `results/figures/`: main figures in PNG, PDF, and SVG.
- `results/*/summary.json`: selected aggregate result summaries.
- `outputs/reference_verification.json`: DOI verification record for cited
  references.

## Reproduction

Install Python 3.10 and the packages in `requirements.txt`.

The analysis expects credentialed local copies of MIMIC-IV, eICU-CRD, and
MIMIC-III. Set the source root before running data-building scripts:

```powershell
$env:SEPSIS_PROJECT_ROOT = (Get-Location).Path
$env:SEPSIS_SOURCE_ROOT = "C:\path\to\source-data"
```

Run the focused tests with:

```powershell
python -m pytest -q
```

The full manuscript build and manuscript-level validation are retained in the
private working project and are intentionally excluded from this public code
repository.

## Data access

MIMIC-IV, eICU-CRD, and MIMIC-III are available from PhysioNet to credentialed
users under their data use agreements. Users must obtain independent access.
This repository does not redistribute the source databases, derived
patient-level files, or records that could identify a patient.

## Verification

The released snapshot passed the complete manuscript and artifact validator.
It also passed the unit tests and DOI checks for all 48 references. Aggregate
tables in the assembled Markdown and Word manuscript were compared cell by
cell.

## Citation

Citation metadata are provided in `CITATION.cff`. Repository access:
https://github.com/wzp24k/sepsis-prediction-stability-transport

No open-source license has been selected for this repository. Contact the
corresponding author regarding reuse.
